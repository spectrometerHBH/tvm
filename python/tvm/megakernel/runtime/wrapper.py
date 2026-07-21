# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Megakernel lifecycle framework.

Migrated from the production ``tirx_kernels.megakernel.utils.base.MegaKernelWrapper``.
The wrapper manages ``tvm.megakernel.dsl.TileImpl`` objects and drives them
with the DSL hook vocabulary:

- ``class_init_all``     -> per-class ``TileImpl.init_shared_resources``
- ``class_finalize_all`` -> per-class ``TileImpl.finalize_shared_resources``
- ``device_init_all``    -> per-instance ``TileImpl.device_init`` (called once
  at kernel-entry registration time, before the scheduler exists, so the
  task indices are ``(0, 0, 0)``)
- ``host_init_all``      -> per-instance ``TileImpl.host_init``
- ``run_tile``/``run_tile_prefetch`` -> ``TileImpl.run``/``TileImpl.prefetch``

The module-assembly path of the production wrapper (``get_module`` /
``get_func_static`` / ``get_func_dynamic``) is intentionally excluded:
emitting a module from a spec belongs to the build layer, not the runtime.
"""

from __future__ import annotations

import functools

from tvm.script import tirx as T
from tvm.tirx.bench import CudaProfiler
from tvm.tirx.expr import Var

from ..dsl import TileImpl
from .config import HardwareConfig, RuntimeProfileEvent
from .device import any_sync, f_init_const
from .semaphore import SemaphoreBase
from .smem import SmemManager


class _InitETensorTile(TileImpl):
    """Internal tile that initializes the event tensors of one kernel."""

    VEC_SIZE = 1

    def __init__(self, etensor_and_f_init_pairs, hardware: HardwareConfig | None = None):
        super().__init__()
        self.etensor_and_f_init_pairs = etensor_and_f_init_pairs
        self.total_num_etensors = len(etensor_and_f_init_pairs)
        self.hardware = hardware or HardwareConfig()

    def convert_1d_index_to_nd(self, idx, shape):
        nd_idx = []
        for i in reversed(range(len(shape))):
            nd_idx.append(idx % shape[i])
            idx = idx // shape[i]
        return list(reversed(nd_idx))

    def run(self, m_idx, n_idx, k_idx):
        tid = T.thread_id([self.hardware.num_threads])
        if_frames = [T.If(m_idx == i) for i in range(self.total_num_etensors)]
        then_frames = [T.Then() for i in range(self.total_num_etensors)]
        else_frames = [T.Else() for i in range(self.total_num_etensors - 1)]
        idx = T.alloc_local([1], "int32")
        T.buffer_store(idx, tid * self.VEC_SIZE, [0])
        for i in range(self.total_num_etensors):
            if_frames[i].__enter__()
            with then_frames[i]:
                etensor, f_init = self.etensor_and_f_init_pairs[i]
                if f_init is None:
                    T.evaluate(0)
                else:
                    nelem = functools.reduce(lambda x, y: x * y, etensor.shape, 1)
                    etensor_1d = etensor.view(-1)
                    with T.While(idx[0] < nelem):
                        with T.vectorized(self.VEC_SIZE) as v:
                            T.buffer_store(
                                etensor_1d,
                                f_init(*self.convert_1d_index_to_nd(idx[0] + v, etensor.shape))
                                * (SemaphoreBase.base + 1),
                                idx[0] + v,
                            )
                        T.buffer_store(idx, idx[0] + self.hardware.num_threads * self.VEC_SIZE, [0])
            if i < self.total_num_etensors - 1:
                else_frames[i].__enter__()
        for i in range(self.total_num_etensors - 1, -1, -1):
            if i < self.total_num_etensors - 1:
                else_frames[i].__exit__(None, None, None)
            if_frames[i].__exit__(None, None, None)


class MegaKernelWrapper:
    """Base class for megakernel wrappers."""

    ETENSOR_WORKSPACE_SIZE = 1024 * 1024
    PROFILER_BUFFER_SIZE = int(10000000.0)

    def __init__(
        self,
        config: dict | None = None,
        tp_size: int = 1,
        profiler_on: bool = False,
        hardware: HardwareConfig | None = None,
        prefetch_event=None,
        init_etensor_event=None,
        wait_etensor_init_event=None,
    ):
        self.tp_size = tp_size
        self.config = {} if config is None else config
        self.profiler_on = profiler_on
        self.hardware = hardware or HardwareConfig()
        self.NUM_GROUPS = self.hardware.warp_count
        self.PROFILER_WRITE_STRIDE = self.hardware.sm_count * self.NUM_GROUPS
        self.prefetch_event = prefetch_event or RuntimeProfileEvent.PREFETCH
        self.init_etensor_event = init_etensor_event or RuntimeProfileEvent.INIT_ETENSOR
        self.wait_etensor_init_event = (
            wait_etensor_init_event or RuntimeProfileEvent.WAIT_ETENSOR_INIT
        )
        self.tile_attr = {}
        self.class_list = set()
        self.etensor_and_f_init_pairs = []
        self.num_etensors = {}
        self.etensor_workspace_offset = 0

    def _init_profiler(self, profiler_buffer):
        if self.profiler_on:
            self.profiler = CudaProfiler(
                profiler_buffer, write_stride=self.PROFILER_WRITE_STRIDE, num_groups=self.NUM_GROUPS
            )
        else:
            self.profiler = None

    def _init_tile_scheduler(self, scheduler_class, *args):
        self.tile_scheduler = scheduler_class(*args)

    def _add_tile(self, tile, profiler_event_type, predicate=True):
        self.tile_attr[tile] = (profiler_event_type, predicate)
        self.class_list.add(tile.__class__)
        return tile

    @T.inline
    def init_profiler(self, profiler_buffer):
        self._init_profiler(profiler_buffer)
        warp_id = T.warp_id([self.hardware.warp_count])
        if self.profiler_on:
            self.profiler.init(warp_id)

    def set_smem_manager(self, smem_max_bytes, chunk_size, ptr: Var):
        self.smem_manager = SmemManager(smem_max_bytes, chunk_size, ptr, hardware=self.hardware)

    @T.inline
    def init_tile_scheduler(self, is_dynamic_sch, scheduler_class, *args):
        self._init_tile_scheduler(scheduler_class, *args)
        self.tile_scheduler.init()
        if is_dynamic_sch:
            self.tile_scheduler.next_tile()

    @T.inline
    def run_tile(self, tile: TileImpl, *args, **kwargs):
        event_type = T.meta_var(self.tile_attr[tile][0])
        self.smem_manager.enter_tile_runtime(tile)
        lane_id = T.lane_id([self.hardware.warp_size])
        if self.profiler_on:
            self.profiler.start(event_type, lane_id == 0)
        tile.run(*args, **kwargs)
        if self.profiler_on:
            self.profiler.end(event_type, lane_id == 0)

    @T.inline
    def run_tile_prefetch(self, tile: TileImpl, *args):
        self.smem_manager.enter_tile_runtime(tile)
        lane_id = T.lane_id([self.hardware.warp_size])
        if self.profiler_on:
            self.profiler.start(self.prefetch_event, lane_id == 0)
        tile.prefetch(*args)
        if self.profiler_on:
            self.profiler.end(self.prefetch_event, lane_id == 0)

    def add_etensor(self, sem_class, etensor_workspace, shape, f_init):
        size = functools.reduce(lambda x, y: x * y, shape, 1)
        etensor_buffer = T.decl_buffer(
            shape, "int32", etensor_workspace.data, elem_offset=self.etensor_workspace_offset
        )
        self.etensor_workspace_offset += size
        etensor = sem_class(etensor_buffer)
        self.etensor_and_f_init_pairs.append((etensor_buffer, f_init))
        return etensor

    def set_events_complete(
        self, is_dynamic_sch, Semaphore: type[SemaphoreBase], etensor_workspace_global
    ):
        if not is_dynamic_sch:
            num_evtensors = len(self.etensor_and_f_init_pairs)
            self.evt_etensor_init_complete = self.add_etensor(
                Semaphore,
                etensor_workspace_global,
                shape=[1],
                f_init=f_init_const(num_evtensors + 1 + self.hardware.sm_count),
            )
        else:
            self.evt_etensor_init_complete = None
        self.init_etensor_tile = self._add_tile(
            _InitETensorTile(self.etensor_and_f_init_pairs, self.hardware),
            self.init_etensor_event,
        )

    @T.inline
    def task_impl_init_etensor(self, is_dynamic_sch):
        self.run_tile(
            self.init_etensor_tile,
            self.tile_scheduler.m_idx,
            self.tile_scheduler.n_idx,
            self.tile_scheduler.k_idx,
        )
        if self.evt_etensor_init_complete is not None:
            if self.tile_scheduler.m_idx < len(self.etensor_and_f_init_pairs):
                self.tile_scheduler.notify(
                    self.evt_etensor_init_complete,
                    lambda notify_idx: (1, -1, 0),
                    scope="cta",
                    release=True,
                )

    @T.inline
    def task_impl_wait_etensor_init_complete(self, is_dynamic_sch):
        if not is_dynamic_sch:
            warp_id = T.warp_id([self.hardware.warp_count])
            lane_id = T.lane_id([self.hardware.warp_size])
            if self.profiler_on:
                self.profiler.start(self.wait_etensor_init_event, lane_id == 0)
            state = T.alloc_local([1], "int32")
            state[0] = -1
            while 1:
                if lane_id == 0:
                    T.ptx.ld_global_acquire(
                        state[0], self.evt_etensor_init_complete.sem.ptr_to([0])
                    )
                if any_sync(
                    self.hardware.full_mask,
                    state[0] <= self.hardware.sm_count * (SemaphoreBase.base + 1) and state[0] > 0,
                ):
                    if (lane_id == 0) & (warp_id == 0):
                        T.cuda.atomic_add(
                            self.evt_etensor_init_complete.sem.ptr_to([0]),
                            -(SemaphoreBase.base + 1),
                        )
                    break
                T.cuda.nano_sleep(40)
            if self.profiler_on:
                self.profiler.end(self.wait_etensor_init_event, lane_id == 0)

    def reset(self):
        self.tile_attr = {}
        self.class_list = set()
        self.etensor_and_f_init_pairs = []
        self.etensor_workspace_offset = 0

    def host_init_all(self):
        for tile, (_, predicate) in self.tile_attr.items():
            if predicate:
                tile.host_init()

    def class_init_all(self, smem_manager: SmemManager):
        for cls in self.class_list:
            if getattr(cls, "need_init", True):
                smem_manager.set_tile(cls)
                cls.init_shared_resources(smem_manager)

    def class_finalize_all(self, smem_manager: SmemManager):
        for cls in self.class_list:
            if getattr(cls, "need_init", True):
                cls.finalize_shared_resources(smem_manager)

    def device_init_all(self, smem_manager: SmemManager):
        for tile, (_, predicate) in self.tile_attr.items():
            if predicate:
                smem_manager.set_tile(tile)
                tile.device_init(smem_manager, 0, 0, 0)


__all__ = ["MegaKernelWrapper"]
