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
"""Dynamic tile scheduler (MPMC queue) for megakernel.

Migrated from the production
``tirx_kernels.megakernel.utils.dynamic_scheduler``.  See
``tvm.megakernel.runtime.semaphore`` for the two-phase counter protocol
notes that drive the pre-notify/pre-push logic below.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from tvm.script import tirx as T
from tvm.tirx.bench import CudaProfiler

from .config import HardwareConfig, RuntimeProfileEvent
from .device import (
    any_sync,
    atomic_add_int32,
    gt,
    stg,
    sts,
    while_ld_global_acquire,
)
from .packing import pack_into_32bit, unpack_from_32bit
from .semaphore import SemaphoreBase
from .tile import Barriers, TileSchedulerBase


class Semaphore(SemaphoreBase):
    def __init__(self, buffer, debug=False, hardware: HardwareConfig | None = None):
        self.sem = buffer
        self.hardware = hardware or HardwareConfig()
        self.state = T.alloc_local([1], "int32")

        # cta-level interface

    @T.inline
    def semaphore_wait(self, *coord, level: Literal["cta", "warp"] = "cta", mask=0xFFFFFFFF):
        if level == "cta":
            while 1:
                T.ptx.ld_global_acquire(
                    self.state[0], self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord))
                )
                if T.cuda.syncthreads_and(self.state[0] == 0):
                    break
                T.cuda.nano_sleep(40)
        elif level == "warp":
            warp_id = T.warp_id([self.hardware.warp_count])
            lane_id = T.lane_id([self.hardware.warp_size])
            if (mask >> warp_id) & 1 == 1:
                self.state[0] = -1
                while 1:
                    if lane_id == 0:
                        T.ptx.ld_global_acquire(
                            self.state[0],
                            self.sem.access_ptr("r", offset=self.sem.elem_offset_of(coord)),
                        )
                    if any_sync(self.hardware.full_mask, self.state[0] == 0):
                        break
                    T.cuda.nano_sleep(40)
        else:
            assert False

    @T.inline
    def semaphore_notify(self, *coord, pre_notify=False, release=False):
        number = T.meta_var(1 if pre_notify else self.base)
        # the old value will be stored in self.state
        self.state[0] = atomic_add_int32(self.sem.ptr_to(coord), -number, release=release)
        if self.state[0] <= 0:
            while 1:
                T.ptx.ld_global_acquire(self.state[0], self.sem.ptr_to(coord))
                if gt(self.state[0], 0):
                    self.state[0] = atomic_add_int32(
                        self.sem.ptr_to(coord), -number, release=release
                    )
                    break
                sleep_time = T.meta_var(800 if pre_notify else 40)
                T.cuda.nano_sleep(sleep_time)

    def is_triggered(self):
        return self.state[0] % self.base == 1


class SchedulerBarrier(Barriers):
    def __init__(self, smem_manager, is_p2c):
        super().__init__(smem_manager, 1, is_p2c)

    @T.inline
    def arrive(self):
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([0]))


@T.meta_class
class MPMCQueue:
    def __init__(
        self,
        capacity: int,
        tasks: T.Buffer,
        head: T.Buffer,
        tail: T.Buffer,
        smem_manager,
        debug=False,
    ):
        # TODO: we currently assume that the queue is infinitely large.
        if capacity & (capacity - 1):
            raise ValueError("capacity must be a power-of-two")
        self.capacity = capacity
        self.mask = capacity - 1
        self.tasks = tasks  # an array of (task_type, m_idx, n_idx, k_idx)
        self.head = head
        self.tail = tail
        self.debug = debug
        self.smem_manager = smem_manager
        self.hardware = smem_manager.hardware

    def _alloc(self):
        self.head_r = T.local_scalar(dtype="int32")
        self.tail_r = T.local_scalar(dtype="int32")
        self.masked_pos = T.local_scalar(dtype="int32")
        self.idx = T.local_scalar(dtype="int32")
        self.tail_smem = self.smem_manager.alloc(
            (self.hardware.warp_count,), "int32", policy="persistent"
        )

    @T.inline
    def init(self):
        self._alloc()

    @T.inline
    def enqueue(self, func_push, level: Literal["thread", "warp", "warpgroup", "cta"]):
        if level == "thread":
            task_type, enqueue_num, m_idx, n_idx, k_idx = func_push(0)
            if self.debug:
                T.cuda.trap_when_assert_failed(enqueue_num == 1)  # notes: enqueue_num must be 1
            self.tail_r = atomic_add_int32(
                self.tail.access_ptr("rw", offset=self.tail.elem_offset_of([T.int32(0)])), 1
            )
            self.masked_pos = self.tail_r & self.mask
            task_info = T.meta_var(
                pack_into_32bit(m_idx, n_idx, k_idx, task_type, host=False, debug=self.debug)
            )
            stg(
                task_info,
                self.tasks.access_ptr("rw", offset=self.tasks.elem_offset_of([self.masked_pos])),
            )
        else:
            lane_id = T.lane_id([self.hardware.warp_size])
            tid_in_wg = T.thread_id_in_wg([self.hardware.warpgroup_size])
            tid = T.thread_id([self.hardware.num_threads])
            wg_id = T.warpgroup_id([self.hardware.warpgroup_count])
            warp_id = T.warp_id([self.hardware.warp_count])
            idx_map = T.meta_var(
                {
                    "warp": (warp_id, lane_id, self.hardware.warp_size),
                    "warpgroup": (
                        wg_id,
                        tid_in_wg,
                        self.hardware.warpgroup_size,
                    ),
                    "cta": (0, tid, self.hardware.num_threads),
                }
            )
            scope_idx, tid_in_scope, tid_stride = idx_map[level]
            enqueue_num = func_push(0)[1]
            if level == "warp":
                if tid_in_scope == 0:
                    self.tail_r = atomic_add_int32(
                        self.tail.access_ptr("rw", offset=self.tail.elem_offset_of([T.int32(0)])),
                        enqueue_num,
                    )
                self.tail_r = T.tvm_warp_shuffle(
                    self.hardware.full_mask,
                    self.tail_r,
                    0,
                    self.hardware.warp_size,
                    self.hardware.warp_size,
                )
            else:
                if tid_in_scope == 0:
                    self.tail_smem[scope_idx] = atomic_add_int32(
                        self.tail.access_ptr("rw", offset=self.tail.elem_offset_of([T.int32(0)])),
                        enqueue_num,
                    )
                if level == "warpgroup":
                    T.ptx.bar.sync(6 + wg_id, self.hardware.warpgroup_size)
                elif level == "cta":
                    T.tvm_storage_sync("shared")
                self.tail_r = self.tail_smem[scope_idx]

            self.idx = tid_in_scope
            while self.idx < enqueue_num:
                self.masked_pos = (self.tail_r + self.idx) & self.mask
                task_type, _, m_idx, n_idx, k_idx = func_push(self.idx)
                task_info = T.meta_var(
                    pack_into_32bit(m_idx, n_idx, k_idx, task_type, host=False, debug=self.debug)
                )
                stg(
                    task_info,
                    self.tasks.access_ptr(
                        "rw", offset=self.tasks.elem_offset_of([self.masked_pos])
                    ),
                )
                self.idx += tid_stride

    @T.inline
    def dequeue(self, fetched_task_info):
        self.head_r = T.cuda.atomic_add(
            self.head.access_ptr("rw", offset=self.head.elem_offset_of([T.int32(0)])), 1
        )
        self.masked_pos = self.head_r & self.mask
        while_ld_global_acquire(
            self.tasks.access_ptr("r", offset=self.tasks.elem_offset_of([self.masked_pos])),
            T.address_of(fetched_task_info),
        )
        # FIXME: enable this when we consider capacity issue
        # self.tasks[self.masked_pos, 0] = -1


class DynamicTileScheduler(TileSchedulerBase):
    MAX_TASKS = 32768

    def __init__(
        self,
        tasks: T.Buffer,
        head: T.Buffer,
        tail: T.Buffer,
        smem_manager,
        profiler: CudaProfiler = None,
        debug=False,
        end_task_type: int = 31,
        fetch_event=None,
        push_event=None,
    ):
        self.queue = MPMCQueue(
            capacity=self.MAX_TASKS, tasks=tasks, head=head, tail=tail, smem_manager=smem_manager
        )
        self.profiler_on = profiler is not None
        self.profiler = profiler
        self.debug = debug
        self.smem_manager = smem_manager
        self.hardware = smem_manager.hardware
        self.scheduler_warp = self.hardware.warp_count - 1
        self.end_task_type = end_task_type
        self.fetch_event = fetch_event or RuntimeProfileEvent.FETCH
        self.push_event = push_event or RuntimeProfileEvent.PUSH

    def _alloc(self):
        self.task_info = T.local_scalar(dtype="int32")
        self.task_type = T.local_scalar(dtype="int32")
        self.m_idx = T.local_scalar(dtype="int32")
        self.n_idx = T.local_scalar(dtype="int32")
        self.k_idx = T.local_scalar(dtype="int32")
        self.idx = T.local_scalar(dtype="int32")
        self.dequeue_phase = T.local_scalar(dtype="int32")
        self.p2c_dequeue_barrier = SchedulerBarrier(self.smem_manager, is_p2c=True)
        self.c2p_dequeue_barrier = SchedulerBarrier(self.smem_manager, is_p2c=False)
        self.packed_value = self.smem_manager.alloc((1,), "int32", align=16, policy="persistent")
        self.semaphore_state = self.smem_manager.alloc(
            (self.hardware.num_threads,), "int32", policy="persistent"
        )

    @T.inline
    def _dequeue_and_store_packed(self):
        self.queue.dequeue(self.task_info)
        sts(self.task_info, self.packed_value.ptr_to([0]))

    @T.inline
    def _fetch_from_queue(self):
        warp_id = T.warp_id([self.hardware.warp_count])
        # fetch from GEMM queue
        if warp_id == self.scheduler_warp:
            if T.ptx.elect_sync():
                self.c2p_dequeue_barrier.wait(0, self.dequeue_phase)
                self._dequeue_and_store_packed()
                self.p2c_dequeue_barrier.arrive()
        self.p2c_dequeue_barrier.wait(0, self.dequeue_phase)
        unpack_from_32bit(
            self.packed_value[0],
            T.address_of(self.task_type),
            T.address_of(self.m_idx),
            T.address_of(self.n_idx),
            T.address_of(self.k_idx),
        )
        self.c2p_dequeue_barrier.arrive()
        self.dequeue_phase = self.dequeue_phase ^ 1

    @T.inline
    def init(self):
        tid = T.thread_id([self.hardware.num_threads])
        self._alloc()
        self.queue.init()
        self.dequeue_phase = 0
        if tid == 0:
            self.p2c_dequeue_barrier.init(1)
            self.c2p_dequeue_barrier.init(self.hardware.num_threads)
        T.tvm_storage_sync("shared")
        T.ptx.fence.proxy_async("shared::cta")
        T.ptx.fence.mbarrier_init()

    @T.inline
    def next_tile(self):
        lane_id = T.lane_id([self.hardware.warp_size])
        if self.profiler_on:
            self.profiler.start(self.fetch_event, lane_id == 0)
        self._fetch_from_queue()
        if self.profiler_on:
            self.profiler.end(self.fetch_event, lane_id == 0)

    def get_idx_and_task_type(self):
        return [self.m_idx, self.n_idx, self.k_idx], self.task_type

    @T.inline
    def wait(
        self, evt: Semaphore, *coord, wait_level: Literal["cta", "warp"] = "cta", mask=0xFFFFFFFF
    ):
        evt.semaphore_wait(*coord, level=wait_level, mask=mask)

    @T.inline
    def notify(
        self,
        evt: Semaphore,
        func_notify,
        scope: Literal["thread", "warp", "warpgroup", "cta"] = "thread",
        scope_id=0,
        pre_notify=False,
        release=False,
    ):
        # Notes: Here each thread will notify only at most one time,
        #        and the tids of the threads involved among scope in the
        #        notification process start from 0 and increment sequentially.
        # Notes: (notify_num, coord) = func_notify(notify_idx)
        # Notes: scope_id = -1 represents that each scope will separately notify

        max_notify_num_map = T.meta_var(
            {
                "thread": 1,
                "warp": self.hardware.warp_size,
                "warpgroup": self.hardware.warpgroup_size,
                "cta": self.hardware.num_threads,
            }
        )
        max_scope_id_map = T.meta_var(
            {
                "thread": self.hardware.num_threads,
                "warp": self.hardware.warp_count,
                "warpgroup": self.hardware.warpgroup_count,
                "cta": 1,
            }
        )

        @T.inline
        def sync(scope: Literal["thread", "warp", "warpgroup", "cta"], scope_id=0):
            if scope == "thread":
                pass
            elif scope == "warp":
                T.cuda.warp_sync()
            elif scope == "warpgroup":
                T.ptx.bar.sync(6 + scope_id, self.hardware.warpgroup_size)
            elif scope == "cta":
                T.tvm_storage_sync("shared")

        wg_id = T.warpgroup_id([self.hardware.warpgroup_count])
        warp_id = T.warp_id([self.hardware.warp_count])
        tid = T.thread_id([self.hardware.num_threads])
        tid_in_wg = T.thread_id_in_wg([self.hardware.warpgroup_size])
        lane_id = T.lane_id([self.hardware.warp_size])
        idx_map = T.meta_var(
            {
                "thread": (tid, 0),
                "warp": (warp_id, lane_id),
                "warpgroup": (wg_id, tid_in_wg),
                "cta": (0, tid),
            }
        )
        idx = T.meta_var(idx_map[scope])
        if self.debug:
            T.cuda.trap_when_assert_failed(scope_id == -1 or scope_id < max_scope_id_map[scope])
        if scope_id == -1 or idx[0] == scope_id:
            if not pre_notify:
                sync(scope, scope_id)
            notify_info = T.meta_var(func_notify(idx[1]))
            notify_num = T.meta_var(notify_info[0])
            coord = T.meta_var(notify_info[1:])
            if self.debug:
                T.cuda.trap_when_assert_failed(notify_num <= max_notify_num_map[scope])
            if idx[1] < notify_num:
                evt.semaphore_notify(*coord, pre_notify=pre_notify, release=release)

    def _enqueue(self, idx, func_trigger_list, push_level):
        if not isinstance(func_trigger_list, list):
            func_trigger_list = [func_trigger_list]
        for func_trigger in func_trigger_list:
            self.queue.enqueue(func_trigger(idx), push_level)

    @T.inline
    def pre_notify_and_push(
        self,
        evt: Semaphore,
        func_notify,
        func_trigger_list,
        push_level: Literal["thread", "warp", "warpgroup", "cta"],
        scope: Literal["thread", "warp", "warpgroup", "cta"],
        scope_id=0,
    ):
        max_notify_num_map = T.meta_var(
            {
                "thread": 1,
                "warp": self.hardware.warp_size,
                "warpgroup": self.hardware.warpgroup_size,
                "cta": self.hardware.num_threads,
            }
        )
        max_scope_id_map = T.meta_var(
            {
                "thread": self.hardware.num_threads,
                "warp": self.hardware.warp_count,
                "warpgroup": self.hardware.warpgroup_count,
                "cta": 1,
            }
        )

        wg_id = T.warpgroup_id([self.hardware.warpgroup_count])
        warp_id = T.warp_id([self.hardware.warp_count])
        tid = T.thread_id([self.hardware.num_threads])
        tid_in_wg = T.thread_id_in_wg([self.hardware.warpgroup_size])
        warp_id_in_wg = T.warp_id_in_wg([self.hardware.warps_per_warpgroup])
        lane_id = T.lane_id([self.hardware.warp_size])
        idx_map = T.meta_var(
            {
                "thread": (tid, 0),
                "warp": (warp_id, lane_id),
                "warpgroup": (wg_id, tid_in_wg),
                "cta": (0, tid),
            }
        )
        idx_in_scope_map = T.meta_var(
            {
                "thread": {"thread": 0},
                "warp": {"thread": lane_id, "warp": 0},
                "warpgroup": {"thread": tid_in_wg, "warp": warp_id_in_wg, "warpgroup": 0},
                "cta": {"thread": tid, "warp": warp_id, "warpgroup": wg_id, "cta": 0},
            }
        )
        stride_in_scope_map = T.meta_var(
            {
                "warp": {"warp": 1},
                "warpgroup": {
                    "warp": self.hardware.warps_per_warpgroup,
                    "warpgroup": 1,
                },
                "cta": {
                    "warp": self.hardware.warp_count,
                    "warpgroup": self.hardware.warpgroup_count,
                    "cta": 1,
                },
            }
        )
        scope_id_map = T.meta_var({"thread": tid, "warp": warp_id, "warpgroup": wg_id, "cta": 0})
        new_scope_id = T.if_then_else(scope_id == -1, scope_id_map[scope], scope_id)
        idx = T.meta_var(idx_map[scope])
        if self.debug:
            T.cuda.trap_when_assert_failed(scope_id == -1 or scope_id < max_scope_id_map[scope])
        if idx[0] == new_scope_id:
            notify_info = T.meta_var(func_notify(idx[1]))
            notify_num = T.meta_var(notify_info[0])
            coord_notify = T.meta_var(notify_info[1:])
            if self.debug:
                T.cuda.trap_when_assert_failed(notify_num <= max_notify_num_map[scope])
            if idx[1] < notify_num:
                evt.semaphore_notify(*coord_notify, pre_notify=True)
            if self.profiler_on:
                self.profiler.start(self.push_event, lane_id == 0)
            if scope == "thread":
                if tid == new_scope_id:
                    if push_level == "thread":
                        if evt.is_triggered():
                            self._enqueue(0, func_trigger_list, push_level)
                    else:
                        assert False
            elif scope == "warp":
                if warp_id == new_scope_id:
                    if push_level == "thread":
                        if lane_id < notify_num:
                            if evt.is_triggered():
                                self._enqueue(lane_id, func_trigger_list, push_level)
                    elif push_level == "warp":
                        self.semaphore_state[tid] = evt.state[0]
                        T.cuda.warp_sync()
                        self.idx = idx_in_scope_map[scope][push_level]
                        while self.idx < notify_num:
                            evt.state[0] = self.semaphore_state[
                                new_scope_id * self.hardware.warp_size + self.idx
                            ]
                            if evt.is_triggered():
                                self._enqueue(self.idx, func_trigger_list, push_level)
                            self.idx += stride_in_scope_map[scope][push_level]
                    else:
                        assert False
            elif scope == "warpgroup":
                if wg_id == new_scope_id:
                    if push_level == "thread":
                        if tid_in_wg < notify_num:
                            if evt.is_triggered():
                                self._enqueue(tid_in_wg, func_trigger_list, push_level)
                    elif push_level == "warp" or push_level == "warpgroup":
                        self.semaphore_state[tid] = evt.state[0]
                        T.cuda.warpgroup_sync(6 + wg_id)
                        self.idx = idx_in_scope_map[scope][push_level]
                        while self.idx < notify_num:
                            evt.state[0] = self.semaphore_state[
                                new_scope_id
                                * self.hardware.num_threads
                                // self.hardware.warpgroup_count
                                + self.idx
                            ]
                            if evt.is_triggered():
                                self._enqueue(self.idx, func_trigger_list, push_level)
                            self.idx += stride_in_scope_map[scope][push_level]
                    else:
                        assert False
            elif scope == "cta":
                if push_level == "thread":
                    if tid < notify_num:
                        if evt.is_triggered():
                            self._enqueue(tid, func_trigger_list, push_level)
                elif push_level == "warp" or push_level == "warpgroup" or push_level == "cta":
                    self.semaphore_state[tid] = evt.state[0]
                    T.tvm_storage_sync("shared")
                    self.idx = idx_in_scope_map[scope][push_level]
                    while self.idx < notify_num:
                        evt.state[0] = self.semaphore_state[self.idx]
                        if evt.is_triggered():
                            self._enqueue(self.idx, func_trigger_list, push_level)
                        self.idx += stride_in_scope_map[scope][push_level]
                else:
                    assert False
            else:
                assert False
            if self.profiler_on:
                self.profiler.end(self.push_event, lane_id == 0)

    def valid(self):
        return self.task_type != self.end_task_type


class MPMCQueueHost:
    def __init__(self, capacity: int, packing=None):
        self.capacity = capacity
        self.packing = packing
        self.tasks = np.full((capacity,), -1, dtype=np.int32)
        self.head = np.zeros((1,), dtype=np.int32)
        self.tail = np.zeros((1,), dtype=np.int32)
        self.head[0] = 0
        self.tail[0] = 0

    def enqueue(self, task_type, m_idx, n_idx, k_idx):
        pos = self.tail[0] & (self.capacity - 1)
        self.tasks[pos] = pack_into_32bit(m_idx, n_idx, k_idx, task_type, packing=self.packing)
        self.tail[0] = self.tail[0] + 1


__all__ = [
    "DynamicTileScheduler",
    "MPMCQueue",
    "MPMCQueueHost",
    "SchedulerBarrier",
    "Semaphore",
]
