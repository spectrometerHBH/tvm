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
"""Default static scheduler and event semaphore helpers."""

from __future__ import annotations

from typing import Any

from tvm.script import tirx as T

from ..dsl import SmemManager


def _gt(lhs, rhs):
    return T.cuda.func_call(
        "tirx_megakernel_gt",
        lhs,
        rhs,
        source_code="""
__forceinline__ __device__ bool tirx_megakernel_gt(int32_t a, int32_t b) {
    return a > b;
}
""",
        return_type="bool",
    )


class TIRXSemaphore:
    """Counter semaphore backed by a logical event buffer."""

    base = 1 << 16

    def __init__(self, buffer, *, sleep_cycles: int = 40):
        self.buffer = buffer
        self.state = T.alloc_buffer((1,), "int32", scope="local", align=4)
        self.sleep_cycles = sleep_cycles

    @T.inline
    def semaphore_wait(self, *coord, level: str = "cta", mask=0xFFFFFFFF) -> None:
        if level == "cta":
            while 1:
                T.ptx.ld_global_acquire(
                    self.state[0],
                    self.buffer.access_ptr("r", offset=self.buffer.elem_offset_of(coord)),
                )
                if T.cuda.syncthreads_and(self.state[0] == 0):
                    break
                T.cuda.nano_sleep(self.sleep_cycles)
        elif level == "warp":
            warp_id = T.warp_id([8])
            lane_id = T.lane_id([32])
            if ((mask >> warp_id) & 1) == 1:
                self.state[0] = -1
                while 1:
                    if lane_id == 0:
                        T.ptx.ld_global_acquire(
                            self.state[0],
                            self.buffer.access_ptr("r", offset=self.buffer.elem_offset_of(coord)),
                        )
                    if T.ptx.any_sync(0xFFFFFFFF, self.state[0] == 0):
                        break
                    T.cuda.nano_sleep(self.sleep_cycles)
        else:
            raise ValueError(f"unsupported wait level {level!r}")

    @T.inline
    def semaphore_notify(self, *coord, rank=-1, release: bool = False) -> None:
        if release:
            T.cuda.thread_fence()
        self.state[0] = T.cuda.atomic_add(self.buffer.ptr_to(coord), -(self.base + 1))
        if self.state[0] <= 0:
            while 1:
                T.ptx.ld_global_acquire(self.state[0], self.buffer.ptr_to(coord))
                if _gt(self.state[0], 0):
                    if release:
                        T.cuda.thread_fence()
                    self.state[0] = T.cuda.atomic_add(self.buffer.ptr_to(coord), -(self.base + 1))
                    break
                T.cuda.nano_sleep(self.sleep_cycles)


class StaticTileScheduler:
    """Persistent static queue scheduler used by the default policy."""

    MAX_TASKS = 128

    def __init__(
        self,
        exec_queue: Any,
        smem_manager: SmemManager,
        *,
        debug: bool = False,
        sm_count: int = 1,
        num_threads: int = 256,
        max_tasks: int = MAX_TASKS,
        end_job_id: int = 31,
        warp_count: int | None = None,
        warpgroup_count: int | None = None,
        warpgroup_size: int = 128,
    ):
        self.exec_queue = exec_queue
        self.smem_manager = smem_manager
        self.debug = debug
        self.sm_count = sm_count
        self.num_threads = num_threads
        self.max_tasks = max_tasks
        self.end_job_id = end_job_id
        self.warp_count = warp_count or max(1, num_threads // 32)
        self.warpgroup_count = warpgroup_count or max(1, num_threads // warpgroup_size)
        self.warpgroup_size = warpgroup_size

    def _alloc(self) -> None:
        self.m_idx = T.alloc_buffer((1,), "int32", scope="local")
        self.n_idx = T.alloc_buffer((1,), "int32", scope="local")
        self.k_idx = T.alloc_buffer((1,), "int32", scope="local")
        self.task_type = T.alloc_buffer((1,), "int32", scope="local")
        self.tile_idx = T.alloc_buffer((1,), "int32", scope="local")
        self.queue_smem = self.smem_manager.alloc(
            (self.max_tasks,), "int32", align=16, policy="persistent"
        )

    @T.inline
    def _update_current(self) -> None:
        packed = T.alloc_buffer((1,), "int32", scope="local")
        packed[0] = self.queue_smem[self.tile_idx[0]]
        self.task_type[0] = T.bitwise_and(packed[0], 0x1F)
        self.m_idx[0] = T.bitwise_and(T.shift_right(packed[0], 5), 0x1FFF)
        self.n_idx[0] = T.bitwise_and(T.shift_right(packed[0], 18), 0x3FF)
        self.k_idx[0] = T.bitwise_and(T.shift_right(packed[0], 28), 0xF)

    @T.inline
    def init(self) -> None:
        self._alloc()
        bx = T.cta_id([self.sm_count])
        tid = T.thread_id([self.num_threads])
        self.tile_idx[0] = 0
        for k in T.serial(0, (self.max_tasks + self.num_threads - 1) // self.num_threads):
            idx = k * self.num_threads + tid
            if idx < self.max_tasks:
                self.queue_smem[idx] = self.exec_queue[bx, idx]
        T.tvm_storage_sync("shared")
        self._update_current()

    def indices(self):
        return self.m_idx[0], self.n_idx[0], self.k_idx[0]

    @T.inline
    def next_tile(self) -> None:
        self.tile_idx[0] = self.tile_idx[0] + 1
        self._update_current()

    @T.inline
    def wait(self, semaphore, *coord, wait_level="cta", mask=0xFFFFFFFF) -> None:
        semaphore.semaphore_wait(*coord, level=wait_level, mask=mask)

    @T.inline
    def notify(
        self,
        semaphore,
        func_notify,
        scope="thread",
        scope_id=0,
        release=False,
    ) -> None:
        max_notify = T.meta_var(
            {"thread": 1, "warp": 32, "warpgroup": self.warpgroup_size, "cta": self.num_threads}
        )
        max_scope_id = T.meta_var(
            {
                "thread": self.num_threads,
                "warp": self.warp_count,
                "warpgroup": self.warpgroup_count,
                "cta": 1,
            }
        )
        wg_id = T.warpgroup_id([self.warpgroup_count])
        warp_id = T.warp_id([self.warp_count])
        tid = T.thread_id([self.num_threads])
        tid_in_wg = T.thread_id_in_wg([self.warpgroup_size])
        lane_id = T.lane_id([32])
        idx = T.meta_var(
            {
                "thread": (tid, 0),
                "warp": (warp_id, lane_id),
                "warpgroup": (wg_id, tid_in_wg),
                "cta": (0, tid),
            }[scope]
        )
        if self.debug:
            T.cuda.trap_when_assert_failed(scope_id == -1 or scope_id < max_scope_id[scope])
        if scope_id == -1 or idx[0] == scope_id:
            self._sync_notify_scope(scope, scope_id)
            notify_info = T.meta_var(func_notify(idx[1]))
            notify_num = notify_info[0]
            rank = notify_info[1]
            coord = T.meta_var(notify_info[2:])
            if self.debug:
                T.cuda.trap_when_assert_failed(notify_num <= max_notify[scope])
            if idx[1] < notify_num:
                semaphore.semaphore_notify(*coord, rank=rank, release=release)

    @T.inline
    def _sync_notify_scope(self, scope: str, scope_id: int = 0) -> None:
        if scope == "thread":
            pass
        elif scope == "warp":
            T.cuda.warp_sync()
        elif scope == "warpgroup":
            T.ptx.bar.sync(6 + scope_id, self.warpgroup_size)
        elif scope == "cta":
            T.tvm_storage_sync("shared")
        else:
            raise ValueError(f"unsupported notify scope {scope!r}")

    def valid(self):
        return (self.tile_idx[0] < self.max_tasks) & (self.task_type[0] != self.end_job_id)


__all__ = ["StaticTileScheduler", "TIRXSemaphore"]
