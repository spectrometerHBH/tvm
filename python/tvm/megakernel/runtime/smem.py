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
"""Chunked shared-memory manager with the mbarrier phase protocol.

Migrated from the production ``tirx_kernels.megakernel.utils.base.SmemManager``.
It implements the ``tvm.megakernel.dsl.SmemManager`` abstract API and is the
shared-memory manager used by the runtime-library builder.
"""

from __future__ import annotations

from typing import Literal

from tvm.script import tirx as T
from tvm.tirx.expr import Var

from ..dsl import SmemManager as DslSmemManager
from .config import HardwareConfig


@T.meta_class
class SmemManager(DslSmemManager):
    """Shared memory manager"""

    def __init__(
        self,
        smem_max_bytes,
        chunk_size,
        ptr: Var,
        fusion_mode=False,
        hardware: HardwareConfig | None = None,
    ):
        self.smem_max_bytes = smem_max_bytes
        self.chunk_size = chunk_size
        self.chunk_num = smem_max_bytes // chunk_size
        if self.chunk_num > 32:
            raise ValueError("chunk_num must be <= 32")
        self.ptr = ptr
        self.hardware = hardware or HardwareConfig()
        self.reguler_pool_allocator = T.SMEMPool(ptr)
        self.persistent_pool_allocator = T.SMEMPool(None if fusion_mode else ptr)
        self.tiles = {}
        self.runtime_tile_chunk_count = {}
        self.runtime_tile_advance_count = {}
        self.protocol_errors = []
        self.bufs = {}
        self.persistent_bufs = {}
        self.cur_tile_name = ""
        self.persistent_pool_allocator.move_base_to(self.chunk_size * self.chunk_num)
        self.fusion_mode = fusion_mode
        self._pool_allocators = {
            "persistent": self.persistent_pool_allocator,
            "shared": self.reguler_pool_allocator,
            "exclusive": self.reguler_pool_allocator,
        }

    @property
    def pool_allocator(self):
        return self.reguler_pool_allocator

    def _inner_alloc(self):
        self.mbar = self.alloc((self.chunk_num,), "uint64", policy="persistent")
        self.shared_count = self.alloc((1,), "int32", policy="persistent")
        if self.fusion_mode:
            self.cur_phase = T.alloc_local([1], "int32", scope="local.persistent")
            self.reg_count = T.alloc_local([1], "int32", scope="local.persistent")
        else:
            self.cur_phase = T.alloc_local([1], "int32")
            self.reg_count = T.alloc_local([1], "int32")

    @T.inline
    def init(self):
        tid = T.thread_id([self.hardware.num_threads])
        self.check_smem_well_formed(debug=False)
        self._inner_alloc()
        self.cur_phase[0] = 1
        if tid == 0:
            for i in T.serial(self.chunk_num):
                T.ptx.mbarrier.init(self.mbar.ptr_to([i]), 1)
            self.shared_count[0] = 0
        T.tvm_storage_sync("shared")
        T.ptx.fence.mbarrier_init()
        T.ptx.fence.proxy_async("shared::cta")

    def alloc(
        self,
        shape,
        dtype="float32",
        strides=None,
        scope="shared.dyn",
        align=1,
        layout="default",
        split=1,
        policy: Literal["shared", "exclusive", "persistent"] = "shared",
        method: Literal["shared", "exclusive", "persistent"] | None = None,
    ):
        if method is not None:
            # Legacy spelling kept for the production tile tasks.
            policy = method
        if policy not in self.VALID_POLICIES:
            raise ValueError(f"unsupported smem policy {policy!r}")
        if "shared" not in scope:
            raise ValueError("smem manager only allocates shared-scope buffers")
        pool_allocator = self._pool_allocators[policy]
        beg = pool_allocator.offset
        if align > 0:
            beg = (beg + align - 1) // align * align
        if self.fusion_mode and policy == "persistent":
            scope = "shared.persistent"
        buf = pool_allocator.alloc(shape, dtype, strides, scope, align, layout)
        end = pool_allocator.offset
        size = end - beg
        if size % split != 0:
            raise ValueError("smem allocation size must be divisible by split")
        if policy == "persistent":
            self.persistent_bufs[buf] = (beg, end)
        else:
            if policy == "shared":
                if len(self.tiles[self.cur_tile_name][1]["exclusive"]) != 0:
                    raise ValueError(
                        "Cannot use both shared and shared/exclusive methods at the same time"
                    )
            elif policy == "exclusive":
                if len(self.tiles[self.cur_tile_name][1]["shared"]) != 0:
                    raise ValueError(
                        "Cannot use both shared and shared/exclusive methods at the same time"
                    )
            buf_info = (split, beg, size, policy)
            self.tiles[self.cur_tile_name][0] = max(
                self.tiles[self.cur_tile_name][0], (end - 1) // self.chunk_size
            )
            self.tiles[self.cur_tile_name][1][policy].append(buf_info)
            self.bufs[buf] = buf_info
            if policy == "exclusive":
                for split_idx in range(split):
                    beg_chunk_id = (beg + size // split * split_idx) // self.chunk_size
                    end_chunk_id = (beg + size // split * (split_idx + 1) - 1) // self.chunk_size
                    for chunk_id in range(beg_chunk_id, end_chunk_id + 1):
                        self.tiles[self.cur_tile_name][2][chunk_id] += 1
        return buf

    def check_smem_well_formed(self, debug=False):
        for _, buf_info_dict, _ in self.tiles.values():
            checked_exclusive = []
            check_overlap = []
            for policy in ["shared", "exclusive"]:
                for split, beg, size, _ in buf_info_dict[policy]:
                    end = beg + size
                    if end > self.chunk_num * self.chunk_size:
                        raise ValueError("smem allocation exceeds the chunked region")
                    for beg_other, end_other in check_overlap:
                        if not (beg >= end_other or beg_other >= end):
                            raise ValueError("Overlap detected in smem allocation")
                    check_overlap.append((beg, end))
                    if policy == "exclusive":
                        for split_idx in range(split):
                            beg_chunk_id = (beg + size // split * split_idx) // self.chunk_size
                            end_chunk_id = (
                                beg + size // split * (split_idx + 1) - 1
                            ) // self.chunk_size
                            for beg_id, end_id in checked_exclusive:
                                if not (beg_id > end_chunk_id or end_id < beg_chunk_id):
                                    raise ValueError("Exclusive chunk overlap detected")
                            checked_exclusive.append((beg_chunk_id, end_chunk_id))
                    else:
                        beg_chunk_id = beg // self.chunk_size
                        end_chunk_id = (end - 1) // self.chunk_size
                        checked_exclusive.append((beg_chunk_id, end_chunk_id))
        for beg_persistent, end_persistent in self.persistent_bufs.values():
            if not (
                beg_persistent >= self.chunk_size * self.chunk_num
                and end_persistent <= self.smem_max_bytes
            ):
                raise ValueError("persistent smem allocation is outside its reserved region")
            for _, beg, size, _ in self.bufs.values():
                if not (beg >= end_persistent or beg_persistent >= beg + size):
                    raise ValueError("persistent and transient smem allocations overlap")
        persistent_buf_list = list(self.persistent_bufs.values())
        for i in range(len(persistent_buf_list)):
            beg_i, end_i = persistent_buf_list[i]
            for j in range(i + 1, len(persistent_buf_list)):
                beg_j, end_j = persistent_buf_list[j]
                if not (beg_i >= end_j or beg_j >= end_i):
                    raise ValueError("persistent smem allocations overlap")
        if debug:
            self._debug_print()

    def _debug_print(self):
        for k, v in self.tiles.items():
            print(k, v)
        for k, v in self.bufs.items():
            print(k, v)
        for k, v in self.persistent_bufs.items():
            print(k, v)

    def set_tile(self, cur_tile):
        if cur_tile is None:
            self.cur_tile_name = "default"
        else:
            self.cur_tile_name = str(cur_tile)
        self.tiles.setdefault(
            self.cur_tile_name,
            [
                -1,
                {"exclusive": [], "shared": []},
                [0 for _ in range(self.chunk_num)],
            ],
        )
        self.runtime_tile_chunk_count.setdefault(
            self.cur_tile_name,
            [[0 for _ in range(self.chunk_num)] for _ in range(2)],
        )
        self.runtime_tile_advance_count.setdefault(self.cur_tile_name, 0)
        self.reguler_pool_allocator.move_base_to(0)

    def _assert_cond(self, cond, message="smem manager protocol violation"):
        if not cond:
            raise ValueError(message)

    @T.inline
    def advance(self):
        self._mark_advance()
        self.cur_phase[0] = self.cur_phase[0] ^ 1

    def _mark_advance(self):
        self.runtime_tile_advance_count[self.cur_tile_name] = (
            self.runtime_tile_advance_count[self.cur_tile_name] + 1
        )

    def enter_tile_runtime(self, cur_tile):
        self.cur_tile_name = str(cur_tile)

    def exit_tile_runtime(self):
        self._check_runtime()
        self.cur_tile_name = ""

    def _check_runtime(self):
        tile = self.tiles.get(self.cur_tile_name)
        if tile is None:
            self.protocol_errors.append(
                f"tile {self.cur_tile_name!r} runtime hook has no smem allocation plan"
            )
            return
        allocations = tile[1]["shared"] + tile[1]["exclusive"]
        if not allocations:
            return
        waits, arrivals = self.runtime_tile_chunk_count[self.cur_tile_name]
        missing_waits = [chunk for chunk, count in enumerate(waits) if count == 0]
        missing_arrivals = [chunk for chunk, count in enumerate(arrivals) if count == 0]
        mismatched = [
            chunk
            for chunk, (wait_count, arrive_count) in enumerate(zip(waits, arrivals))
            if wait_count != arrive_count
        ]
        if missing_waits or missing_arrivals or mismatched:
            self.protocol_errors.append(
                f"tile {self.cur_tile_name!r} runtime hook violates the managed smem "
                "phase protocol: every chunk must be acquired and released equally; "
                f"missing acquire={missing_waits}, missing release={missing_arrivals}, "
                f"mismatched={mismatched}"
            )
            return
        advance_count = self.runtime_tile_advance_count[self.cur_tile_name]
        if advance_count != 1:
            self.protocol_errors.append(
                f"tile {self.cur_tile_name!r} runtime hook must call advance() exactly "
                f"once after managed smem release, got {advance_count}"
            )

    def _mark_chunk_range(self, kind: Literal["wait", "arrive"], beg: int, end: int):
        counts = self.runtime_tile_chunk_count[self.cur_tile_name][0 if kind == "wait" else 1]
        for chunk_id in range(beg, end + 1):
            counts[chunk_id] += 1

    def _buffer_chunk_range(self, buffer, split_idx: int) -> tuple[int, int]:
        split, beg, size, _ = self.bufs[buffer]
        split_size = size // split
        return (
            (beg + split_size * split_idx) // self.chunk_size,
            (beg + split_size * (split_idx + 1) - 1) // self.chunk_size,
        )

    def _mark_buffer_chunks(self, kind: Literal["wait", "arrive"], buffer, split_idx) -> None:
        if isinstance(split_idx, int) and not isinstance(split_idx, bool):
            beg_chunk_id, end_chunk_id = self._buffer_chunk_range(buffer, split_idx)
        else:
            _, beg, size, _ = self.bufs[buffer]
            beg_chunk_id = beg // self.chunk_size
            end_chunk_id = (beg + size - 1) // self.chunk_size
        self._mark_chunk_range(kind, beg_chunk_id, end_chunk_id)

    def _mark_runtime_chunk(self, kind: Literal["wait", "arrive"], chunk_id):
        if isinstance(chunk_id, bool) or not isinstance(chunk_id, int):
            raise ValueError(
                f"tile {self.cur_tile_name!r} runtime hook uses a dynamic smem chunk id; "
                "the phase protocol requires statically identifiable chunks"
            )
        if not 0 <= chunk_id < self.chunk_num:
            raise ValueError(f"managed smem chunk id {chunk_id} is out of range")
        self._mark_chunk_range(kind, chunk_id, chunk_id)

    def _mark_unused_chunks(self, kind: Literal["wait", "arrive"], cur_tile):
        # Keep this bookkeeping outside the inline macro's local assignment.
        # The TIRX parser materializes ``first_unused`` as a BufferLoad for the
        # emitted predicate, while the allocation-plan value is still a Python
        # integer here and can be audited exactly.
        first_unused = self.tiles[str(cur_tile)][0] + 1
        if first_unused < self.chunk_num:
            self._mark_chunk_range(kind, first_unused, self.chunk_num - 1)

    @T.inline
    def wait_all(self, level: Literal["cta", "warpgroup"] = "cta"):
        self._mark_chunk_range("wait", 0, self.chunk_num - 1)
        lane_id = T.lane_id([self.hardware.warp_size])
        warp_id = T.warp_id([self.hardware.warp_count])
        wg_id = T.warpgroup_id([self.hardware.warpgroup_count])
        if level == "cta":
            if warp_id == 0:
                if lane_id < self.chunk_num:
                    T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])
            T.tvm_storage_sync("shared")
        elif level == "warpgroup":
            if warp_id % self.hardware.warps_per_warpgroup == 0:
                if lane_id < self.chunk_num:
                    T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])
            T.ptx.bar.sync(6 + wg_id, self.hardware.warpgroup_size)

    def acquire_all(self, level="cta"):
        """Alias for ``wait_all`` required by the DSL smem-manager API."""

        self.wait_all(level)

    @T.inline
    def wait_specific(self, lane_id, buffer, split_idx: int):
        self._assert_cond(
            buffer in self.bufs and buffer not in self.persistent_bufs,
            "wait_specific requires a transient buffer",
        )
        self._assert_cond(
            self.bufs[buffer][3] == "exclusive", "wait_specific requires an exclusive buffer"
        )
        beg_chunk_id, end_chunk_id = self._buffer_chunk_range(buffer, split_idx)
        self._mark_buffer_chunks("wait", buffer, split_idx)
        if (lane_id >= beg_chunk_id) & (lane_id <= end_chunk_id):
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])

    @T.inline
    def wait_unused(self, lane_id, cur_tile):
        self._assert_cond(
            len(self.tiles[self.cur_tile_name][1]["shared"]) == 0,
            "wait_unused requires exclusive-only allocations",
        )
        self._mark_unused_chunks("wait", cur_tile)
        first_unused = self.tiles[str(cur_tile)][0] + 1
        if (lane_id < self.chunk_num) & (lane_id >= first_unused):
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([lane_id]), self.cur_phase[0])

    @T.inline
    def wait_chunk(self, chunk_id):
        self._mark_runtime_chunk("wait", chunk_id)
        T.ptx.mbarrier.try_wait(self.mbar.ptr_to([chunk_id]), self.cur_phase[0])

    @T.inline
    def wait_specific_one_thread(self, buffer, split_idx: int):
        self._assert_cond(
            buffer in self.bufs and buffer not in self.persistent_bufs,
            "wait_specific_one_thread requires a transient buffer",
        )
        self._assert_cond(
            self.bufs[buffer][3] == "exclusive",
            "wait_specific_one_thread requires an exclusive buffer",
        )
        beg_chunk_id, end_chunk_id = self._buffer_chunk_range(buffer, split_idx)
        self._mark_buffer_chunks("wait", buffer, split_idx)
        for idx in T.serial(0, end_chunk_id - beg_chunk_id + 1):
            T.ptx.mbarrier.try_wait(self.mbar.ptr_to([beg_chunk_id + idx]), self.cur_phase[0])

    @T.inline
    def arrive_all(self, level: Literal["cta", "warpgroup"] = "cta"):
        self._mark_chunk_range("arrive", 0, self.chunk_num - 1)
        lane_id = T.lane_id([self.hardware.warp_size])
        warp_id = T.warp_id([self.hardware.warp_count])
        wg_id = T.warpgroup_id([self.hardware.warpgroup_count])
        if level == "cta":
            T.tvm_storage_sync("shared")
            if warp_id == 0:
                if lane_id < self.chunk_num:
                    T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))
        elif level == "warpgroup":
            self.reg_count[0] = 0
            T.ptx.bar.sync(6 + wg_id, self.hardware.warpgroup_size)
            if warp_id % self.hardware.warps_per_warpgroup == 0:
                if lane_id == 0:
                    self.reg_count[0] = T.cuda.atomic_add(T.address_of(self.shared_count[0]), 1) + 1
                    if self.reg_count[0] == self.hardware.warpgroup_count:
                        T.cuda.atomic_add(
                            T.address_of(self.shared_count[0]), -self.hardware.warpgroup_count
                        )
                self.reg_count[0] = T.tvm_warp_shuffle(
                    self.hardware.full_mask,
                    self.reg_count[0],
                    0,
                    self.hardware.warp_size,
                    self.hardware.warp_size,
                )
                if self.reg_count[0] == self.hardware.warpgroup_count:
                    if lane_id < self.chunk_num:
                        T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))

    def release_all(self, level="cta"):
        """Alias for ``arrive_all`` required by the DSL smem-manager API."""

        self.arrive_all(level)

    @T.inline
    def arrive_specific(self, lane_id, buffer, split_idx: int):
        self._assert_cond(
            buffer in self.bufs and buffer not in self.persistent_bufs,
            "arrive_specific requires a transient buffer",
        )
        self._assert_cond(
            self.bufs[buffer][3] == "exclusive", "arrive_specific requires an exclusive buffer"
        )
        beg_chunk_id, end_chunk_id = self._buffer_chunk_range(buffer, split_idx)
        self._mark_buffer_chunks("arrive", buffer, split_idx)
        if (lane_id >= beg_chunk_id) & (lane_id <= end_chunk_id):
            T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))

    @T.inline
    def arrive_unused(self, lane_id, cur_tile):
        self._assert_cond(
            len(self.tiles[self.cur_tile_name][1]["shared"]) == 0,
            "arrive_unused requires exclusive-only allocations",
        )
        self._mark_unused_chunks("arrive", cur_tile)
        first_unused = self.tiles[str(cur_tile)][0] + 1
        if (lane_id < self.chunk_num) & (lane_id >= first_unused):
            T.ptx.mbarrier.arrive(self.mbar.ptr_to([lane_id]))

    @T.inline
    def arrive_chunk(self, chunk_id):
        self._mark_runtime_chunk("arrive", chunk_id)
        T.ptx.mbarrier.arrive(self.mbar.ptr_to([chunk_id]))

    def commit(self):
        """Validate the allocation plan, as required by the DSL API."""

        self.check_smem_well_formed(debug=False)


__all__ = ["SmemManager"]
