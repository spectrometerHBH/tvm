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
"""Concrete shared-memory lowering for the default TIRX backend."""

from __future__ import annotations

from typing import Literal

from tvm.script import tirx as T

from ..dsl import SmemAllocRecord, SmemManager, TileSpec


def _tile_key(tile) -> str:
    """Return the stable allocation/phase key for one declaration owner."""

    if tile is None:
        return "default"
    if isinstance(tile, TileSpec):
        return tile.name
    return str(tile)


@T.inline
def _wait_all_chunks(mbar, phase, chunk_count: T.constexpr, warp_count: T.constexpr):
    lane_id = T.lane_id([32])
    warp_id = T.warp_id([warp_count])
    if warp_id == 0:
        if lane_id < chunk_count:
            T.ptx.mbarrier.try_wait(mbar.ptr_to([lane_id]), phase[0])
    T.tvm_storage_sync("shared")


@T.inline
def _release_all_chunks(mbar, chunk_count: T.constexpr, warp_count: T.constexpr):
    lane_id = T.lane_id([32])
    warp_id = T.warp_id([warp_count])
    T.tvm_storage_sync("shared")
    if warp_id == 0:
        if lane_id < chunk_count:
            T.ptx.mbarrier.arrive(mbar.ptr_to([lane_id]))


@T.inline
def _advance_phase(phase):
    phase[0] = phase[0] ^ 1


class TIRXSmemManager(SmemManager):
    """Allocate shared memory and lower phase operations to CTA mbarriers."""

    def __init__(
        self,
        smem_max_bytes: int,
        chunk_size: int,
        *,
        num_threads: int = 256,
        warp_count: int | None = None,
    ):
        if smem_max_bytes <= 0 or chunk_size <= 0:
            raise ValueError("shared-memory sizes must be positive")
        self.smem_max_bytes = smem_max_bytes
        self.chunk_size = chunk_size
        self.chunk_num = (smem_max_bytes + chunk_size - 1) // chunk_size
        if self.chunk_num > 32:
            raise ValueError("shared-memory manager supports at most 32 chunks")
        self.num_threads = num_threads
        self.warp_count = warp_count or max(1, num_threads // 32)

        regular = T.SMEMPool()
        persistent = T.SMEMPool(regular.ptr)
        persistent.move_base_to(chunk_size * max(0, self.chunk_num - 1))
        self.pool_allocator = {
            "persistent": persistent,
            "shared": regular,
            "exclusive": regular,
        }
        self.tiles = {}
        self.bufs = {}
        self.persistent_bufs = {}
        self.cur_tile_name = ""
        self.exist_bufs = {}
        self.records: list[SmemAllocRecord] = []
        self.tile_phase_state = {}
        self.tile_class_keys: dict[str, str] = {}
        self.class_tile_keys: dict[str, set[str]] = {}
        self.mbar = None
        self.cur_phase = None

    def set_tile(self, tile) -> None:
        """Begin declaration-time allocation recording for one tile."""

        self._select_owner(tile)

    def enter_tile_runtime(self, tile) -> None:
        self._select_owner(tile)

    def exit_tile_runtime(self) -> None:
        self.cur_tile_name = ""

    def alloc(
        self,
        shape,
        dtype="float32",
        strides=None,
        scope="shared.dyn",
        align=0,
        buffer_type="",
        axis_separators=None,
        layout="default",
        split=1,
        name=None,
        policy: Literal["shared", "exclusive", "persistent"] = "shared",
    ):
        """Allocate and record one managed shared-memory buffer."""

        if policy not in self.VALID_POLICIES:
            valid = ", ".join(sorted(self.VALID_POLICIES))
            raise ValueError(f"unsupported smem policy {policy!r}; expected {valid}")
        if not isinstance(split, int) or isinstance(split, bool) or split <= 0:
            raise ValueError("smem split must be a positive integer")
        if name is not None:
            suffix = self.exist_bufs.setdefault(name, 0)
            self.exist_bufs[name] += 1
            if suffix:
                name = f"{name}{suffix}"
        if buffer_type:
            raise ValueError("TIRXSmemManager does not support buffer_type")
        if axis_separators:
            raise ValueError("TIRXSmemManager does not support axis_separators")

        allocator = self.pool_allocator[policy]
        begin = allocator.offset
        if align > 0:
            begin = (begin + align - 1) // align * align
        buffer = allocator.alloc(
            shape,
            dtype,
            strides,
            scope,
            align,
            layout,
        )
        end = allocator.offset
        size = end - begin
        if size % split:
            raise ValueError("smem allocation size must be divisible by split")

        if policy == "persistent":
            self.persistent_bufs[buffer] = begin, end
        else:
            buffers = self.tiles.setdefault(self.cur_tile_name, {"exclusive": [], "shared": []})
            other_policy = "exclusive" if policy == "shared" else "shared"
            if any(
                self.tiles[owner_key][other_policy]
                for owner_key in self._related_owner_keys(self.cur_tile_name)
            ):
                raise ValueError(
                    "one tile cannot mix shared and exclusive smem allocation policies"
                )
            info = split, begin, size, policy
            buffers[policy].append(info)
            self.bufs[buffer] = info
            self._phase_state()["uses_managed_smem"] = True
        self.records.append(
            SmemAllocRecord(
                buffer,
                shape,
                dtype,
                policy,
                {
                    "strides": strides,
                    "scope": scope,
                    "align": align,
                    "buffer_type": buffer_type,
                    "axis_separators": axis_separators,
                    "layout": layout,
                    "split": split,
                    "name": name,
                },
            )
        )
        return buffer

    def _allocate_phase_state(self) -> None:
        self.mbar = self.alloc(
            (self.chunk_num,),
            "uint64",
            align=8,
            name="megakernel_smem_mbar",
            policy="persistent",
        )
        self.cur_phase = T.alloc_buffer((1,), "int32", scope="local", align=4)

    @T.inline
    def init(self) -> None:
        """Allocate and initialize the chunk-level CTA mbarriers."""

        self._allocate_phase_state()
        self.cur_phase[0] = 1
        tid = T.thread_id([self.num_threads])
        if tid == 0:
            for index in T.serial(self.chunk_num):
                T.ptx.mbarrier.init(self.mbar.ptr_to([index]), 1)
        T.tvm_storage_sync("shared")
        T.ptx.fence.mbarrier_init()
        T.ptx.fence.proxy_async("shared::cta")

    def acquire_all(self, level="cta") -> None:
        """Wait until all managed shared-memory chunks can be reused."""

        if level != "cta":
            raise ValueError("TIRXSmemManager.acquire_all supports only level='cta'")
        self._phase_state()["acquire_all"] = True
        self._phase_state()["phase_ops"].append("acquire_all")
        _wait_all_chunks(self.mbar, self.cur_phase, self.chunk_num, self.warp_count)

    def wait_all(self, level="cta") -> None:
        """Compatibility spelling for ``acquire_all``."""

        self.acquire_all(level)

    def release_all(self, level="cta") -> None:
        """Release all managed shared-memory chunks at CTA scope."""

        if level != "cta":
            raise ValueError("TIRXSmemManager.release_all supports only level='cta'")
        self._phase_state()["release_all"] = True
        self._phase_state()["phase_ops"].append("release_all")
        _release_all_chunks(self.mbar, self.chunk_num, self.warp_count)

    def advance(self) -> None:
        """Flip the mbarrier phase used by the next tile execution."""

        self._phase_state()["advance"] = True
        self._phase_state()["phase_ops"].append("advance")
        _advance_phase(self.cur_phase)

    def validate_tile_phase(self, tile) -> None:
        """Require acquire/release calls for tiles using managed transient SMEM."""

        tile_name = _tile_key(tile)
        state = self.tile_phase_state.get(tile_name, {})
        class_state = {}
        if isinstance(tile, TileSpec):
            class_state = self.tile_phase_state.get(_tile_key(type(tile.impl)), {})
        if not (state.get("uses_managed_smem") or class_state.get("uses_managed_smem")):
            return
        missing = []
        if not state.get("acquire_all"):
            missing.append("acquire_all()")
        if not state.get("release_all"):
            missing.append("release_all()")
        if not state.get("advance"):
            missing.append("advance()")
        if missing:
            display_name = "default" if tile is None else getattr(tile, "name", tile_name)
            raise ValueError(
                f"tile {display_name!r} allocates managed shared memory but does not call "
                + " and ".join(missing)
            )
        phase_ops = state.get("phase_ops", ())
        expected = ("acquire_all", "release_all", "advance")
        if len(phase_ops) % len(expected) or any(
            tuple(phase_ops[index : index + len(expected)]) != expected
            for index in range(0, len(phase_ops), len(expected))
        ):
            display_name = "default" if tile is None else getattr(tile, "name", tile_name)
            raise ValueError(
                f"tile {display_name!r} must call acquire_all(), release_all(), and advance() "
                "in order for each managed shared-memory phase"
            )

    def commit(self) -> None:
        """Validate allocation intervals and commit the dynamic-SMEM size."""

        self._validate_allocations()
        self.pool_allocator["shared"].commit(self.smem_max_bytes)

    def _phase_state(self):
        return self.tile_phase_state.setdefault(
            self.cur_tile_name,
            {
                "uses_managed_smem": False,
                "acquire_all": False,
                "release_all": False,
                "advance": False,
                "phase_ops": [],
            },
        )

    def _select_owner(self, owner) -> None:
        owner_key = _tile_key(owner)
        self.cur_tile_name = owner_key
        self.tiles.setdefault(owner_key, {"exclusive": [], "shared": []})
        self._phase_state()
        if isinstance(owner, TileSpec):
            class_key = _tile_key(type(owner.impl))
            previous = self.tile_class_keys.setdefault(owner_key, class_key)
            if previous != class_key:
                raise ValueError(f"tile owner key {owner_key!r} maps to more than one class")
            self.class_tile_keys.setdefault(class_key, set()).add(owner_key)
            self.tiles.setdefault(class_key, {"exclusive": [], "shared": []})
        self.pool_allocator["shared"].move_base_to(
            self._transient_end(self._related_owner_keys(owner_key))
        )

    def _related_owner_keys(self, owner_key: str) -> tuple[str, ...]:
        if owner_key in self.tile_class_keys:
            return tuple(dict.fromkeys((self.tile_class_keys[owner_key], owner_key)))
        if owner_key in self.class_tile_keys:
            return (owner_key, *sorted(self.class_tile_keys[owner_key]))
        return (owner_key,)

    def _transient_end(self, owner_keys) -> int:
        end = 0
        for owner_key in owner_keys:
            for policy in ("shared", "exclusive"):
                for _, begin, size, _ in self.tiles[owner_key][policy]:
                    end = max(end, begin + size)
        return end

    def _validate_allocations(self) -> None:
        persistent = list(self.persistent_bufs.values())
        for begin, end in persistent:
            if not 0 <= begin <= end <= self.smem_max_bytes:
                raise ValueError("persistent smem allocation exceeds the configured capacity")
        _validate_non_overlapping(persistent, "persistent smem allocations overlap")

        owner_groups = []
        grouped_owners = set()
        for tile_key in self.tile_class_keys:
            owner_keys = self._related_owner_keys(tile_key)
            owner_groups.append(owner_keys)
            grouped_owners.update(owner_keys)
        owner_groups.extend(
            (owner_key,) for owner_key in self.tiles if owner_key not in grouped_owners
        )

        for owner_keys in owner_groups:
            policy_buffers = {
                policy: [info for owner_key in owner_keys for info in self.tiles[owner_key][policy]]
                for policy in ("shared", "exclusive")
            }
            if policy_buffers["shared"] and policy_buffers["exclusive"]:
                raise ValueError(
                    "one tile cannot mix shared and exclusive smem allocation policies"
                )
            intervals = []
            for policy in ("shared", "exclusive"):
                for _, begin, size, _ in policy_buffers[policy]:
                    interval = begin, begin + size
                    if interval[1] > self.smem_max_bytes:
                        raise ValueError("managed smem allocation exceeds the configured capacity")
                    for persistent_interval in persistent:
                        if _overlap(interval, persistent_interval):
                            raise ValueError("persistent and transient smem allocations overlap")
                    intervals.append(interval)
            _validate_non_overlapping(intervals, "transient smem allocations overlap")


def _validate_non_overlapping(intervals, message: str) -> None:
    for index, interval in enumerate(intervals):
        for other in intervals[index + 1 :]:
            if _overlap(interval, other):
                raise ValueError(message)


def _overlap(lhs, rhs) -> bool:
    return lhs[0] < rhs[1] and rhs[0] < lhs[1]


__all__ = ["TIRXSmemManager"]
