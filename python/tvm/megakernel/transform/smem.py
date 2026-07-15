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
"""Managed shared-memory support for parser-style megakernel lowering."""

from __future__ import annotations

from typing import Literal

from tvm.script import tirx as T

from ..dsl import SmemAllocRecord, SmemManager


class TIRXSmemManager(SmemManager):
    """Concrete TIRX shared-memory manager used by the default backend."""

    def __init__(self, smem_max_bytes: int, chunk_size: int):
        if smem_max_bytes <= 0 or chunk_size <= 0:
            raise ValueError("shared-memory sizes must be positive")
        self.smem_max_bytes = smem_max_bytes
        self.chunk_size = chunk_size
        self.chunk_num = smem_max_bytes // chunk_size
        if self.chunk_num > 32:
            raise ValueError("shared-memory manager supports at most 32 chunks")

        regular = T.SMEMPool()
        persistent = T.SMEMPool(regular.ptr)
        persistent.move_base_to(chunk_size * max(0, self.chunk_num - 1))
        self.pool_allocator = {
            "persistent": persistent,
            "shared": regular,
            "exclusive": regular,
        }
        self.tiles = {}
        self.runtime_tile_chunk_count = {}
        self.bufs = {}
        self.persistent_bufs = {}
        self.cur_tile_name = ""
        self.exist_bufs = {}
        self.records: list[SmemAllocRecord] = []

    def set_tile(self, tile) -> None:
        """Begin declaration-time allocation recording for a tile."""

        self.cur_tile_name = "default" if tile is None else str(tile)
        self.tiles[self.cur_tile_name] = [
            -1,
            {"exclusive": [], "shared": []},
            [0 for _ in range(self.chunk_num)],
        ]
        self.runtime_tile_chunk_count[self.cur_tile_name] = [
            [0 for _ in range(self.chunk_num)],
            [0 for _ in range(self.chunk_num)],
        ]
        self.pool_allocator["shared"].move_base_to(0)

    def enter_tile_runtime(self, tile) -> None:
        self.cur_tile_name = str(tile)

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
        if policy not in self.VALID_POLICIES:
            valid = ", ".join(sorted(self.VALID_POLICIES))
            raise ValueError(f"unsupported smem policy {policy!r}; expected {valid}")
        if split <= 0:
            raise ValueError("smem split must be positive")
        if name is not None:
            suffix = self.exist_bufs.setdefault(name, 0)
            self.exist_bufs[name] += 1
            if suffix:
                name = f"{name}{suffix}"

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
            buffer_type,
            axis_separators,
            layout,
        )
        end = allocator.offset
        size = end - begin
        if size % split:
            raise ValueError("smem allocation size must be divisible by split")

        if policy == "persistent":
            self.persistent_bufs[buffer] = (begin, end)
        elif self.cur_tile_name in self.tiles:
            info = (split, begin, size, policy)
            self.tiles[self.cur_tile_name][0] = max(
                self.tiles[self.cur_tile_name][0], (end - 1) // self.chunk_size
            )
            self.tiles[self.cur_tile_name][1][policy].append(info)
            self.bufs[buffer] = info
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

    def commit(self) -> None:
        self.pool_allocator["shared"].commit(self.smem_max_bytes)

    def acquire_all(self, level="cta") -> None:
        T.evaluate(T.call_extern("void", "tirx.megakernel.smem.wait_all", level))

    def release_all(self, level="cta") -> None:
        T.evaluate(T.call_extern("void", "tirx.megakernel.smem.release_all", level))

    def advance(self) -> None:
        T.evaluate(T.call_extern("void", "tirx.megakernel.smem.advance"))


__all__ = ["TIRXSmemManager"]
