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
"""32-bit task wire format shared by the device and host queue paths.

One packed task is a single ``int32``: ``task_type`` in the low bits, then
``m_idx``, ``n_idx``, and ``k_idx``.  The production layout is
``task_type: [0:5], m_idx: [5:18], n_idx: [18:28], k_idx: [28:32]``; the bit
widths are parameters so other layouts can be plugged in, but every queue
producer and consumer in one kernel must share the same ``TaskPacking``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tvm.script import tirx as T


@dataclass(frozen=True)
class TaskPacking:
    """Bit-field widths of one packed 32-bit task."""

    task_type_bits: int = 5
    m_bits: int = 13
    n_bits: int = 10
    k_bits: int = 4

    def __post_init__(self):
        for width in (self.task_type_bits, self.m_bits, self.n_bits, self.k_bits):
            if not isinstance(width, int) or width <= 0:
                raise ValueError("task packing bit widths must be positive integers")
        if self.task_type_bits + self.m_bits + self.n_bits + self.k_bits != 32:
            raise ValueError("task packing bit widths must add up to 32")

    @property
    def max_task_type(self) -> int:
        return 1 << self.task_type_bits

    @property
    def max_m_idx(self) -> int:
        return 1 << self.m_bits

    @property
    def max_n_idx(self) -> int:
        return 1 << self.n_bits

    @property
    def max_k_idx(self) -> int:
        return 1 << self.k_bits

    @property
    def m_shift(self) -> int:
        return self.task_type_bits

    @property
    def n_shift(self) -> int:
        return self.task_type_bits + self.m_bits

    @property
    def k_shift(self) -> int:
        return self.task_type_bits + self.m_bits + self.n_bits


_DEFAULT_PACKING = TaskPacking()


def _check_fields(m_idx, n_idx, k_idx, task_type, packing: TaskPacking):
    if not (
        task_type < packing.max_task_type
        and m_idx < packing.max_m_idx
        and n_idx < packing.max_n_idx
        and k_idx < packing.max_k_idx
    ):
        raise ValueError(
            f"task fields out of range: task_type={task_type}, m_idx={m_idx}, "
            f"n_idx={n_idx}, k_idx={k_idx} for packing {packing}"
        )


def pack_into_32bit(m_idx, n_idx, k_idx, task_type, host=True, debug=False, packing=None):
    """Pack one task into an int32, on the host or in emitted device code."""

    if packing is None:
        packing = _DEFAULT_PACKING
    if host:
        if debug:
            _check_fields(m_idx, n_idx, k_idx, task_type, packing)
        return (
            np.int64(
                [
                    task_type
                    | (m_idx << packing.m_shift)
                    | (n_idx << packing.n_shift)
                    | (k_idx << packing.k_shift)
                ]
            )
            .astype(np.int32)
            .item()
        )
    if debug:
        T.cuda.trap_when_assert_failed(task_type < packing.max_task_type)
        T.cuda.trap_when_assert_failed(m_idx < packing.max_m_idx)
        T.cuda.trap_when_assert_failed(n_idx < packing.max_n_idx)
        T.cuda.trap_when_assert_failed(k_idx < packing.max_k_idx)
    return (
        task_type
        | (m_idx << packing.m_shift)
        | (n_idx << packing.n_shift)
        | (k_idx << packing.k_shift)
    )


def unpack_from_32bit_host(task_info, packing=None):
    """Pure-host mirror of the emitted ``unpack_from_32bit`` device helper."""

    if packing is None:
        packing = _DEFAULT_PACKING
    task_info = np.int32(task_info).item()
    task_type = task_info & (packing.max_task_type - 1)
    m_idx = (task_info >> packing.m_shift) & (packing.max_m_idx - 1)
    n_idx = (task_info >> packing.n_shift) & (packing.max_n_idx - 1)
    k_idx = (task_info >> packing.k_shift) & (packing.max_k_idx - 1)
    return task_type, m_idx, n_idx, k_idx


def _unpack_from_32bit_code(packing: TaskPacking) -> str:
    signature = (
        "__forceinline__ __device__ void unpack_from_32bit("
        "int32_t task_info, int32_t* task_type_ptr, int32_t* m_idx_ptr, "
        "int32_t* n_idx_ptr, int32_t* k_idx_ptr) {"
    )
    return f"""
{signature}
    *task_type_ptr = task_info & 0b{packing.max_task_type - 1:b};
    *m_idx_ptr = (task_info >> {packing.m_shift}) & 0b{packing.max_m_idx - 1:b};
    *n_idx_ptr = (task_info >> {packing.n_shift}) & 0b{packing.max_n_idx - 1:b};
    *k_idx_ptr = (task_info >> {packing.k_shift}) & 0b{packing.max_k_idx - 1:b};
}}
"""


@T.inline
def unpack_from_32bit(task_info, task_type_ptr, m_idx_ptr, n_idx_ptr, k_idx_ptr, packing=None):
    T.cuda.func_call(
        "unpack_from_32bit",
        task_info,
        task_type_ptr,
        m_idx_ptr,
        n_idx_ptr,
        k_idx_ptr,
        source_code=_unpack_from_32bit_code(packing or _DEFAULT_PACKING),
    )


__all__ = [
    "TaskPacking",
    "pack_into_32bit",
    "unpack_from_32bit",
    "unpack_from_32bit_host",
]
