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
"""Device-code emission helpers shared by the runtime building blocks.

These are migrations of the emission helpers in the production
``tirx_kernels.megakernel.utils.utils`` that the migrated runtime classes
use: warp collectives, local relaxed/release atomics, and queue load/store
primitives.
"""

from __future__ import annotations

from tvm.script import tirx as T


def any_sync(mask, pred):
    return T.cuda.func_call(
        "any_sync",
        mask,
        pred,
        source_code="""
__forceinline__ __device__ int any_sync(unsigned mask, int pred) {
  return __any_sync(mask, pred);
}
""",
        return_type="int32",
    )


def gt(lhs, rhs):
    return T.cuda.func_call(
        "gt",
        lhs,
        rhs,
        source_code="""
__forceinline__ __device__ bool gt(int32_t a, int32_t b) {
    return a > b;
}
""",
        return_type="bool",
    )


def atomic_add_int32_local_release(addr, value):
    func = """
__forceinline__ __device__ int32_t atomic_add_int32_release(int32_t* addr, int32_t value) {
    int32_t old_value;
    asm volatile ("atom.release.gpu.global.add.s32 %0, [%1], %2;"
                  : "=r"(old_value)
                  : "l"(addr), "r"(value)
                  : "memory");
    return old_value;
}
"""
    return T.cuda.func_call(
        "atomic_add_int32_release", addr, value, source_code=func, return_type="int32"
    )


def atomic_add_int32_local(addr, value):
    func = """
__forceinline__ __device__ int32_t atomic_add_int32(int32_t* addr, int32_t value) {
    return atomicAdd(addr, value);
}
"""
    return T.cuda.func_call("atomic_add_int32", addr, value, source_code=func, return_type="int32")


def atomic_add_int32(addr, value, release=False):
    if release:
        return atomic_add_int32_local_release(addr, value)
    return atomic_add_int32_local(addr, value)


def stg_local(v, dst_addr):
    func = """
    __forceinline__ __device__ void stg_local(int32_t v, void* dst_addr) {
        asm volatile("st.global.release.gpu.b32 [%0], %1;"
                    :
                    : "l"(dst_addr), "r"(v)
                    : "memory");
    }
    """
    return T.cuda.func_call("stg_local", v, dst_addr, source_code=func)


def stg(v, dst_addr):
    return stg_local(v, dst_addr)


_WHILE_LD_GLOBAL_ACQUIRE_LOAD = (
    'asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\\n" : "=r"(*task_info) '
    ': "l"(addr) : "memory");'
)

_WHILE_LD_GLOBAL_ACQUIRE_CODE = (
    """
__forceinline__ __device__ void while_ld_global_acquire(int32_t* addr, int32_t* task_info) {
  """
    + _WHILE_LD_GLOBAL_ACQUIRE_LOAD
    + """
  while (*task_info == -1) {
    __nanosleep(800);
    """
    + _WHILE_LD_GLOBAL_ACQUIRE_LOAD
    + """
  }
}
"""
)


@T.inline
def while_ld_global_acquire(addr, task_info):
    T.cuda.func_call(
        "while_ld_global_acquire",
        addr,
        task_info,
        source_code=_WHILE_LD_GLOBAL_ACQUIRE_CODE,
    )


@T.inline
def sts(value, dst_addr):
    T.cuda.func_call(
        "sts",
        value,
        dst_addr,
        source_code="""
__forceinline__ __device__ void sts(int32_t v, void* dst_addr) {
    asm volatile("st.shared.b32 [%0], %1;"
                 :
                 : "l"(dst_addr), "r"(v)
                 : "memory");
}
""",
    )


def f_init_const(c):
    return lambda *args: c


__all__ = [
    "any_sync",
    "atomic_add_int32",
    "f_init_const",
    "gt",
    "stg",
    "sts",
    "while_ld_global_acquire",
]
