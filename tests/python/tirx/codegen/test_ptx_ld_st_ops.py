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
"""Unit tests for generic PTX ``T.ptx.ld`` / ``T.ptx.st`` vector copy ops."""

import numpy as np
import pytest

import tvm
from tvm.ir import Op
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env

DEV = tvm.cuda(0)
TARGET = tvm.target.Target("cuda")


def _build_and_run(func, *np_args):
    mod = tvm.compile(tvm.IRModule({"main": func}), target=TARGET, tir_pipeline="tirx")
    rt_args = [tvm.runtime.tensor(a, device=DEV) for a in np_args]
    mod(*rt_args)
    return (*tuple(a.numpy() for a in rt_args), mod)


def test_ptx_ld_st_ops_registered():
    """PTX ld/st must be registered TIR ops and exposed on the T.ptx namespace."""
    for name in ("tirx.ptx.ld", "tirx.ptx.st"):
        Op.get(name)  # raises if unregistered

    for attr in (
        "ld",
        "st",
        "ld_acquire",
        "st_release",
        "ld_volatile",
        "st_volatile",
    ):
        assert hasattr(T.ptx, attr), attr


def test_ptx_ld_st_codegen_emits_shared_asm():
    """Shared ↔ register typed copies must codegen to ``ld.shared`` / ``st.shared``."""

    # fmt: off
    @T.prim_func
    def copy_kernel(d_ptr: T.handle) -> None:
        D = T.match_buffer(d_ptr, (4,), "uint32")
        T.device_entry()
        T.warp_id([4])
        T.cta_id([1])
        T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        smem = T.alloc_buffer((4,), "uint32", scope="shared")
        reg = T.alloc_local((4,), "uint32")
        if tid_in_wg == 0:
            T.ptx.st(
                smem.ptr_to([0]), src=reg.ptr_to([0]), space="shared", vec="v4", ptx_type="u32"
            )
        T.cuda.cta_sync()
        if tid_in_wg == 0:
            T.ptx.ld(
                smem.ptr_to([0]), "uint32", "u32", dst=reg.ptr_to([0]), space="shared", vec="v4"
            )
        Tx.copy(D[0:4], reg[:])
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": copy_kernel}), target=target, tir_pipeline="tirx")
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "ld.shared" in src, "PTX ld did not emit ld.shared"
    assert "st.shared" in src, "PTX st did not emit st.shared"
    assert "tvm_builtin_ptx_ld" in src
    assert "tvm_builtin_ptx_st" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_ptx_ld_st_shared_reg_roundtrip_gpu_v4():
    """128-bit reg --st--> shared --ld--> reg roundtrip on GPU."""

    # fmt: off
    @T.prim_func
    def copy_kernel(d_ptr: T.handle) -> None:
        D = T.match_buffer(d_ptr, (4,), "uint32")
        T.device_entry()
        T.warp_id([4])
        T.cta_id([1])
        T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        smem = T.alloc_buffer((4,), "uint32", scope="shared")
        reg = T.alloc_local((4,), "uint32")
        if tid_in_wg == 0:
            reg[0] = T.uint32(1)
            reg[1] = T.uint32(2)
            reg[2] = T.uint32(3)
            reg[3] = T.uint32(4)
            T.ptx.st(
                smem.ptr_to([0]), src=reg.ptr_to([0]), space="shared", vec="v4", ptx_type="u32"
            )
            reg[0] = T.uint32(0)
            reg[1] = T.uint32(0)
            reg[2] = T.uint32(0)
            reg[3] = T.uint32(0)
            T.ptx.ld(
                smem.ptr_to([0]), "uint32", "u32", dst=reg.ptr_to([0]), space="shared", vec="v4"
            )
            Tx.copy(D[0:4], reg[:])
    # fmt: on

    out_np = np.zeros(4, dtype="uint32")
    result, mod = _build_and_run(copy_kernel, out_np)
    np.testing.assert_array_equal(result, [1, 2, 3, 4])
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "ld.shared.v4" in src
    assert "st.shared.v4" in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_ptx_ld_st_shared_reg_roundtrip_gpu_scalar():
    """32-bit scalar reg --st--> shared --ld--> reg roundtrip on GPU."""

    # fmt: off
    @T.prim_func
    def copy_kernel(d_ptr: T.handle) -> None:
        D = T.match_buffer(d_ptr, (1,), "uint32")
        T.device_entry()
        T.warp_id([4])
        T.cta_id([1])
        T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])
        smem = T.alloc_buffer((1,), "uint32", scope="shared")
        reg = T.alloc_local((1,), "uint32")
        if tid_in_wg == 0:
            reg[0] = T.uint32(42)
            T.ptx.st(
                smem.ptr_to([0]), src=reg.ptr_to([0]), space="shared", vec="", ptx_type="u32"
            )
            reg[0] = T.uint32(0)
            T.ptx.ld(
                smem.ptr_to([0]), "uint32", "u32", dst=reg.ptr_to([0]), space="shared", vec=""
            )
            Tx.copy(D[0:1], reg[:1])
    # fmt: on

    out_np = np.zeros(1, dtype="uint32")
    result, _mod = _build_and_run(copy_kernel, out_np)
    np.testing.assert_array_equal(result, [42])
