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
"""Regression: reg dispatch nested @T.prim_func must accept static PTX ld/st calls."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TileLayout, tid_in_wg

TVM_ROOT = Path(__file__).resolve().parents[7]
REG_PATH = TVM_ROOT / "python/tvm/backend/cuda/operator/tile_primitive/copy/reg.py"
GMEM_SMEM_PATH = TVM_ROOT / "python/tvm/backend/cuda/operator/tile_primitive/copy/gmem_smem.py"


def _emit_nested_ptx_ld_st(*, space: str):
    """Mirror reg/gmem_smem inline PTX ld/st inside nested @T.prim_func."""

    @T.prim_func(check_well_formed=False)
    def impl():
        smem = T.alloc_buffer((1,), "float32", scope="shared.dyn")
        reg = T.alloc_local((1,), "float32")
        if space == "shared":
            T.ptx.ld(smem.ptr_to([0]), "uint32", "u32", dst=reg.ptr_to([0]), space="shared", vec="")
        else:
            T.ptx.ld(smem.ptr_to([0]), "uint32", "u32", dst=reg.ptr_to([0]), space="global", vec="")
            T.ptx.st(smem.ptr_to([0]), src=reg.ptr_to([0]), space="shared", vec="", ptx_type="u32")

    return impl


def _warpgroup_shared_roundtrip_kernel():
    """Minimal warpgroup shared<->reg roundtrip (same shape as test_reg.py)."""
    n_threads, k = 128, 32
    shape = (n_threads, k)
    full_slices = (slice(0, n_threads), slice(0, k))
    r_layout = TileLayout(S[shape : (1 @ tid_in_wg, 1)])
    s_layout = TileLayout(S[shape])

    @T.prim_func
    def kernel(B_ptr: T.handle) -> None:
        B = T.match_buffer(B_ptr, shape, "float32")
        T.device_entry()
        T.cta_id([1])
        T.warpgroup_id([n_threads // 128])
        T.warp_id_in_wg([4])
        T.lane_id([32])
        T.thread_id_in_wg([128])
        tid = T.thread_id([n_threads])
        A_smem = T.alloc_buffer(shape, "float32", scope="shared", layout=s_layout)
        for kk in range(k):
            A_smem[tid, kk] = T.float32(tid + kk)
        T.cuda.cta_sync()
        R_local = T.alloc_buffer(shape, "float32", scope="local", layout=r_layout)
        Tx.wg.copy(R_local[full_slices], A_smem[full_slices])
        for kk in range(k):
            A_smem[tid, kk] = T.float32(0)
        T.cuda.cta_sync()
        Tx.wg.copy(A_smem[full_slices], R_local[full_slices])
        T.cuda.cta_sync()
        for kk in range(k):
            B[tid, kk] = A_smem[tid, kk]

    return kernel


def _compile_and_check_reg_dispatch(kernel):
    target = tvm.target.Target("cuda")
    with target, warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod = tvm.compile(tvm.IRModule({"main": kernel}), target=target, tir_pipeline="tirx")
    fb = [w.message for w in caught if "copy/fallback" in str(w.message)]
    cuda = mod.mod.imports[0].inspect_source("cuda")
    assert not fb, fb
    assert "threadIdx.x) % 128) == 0" not in cuda
    assert "tvm_builtin_ptx_ld" in cuda
    assert "tvm_builtin_ptx_st" in cuda


@pytest.mark.parametrize("space", ["shared", "global"])
def test_nested_prim_func_ptx_ld_st(space):
    impl = _emit_nested_ptx_ld_st(space=space)
    assert impl is not None


def test_reg_emit_uses_static_ptx_ld_st_calls():
    """reg.py must emit explicit T.ptx.ld / T.ptx.st calls, not copy_bytes*."""
    src = REG_PATH.read_text(encoding="utf-8")
    assert "copy_op(" not in src
    assert "copy_bytes" not in src
    assert "T.ptx.ld(" in src
    assert "T.ptx.st(" in src


def test_gmem_smem_emit_uses_static_ptx_ld_st_calls():
    """gmem_smem.py must emit explicit T.ptx.ld / T.ptx.st via local scratch."""
    src = GMEM_SMEM_PATH.read_text(encoding="utf-8")
    assert "copy_op" not in src
    assert "copy_bytes" not in src
    assert "copy_ptx_form" in src
    assert "T.ptx.ld(" in src
    assert "T.ptx.st(" in src
    assert "alloc_local" in src


def test_reg_warpgroup_shared_roundtrip_no_fallback():
    """Warpgroup shared<->reg copy must stay on reg dispatch (no scalar fallback)."""
    _compile_and_check_reg_dispatch(_warpgroup_shared_roundtrip_kernel())
