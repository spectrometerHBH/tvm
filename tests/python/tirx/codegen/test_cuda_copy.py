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
"""Tests for PTX ld/st vector copies via local scratch (shared ↔ shared)."""

import numpy as np
import pytest

import tvm
from tvm.script import tirx as T
from tvm.testing import env

DEV = tvm.cuda(0)
TARGET = tvm.target.Target("cuda")


def _build_and_run(func, *np_args):
    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=TARGET, tir_pipeline="tirx")
    rt_args = [tvm.runtime.tensor(a, device=DEV) for a in np_args]
    mod(*rt_args)
    return (*tuple(a.numpy() for a in rt_args), mod)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_ptx_ld_st_shared_copy_128b():
    """128b shared copy via local scratch + PTX ld/st."""

    # fmt: off
    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (4,), "float32")
        T.device_entry()
        T.cta_id([1])
        T.warp_id([1])
        lane = T.lane_id([32])
        src_buf = T.alloc_buffer((4,), "float32", scope="shared")
        dst_buf = T.alloc_buffer((4,), "float32", scope="shared")
        tmp = T.alloc_local((4,), "float32")
        if lane < 4:
            src_buf[lane] = T.float32(lane + 1)
        T.cuda.cta_sync()
        if lane == 0:
            T.ptx.ld(
                src_buf.ptr_to([0]),
                "uint32",
                "u32",
                dst=tmp.ptr_to([0]),
                space="shared",
                vec="v4",
            )
            T.ptx.st(
                dst_buf.ptr_to([0]),
                src=tmp.ptr_to([0]),
                space="shared",
                vec="v4",
                ptx_type="u32",
            )
        T.cuda.cta_sync()
        if lane < 4:
            out[lane] = dst_buf[lane]
        # fmt: on

    out_np = np.zeros(4, dtype="float32")
    result, mod = _build_and_run(func, out_np)
    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0])
    source = mod.mod.imports[0].inspect_source()
    assert "tvm_builtin_ptx_ld" in source
    assert "tvm_builtin_ptx_st" in source
    assert "ld.shared.v4" in source
    assert "st.shared.v4" in source


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_ptx_ld_st_shared_copy_32b():
    """32b shared copy via local scratch + PTX ld/st."""

    # fmt: off
    @T.prim_func
    def func(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (1,), "float32")
        T.device_entry()
        T.cta_id([1])
        T.warp_id([1])
        lane = T.lane_id([32])
        src_buf = T.alloc_buffer((1,), "float32", scope="shared")
        dst_buf = T.alloc_buffer((1,), "float32", scope="shared")
        tmp = T.alloc_local((1,), "float32")
        if lane == 0:
            src_buf[0] = T.float32(42)
        T.cuda.cta_sync()
        if lane == 0:
            T.ptx.ld(
                src_buf.ptr_to([0]),
                "uint32",
                "u32",
                dst=tmp.ptr_to([0]),
                space="shared",
                vec="",
            )
            T.ptx.st(
                dst_buf.ptr_to([0]),
                src=tmp.ptr_to([0]),
                space="shared",
                vec="",
                ptx_type="u32",
            )
        T.cuda.cta_sync()
        if lane == 0:
            out[0] = dst_buf[0]
        # fmt: on

    out_np = np.zeros(1, dtype="float32")
    result, mod = _build_and_run(func, out_np)
    np.testing.assert_allclose(result, [42.0])
    assert "tvm_builtin_ptx_ld" in mod.mod.imports[0].inspect_source()


@pytest.mark.parametrize(
    "vec_len,vec,ptx_type,return_type",
    [(4, "v4", "u32", "uint32"), (2, "v2", "u32", "uint32"), (1, "", "u32", "uint32")],
)
def test_ptx_ld_st_codegen_function_names(vec_len, vec, ptx_type, return_type):
    """Verify PTX ld/st emit the expected helper names."""

    # fmt: off
    @T.prim_func
    def func(dummy_ptr: T.handle):
        dummy = T.match_buffer(dummy_ptr, (16,), "uint8")
        T.device_entry()
        T.cta_id([1])
        T.warp_id([1])
        lane = T.lane_id([32])
        a = T.alloc_buffer((vec_len,), "uint32", scope="shared")
        b = T.alloc_buffer((vec_len,), "uint32", scope="shared")
        tmp = T.alloc_local((vec_len,), "uint32")
        if lane == 0:
            T.ptx.ld(
                a.ptr_to([0]),
                return_type,
                ptx_type,
                dst=tmp.ptr_to([0]),
                space="shared",
                vec=vec,
            )
            T.ptx.st(
                b.ptr_to([0]),
                src=tmp.ptr_to([0]),
                space="shared",
                vec=vec,
                ptx_type=ptx_type,
            )
            dummy[0] = T.uint8(0)
        # fmt: on

    mod = tvm.IRModule({"main": func})
    mod = tvm.compile(mod, target=TARGET, tir_pipeline="tirx")
    source = mod.mod.imports[0].inspect_source()
    assert "tvm_builtin_ptx_ld" in source
    assert "tvm_builtin_ptx_st" in source
