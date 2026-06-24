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
from tvm.tirx.cuda.operator.tile_primitive.copy._common import (
    copy_ptx_form,
    copy_ptx_ld_return_type,
)

DEV = tvm.cuda(0)
TARGET = tvm.target.Target("cuda")


def _build_and_run(func, *np_args):
    mod = tvm.compile(tvm.IRModule({"main": func}), target=TARGET, tir_pipeline="tirx")
    rt_args = [tvm.runtime.tensor(a, device=DEV) for a in np_args]
    mod(*rt_args)
    return (*tuple(a.numpy() for a in rt_args), mod)


def _shared_scratch_copy_kernel(num_bytes: int):
    """Build shared → local scratch → shared copy kernel for ``num_bytes`` width."""
    vec, ptx_type = copy_ptx_form(num_bytes)
    return_type = copy_ptx_ld_return_type(ptx_type)

    if num_bytes == 16:

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
                    return_type,
                    ptx_type,
                    dst=tmp.ptr_to([0]),
                    space="shared",
                    vec=vec,
                )
                T.ptx.st(
                    dst_buf.ptr_to([0]),
                    src=tmp.ptr_to([0]),
                    space="shared",
                    vec=vec,
                    ptx_type=ptx_type,
                )
            T.cuda.cta_sync()
            if lane < 4:
                out[lane] = dst_buf[lane]

    elif num_bytes == 8:

        @T.prim_func
        def func(out_ptr: T.handle):
            out = T.match_buffer(out_ptr, (2,), "float32")
            T.device_entry()
            T.cta_id([1])
            T.warp_id([1])
            lane = T.lane_id([32])
            src_buf = T.alloc_buffer((2,), "float32", scope="shared")
            dst_buf = T.alloc_buffer((2,), "float32", scope="shared")
            tmp = T.alloc_local((2,), "float32")
            if lane < 2:
                src_buf[lane] = T.float32(lane + 10)
            T.cuda.cta_sync()
            if lane == 0:
                T.ptx.ld(
                    src_buf.ptr_to([0]),
                    return_type,
                    ptx_type,
                    dst=tmp.ptr_to([0]),
                    space="shared",
                    vec=vec,
                )
                T.ptx.st(
                    dst_buf.ptr_to([0]),
                    src=tmp.ptr_to([0]),
                    space="shared",
                    vec=vec,
                    ptx_type=ptx_type,
                )
            T.cuda.cta_sync()
            if lane < 2:
                out[lane] = dst_buf[lane]

    elif num_bytes == 4:

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
                    return_type,
                    ptx_type,
                    dst=tmp.ptr_to([0]),
                    space="shared",
                    vec=vec,
                )
                T.ptx.st(
                    dst_buf.ptr_to([0]),
                    src=tmp.ptr_to([0]),
                    space="shared",
                    vec=vec,
                    ptx_type=ptx_type,
                )
            T.cuda.cta_sync()
            if lane == 0:
                out[0] = dst_buf[0]

    elif num_bytes == 2:

        @T.prim_func
        def func(out_ptr: T.handle):
            out = T.match_buffer(out_ptr, (1,), "float16")
            T.device_entry()
            T.cta_id([1])
            T.warp_id([1])
            lane = T.lane_id([32])
            src_buf = T.alloc_buffer((1,), "float16", scope="shared")
            dst_buf = T.alloc_buffer((1,), "float16", scope="shared")
            tmp = T.alloc_local((1,), "float16")
            if lane == 0:
                src_buf[0] = T.float16(7)
            T.cuda.cta_sync()
            if lane == 0:
                T.ptx.ld(
                    src_buf.ptr_to([0]),
                    return_type,
                    ptx_type,
                    dst=tmp.ptr_to([0]),
                    space="shared",
                    vec=vec,
                )
                T.ptx.st(
                    dst_buf.ptr_to([0]),
                    src=tmp.ptr_to([0]),
                    space="shared",
                    vec=vec,
                    ptx_type=ptx_type,
                )
            T.cuda.cta_sync()
            if lane == 0:
                out[0] = dst_buf[0]

    elif num_bytes == 1:

        @T.prim_func
        def func(out_ptr: T.handle):
            out = T.match_buffer(out_ptr, (1,), "uint8")
            T.device_entry()
            T.cta_id([1])
            T.warp_id([1])
            lane = T.lane_id([32])
            src_buf = T.alloc_buffer((1,), "uint8", scope="shared")
            dst_buf = T.alloc_buffer((1,), "uint8", scope="shared")
            tmp = T.alloc_local((1,), "uint32")
            if lane == 0:
                src_buf[0] = T.uint8(255)
            T.cuda.cta_sync()
            if lane == 0:
                T.ptx.ld(
                    src_buf.ptr_to([0]),
                    return_type,
                    ptx_type,
                    dst=tmp.ptr_to([0]),
                    space="shared",
                    vec=vec,
                )
                T.ptx.st(
                    dst_buf.ptr_to([0]),
                    src=tmp.ptr_to([0]),
                    space="shared",
                    vec=vec,
                    ptx_type=ptx_type,
                )
            T.cuda.cta_sync()
            if lane == 0:
                out[0] = dst_buf[0]

    else:
        raise ValueError(f"unsupported copy width {num_bytes} bytes")

    return func


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
@pytest.mark.parametrize(
    "num_bytes,np_dtype,expected",
    [
        pytest.param(16, np.float32, np.array([1, 2, 3, 4], dtype=np.float32), id="128b"),
        pytest.param(8, np.float32, np.array([10, 11], dtype=np.float32), id="64b"),
        pytest.param(4, np.float32, np.array([42], dtype=np.float32), id="32b"),
        pytest.param(2, np.float16, np.array([7], dtype=np.float16), id="16b"),
        pytest.param(1, np.uint8, np.array([255], dtype=np.uint8), id="8b"),
    ],
)
def test_ptx_ld_st_shared_copy_gpu(num_bytes, np_dtype, expected):
    """GPU roundtrip for each supported PTX ld/st copy width (shared → scratch → shared)."""
    kernel = _shared_scratch_copy_kernel(num_bytes)
    out_np = np.zeros(expected.shape, dtype=np_dtype)
    result, mod = _build_and_run(kernel, out_np)
    if np_dtype == np.uint8:
        np.testing.assert_array_equal(result, expected)
    else:
        np.testing.assert_allclose(result, expected)
    src = mod.mod.imports[0].inspect_source("cuda")
    assert "tvm_builtin_ptx_ld" in src
    assert "tvm_builtin_ptx_st" in src
    vec, _ptx_type = copy_ptx_form(num_bytes)
    if vec == "v4":
        assert "ld.shared.v4" in src
        assert "st.shared.v4" in src
    elif vec == "v2":
        assert "ld.shared.v2" in src
        assert "st.shared.v2" in src
