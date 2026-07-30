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
"""Tests for the PTX ``shl`` intrinsic."""

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.backend.cuda import op as cuda_op
from tvm.ir import PrimType, assert_structural_equal
from tvm.script import tirx as T
from tvm.testing import env

_SHL_CASES = [
    ("b16", "uint16", np.uint16, 16, "unsigned short", "h"),
    ("b32", "uint32", np.uint32, 32, "unsigned int", "r"),
    ("b64", "uint64", np.uint64, 64, "unsigned long long", "l"),
]


def _get_source(func):
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": func}),
            target=target,
            tir_pipeline="tirx",
        )
    return mod.mod.imports[0].inspect_source(), mod


def _make_shl_kernel(dtype, ptx_type, size):
    @T.prim_func
    def main(
        values: T.Buffer((size,), dtype),
        shifts: T.Buffer((size,), "uint32"),
        output: T.Buffer((size,), dtype),
    ):
        T.device_entry()
        tx = T.thread_id([32])
        if tx < size:
            output[tx] = T.ptx.shl(values[tx], shifts[tx], ptx_type=ptx_type)

    return main


@pytest.mark.parametrize(
    ("ptx_type", "dtype", "_np_dtype", "_bits", "_c_type", "_constraint"), _SHL_CASES
)
def test_ptx_shl_return_dtype(ptx_type, dtype, _np_dtype, _bits, _c_type, _constraint):
    value = tvm.tirx.Var("value", dtype)
    shift = tvm.tirx.Var("shift", "uint32")
    result = cuda_op.ptx_shl(value, shift, ptx_type)

    assert result.op.name == "tirx.ptx.shl"
    assert result.ty == PrimType(dtype)


def test_ptx_shl_rejects_invalid_type():
    with pytest.raises(ValueError, match="invalid ptx_type='u32'"):
        cuda_op.ptx_shl(tvm.tirx.const(1, "uint32"), tvm.tirx.const(1, "uint32"), "u32")


def test_ptx_shl_tvmscript_round_trip():
    @T.prim_func
    def main(
        a16: T.Buffer((2,), "uint16"),
        a32: T.Buffer((2,), "uint32"),
        a64: T.Buffer((2,), "uint64"),
        shift: T.Buffer((2,), "uint32"),
    ):
        a16[1] = T.ptx.shl(a16[1], shift[1], ptx_type="b16")
        a32[1] = T.ptx.shl(a32[1], shift[1], ptx_type="b32")
        a64[1] = T.ptx.shl(a64[1], shift[1], ptx_type="b64")

    script = main.script()
    reparsed = tvm.script.from_source(script)

    assert 'T.ptx.shl(a16[1], shift[1], "b16")' in script
    assert 'T.ptx.shl(a32[1], shift[1], "b32")' in script
    assert 'T.ptx.shl(a64[1], shift[1], "b64")' in script
    assert reparsed.script() == script
    assert_structural_equal(main, reparsed)


@pytest.mark.parametrize(
    ("ptx_type", "dtype", "_np_dtype", "_bits", "c_type", "constraint"), _SHL_CASES
)
def test_ptx_shl_cuda_source(ptx_type, dtype, _np_dtype, _bits, c_type, constraint):
    src, _ = _get_source(_make_shl_kernel(dtype, ptx_type, 1))
    helper_name = f"tvm_builtin_ptx_shl_{ptx_type}"

    assert f"__forceinline__ __device__ {c_type} {helper_name}({c_type} a, unsigned int b)" in src
    assert (
        f'asm("shl.{ptx_type} %0, %1, %2;"'
        f' : "={constraint}"(ret) : "{constraint}"(a), "r"(b));' in src
    )
    assert "asm volatile" not in src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
@pytest.mark.parametrize(
    ("ptx_type", "dtype", "np_dtype", "bits", "_c_type", "_constraint"), _SHL_CASES
)
def test_ptx_shl_runtime_semantics(ptx_type, dtype, np_dtype, bits, _c_type, _constraint):
    shifts = np.array([0, 1, bits - 1, bits, bits + 1, 255], dtype=np.uint32)
    values = np.full(shifts.shape, (1 << bits) - 1, dtype=np_dtype)
    mask = (1 << bits) - 1
    expected = np.array(
        [
            (int(value) << int(shift)) & mask if shift < bits else 0
            for value, shift in zip(values, shifts)
        ],
        dtype=np_dtype,
    )
    _, mod = _get_source(_make_shl_kernel(dtype, ptx_type, len(shifts)))

    def run_and_check():
        dev = tvm.cuda(0)
        values_nd = tvm.runtime.tensor(values, device=dev)
        shifts_nd = tvm.runtime.tensor(shifts, device=dev)
        output_nd = tvm.runtime.tensor(np.zeros_like(values), device=dev)
        mod(values_nd, shifts_nd, output_nd)
        np.testing.assert_array_equal(output_nd.numpy(), expected)

    tvm.testing.run_with_gpu_lock(run_and_check)
