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

import pytest
import tvm_ffi

import tvm
import tvm.testing
from tvm.script import tirx as T
from tvm.support import nvcc


def test_cuda_target_fast_math_attribute():
    default_target = tvm.target.Target({"kind": "cuda", "arch": "sm_80"})
    precise_target = tvm.target.Target({"kind": "cuda", "arch": "sm_80", "fast-math": False})

    assert bool(default_target.attrs["fast-math"])
    assert not bool(precise_target.attrs["fast-math"])


@pytest.mark.parametrize("compiler", ["nvcc", "nvrtc"])
@pytest.mark.parametrize("target_fast_math", [False, True])
def test_compile_cuda_uses_target_fast_math(monkeypatch, compiler, target_fast_math):
    captured = {}

    def fake_compile(
        code,
        target_format,
        arch,
        options,
        path_target,
        use_nvshmem,
        use_fast_math,
    ):
        captured["use_fast_math"] = use_fast_math
        return bytearray(b"compiled")

    monkeypatch.setattr(nvcc, f"_compile_cuda_{compiler}", fake_compile)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80", "fast-math": target_fast_math})
    with target:
        result = nvcc.compile_cuda("", compiler=compiler)

    assert result == bytearray(b"compiled")
    assert captured["use_fast_math"] is target_fast_math


def test_compile_cuda_explicit_fast_math_overrides_target(monkeypatch):
    captured = {}

    def fake_compile(
        code,
        target_format,
        arch,
        options,
        path_target,
        use_nvshmem,
        use_fast_math,
    ):
        captured["use_fast_math"] = use_fast_math
        return bytearray(b"compiled")

    monkeypatch.setattr(nvcc, "_compile_cuda_nvrtc", fake_compile)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80", "fast-math": True})
    with target:
        nvcc.compile_cuda("", compiler="nvrtc", use_fast_math=False)

    assert captured["use_fast_math"] is False


def test_cuda_build_callback_observes_explicit_target():
    if tvm.get_global_func("ffi.Module.create.cuda", allow_missing=True) is None:
        pytest.skip("CUDA runtime module factory is not enabled")

    @T.prim_func
    def kernel(data: T.Buffer((1,), "float32")):
        T.device_entry()
        thread = T.thread_id([1])
        data[thread] = data[thread] + T.float32(1)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_80", "fast-math": False})
    original_callback = tvm.get_global_func("tvm_callback_cuda_compile")
    captured = {}

    @tvm_ffi.register_global_func("tvm_callback_cuda_compile", override=True)
    def fake_compile(code):
        current = tvm.target.Target.current(allow_none=True)
        captured["source"] = code
        captured["target"] = current
        return bytearray(
            b"//\n.version 8.0\n.target sm_80\n.address_size 64\n"
            b".visible .entry kernel() { ret; }\n"
        )

    try:
        tvm.compile(tvm.IRModule({"kernel": kernel}), target=target, tir_pipeline="tirx")
    finally:
        tvm.register_global_func("tvm_callback_cuda_compile", original_callback, override=True)

    assert "__global__ void" in captured["source"]
    assert "kernel(" in captured["source"]
    assert tvm_ffi.structural_equal(captured["target"], target)
    assert not bool(captured["target"].attrs["fast-math"])


def test_nvrtc_program_name_tracks_format_and_options():
    fast = nvcc._nvrtc_program_name("cubin", [b"--use_fast_math"])
    precise = nvcc._nvrtc_program_name("cubin", [])
    ptx = nvcc._nvrtc_program_name("ptx", [b"--use_fast_math"])

    assert fast == nvcc._nvrtc_program_name("cubin", [b"--use_fast_math"])
    assert len({fast, precise, ptx}) == 3
    assert all(name.startswith("tvm_kernels_") for name in (fast, precise, ptx))
    assert all(name.endswith(".cu") for name in (fast, precise, ptx))


if __name__ == "__main__":
    tvm.testing.main()
