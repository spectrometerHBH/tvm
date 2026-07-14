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

from tvm.support import nvcc


@pytest.mark.parametrize("value", ["1", "true", "ON", "yes"])
def test_cuda_use_fast_math_enabled(value, monkeypatch):
    monkeypatch.setenv("TVM_CUDA_USE_FAST_MATH", value)
    assert nvcc._cuda_use_fast_math()


@pytest.mark.parametrize("value", ["0", "false", "OFF", "no"])
def test_cuda_use_fast_math_disabled(value, monkeypatch):
    monkeypatch.setenv("TVM_CUDA_USE_FAST_MATH", value)
    assert not nvcc._cuda_use_fast_math()


def test_cuda_use_fast_math_defaults_enabled(monkeypatch):
    monkeypatch.delenv("TVM_CUDA_USE_FAST_MATH", raising=False)
    assert nvcc._cuda_use_fast_math()


def test_cuda_use_fast_math_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("TVM_CUDA_USE_FAST_MATH", "sometimes")
    with pytest.raises(ValueError, match="TVM_CUDA_USE_FAST_MATH"):
        nvcc._cuda_use_fast_math()
