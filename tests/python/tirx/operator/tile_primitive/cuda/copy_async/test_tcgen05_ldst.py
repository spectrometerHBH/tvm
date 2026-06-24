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
# pylint: disable=invalid-name, missing-function-docstring
"""Unit tests for tcgen05.ld/st copy_async dispatch helpers."""

import tvm
from tvm.script import tirx as T
from tvm.backend.cuda.operator.tile_primitive.copy_async.tcgen05_ldst import (
    _can_prove_divisible,
)


def test_can_prove_divisible_uint32_offset():
    analyzer = tvm.arith.Analyzer()
    u = T.Var("u", "uint32")
    assert _can_prove_divisible(analyzer, u, 1) is True
    assert _can_prove_divisible(analyzer, u * tvm.tirx.const(2, "uint32"), 2) is True
    assert _can_prove_divisible(analyzer, u, 2) is False


def test_can_prove_divisible_signed_unaffected():
    analyzer = tvm.arith.Analyzer()
    i = T.Var("i", "int32")
    assert _can_prove_divisible(analyzer, i, 1) is True
    assert _can_prove_divisible(analyzer, i * tvm.tirx.const(4, "int32"), 4) is True
    assert _can_prove_divisible(analyzer, i, 4) is False


if __name__ == "__main__":
    tvm.testing.main()
