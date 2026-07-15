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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Canonical NVIDIA IKET workload.

Run with the locked CUTLASS DSL profile::

  TVM_IKET_OFFICIAL_PROFILE=cutlass-4.6.1 \
    run-iket profile --postprocess all -- \
    python tests/python/tirx/iket_profile_workload.py
"""

import numpy as np

import tvm
from tvm.script import tirx as T
from tvm.tirx.bench import IketProfiler


@T.prim_func
def canonical_iket_workload(out: T.Buffer((32,), "int32")):
    T.device_entry()
    iket = IketProfiler()
    tx = T.thread_id([32])
    token = iket.sentinel_token("token")
    iket.range_end(token)
    token = iket.range_start("token")
    iket.mark("checkpoint")
    iket.range_end(token)
    iket.range_push("stack")
    iket.mark("inside_stack")
    iket.range_pop()
    out[tx] = tx + 1


def main():
    executable = IketProfiler().compile(
        canonical_iket_workload,
        target=tvm.target.Target({"kind": "cuda", "arch": "sm_100a"}),
        tir_pipeline="tirx",
    )
    out = tvm.runtime.empty((32,), "int32", device=tvm.cuda())
    executable["canonical_iket_workload"](out)
    tvm.cuda().sync()
    np.testing.assert_array_equal(out.numpy(), np.arange(32, dtype=np.int32) + 1)


if __name__ == "__main__":
    main()
