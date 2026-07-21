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
"""GPU runner for the two-stage reduce demo on the runtime static builder.

Builds the demo spec with ``build_runtime_kernel`` (``scheduler="static"``),
compiles the module for CUDA, uploads the derived static exec queue, launches
the persistent kernel, and checks the result against the torch reference
``torch.sum(A, dim=1, keepdim=True)``.
"""

from __future__ import annotations

import numpy as np

import tvm
from tvm.megakernel.runtime import MegaKernelWrapper
from tvm.megakernel.transform import LoweringOptions, build_runtime_kernel

from .dsl import build_kernel_spec


def run_two_stage_reduce(
    m: int = 1024,
    n: int = 1024,
    *,
    block_m: int = 64,
    block_n: int = 64,
    profiler_on: bool = False,
    seed: int = 0,
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> dict:
    """Run the two-stage reduce demo on the local CUDA device and verify it.

    Returns a small report dict; raises ``AssertionError`` on a numerical
    mismatch.  Requires a CUDA device and torch (for the reference).
    """

    import torch  # local import: only the reference needs torch

    spec = build_kernel_spec(m=m, n=n, block_m=block_m, block_n=block_n)
    spec.validate()
    options = LoweringOptions(scheduler="static", attrs={"profiler": profiler_on})
    build = build_runtime_kernel(spec, options)
    lib = tvm.compile(build.module, tvm.target.Target("cuda"), tir_pipeline="tirx")

    dev = tvm.cuda(0)
    rng = np.random.default_rng(seed)
    a_host = rng.standard_normal((m, n), dtype=np.float32)
    num_block_n = n // block_n
    a_dev = tvm.runtime.tensor(a_host, dev)
    b_dev = tvm.runtime.tensor(np.zeros((m, num_block_n), dtype=np.float32), dev)
    c_dev = tvm.runtime.tensor(np.zeros((m, 1), dtype=np.float32), dev)

    args = [a_dev, b_dev, c_dev]
    if build.event_workspace_size:
        # The event protocol requires a zeroed workspace at launch.
        args.append(
            tvm.runtime.tensor(np.zeros((build.event_workspace_size,), dtype=np.int32), dev)
        )
    args.append(tvm.runtime.tensor(build.exec_queue, dev))
    if profiler_on:
        args.append(
            tvm.runtime.tensor(
                np.zeros((MegaKernelWrapper.PROFILER_BUFFER_SIZE,), dtype=np.uint64),
                dev,
            )
        )

    kernel = lib[spec.name]
    kernel(*args)
    dev.sync()

    result = c_dev.numpy()
    reference = torch.sum(torch.from_numpy(a_host).cuda(), dim=1, keepdim=True).cpu().numpy()
    np.testing.assert_allclose(result, reference, rtol=rtol, atol=atol)
    return {
        "m": m,
        "n": n,
        "max_abs_err": float(np.max(np.abs(result - reference))),
        "central_tasks": len(build.central_tasks),
        "event_workspace_size": build.event_workspace_size,
    }


if __name__ == "__main__":
    for size in (256, 1024):
        report = run_two_stage_reduce(m=size)
        print(
            f"M={report['m']}: OK (max abs err {report['max_abs_err']:.3e}, "
            f"{report['central_tasks']} central tasks)"
        )
