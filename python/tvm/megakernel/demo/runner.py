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


def run_dynamic_count(
    counts=(8, 37),
    upper: int = 64,
    *,
    scheduler: str = "dynamic",
    profiler_on: bool = False,
) -> dict:
    """Run the runtime-scalar dispatch demo on the local CUDA device.

    Launches the kernel once per value in ``counts``, re-uploading the queue
    arrays and re-zeroing the event workspace and output between launches
    (the device mutates all three).  Each launch must produce
    ``out[m] = m + 1`` for ``m < count`` and zero elsewhere — an exact check
    on the dispatched (or, with ``scheduler="static"``, the auto-gated) task
    count.  Returns a small report dict.
    """

    from .dynamic import build_dynamic_count_spec

    spec = build_dynamic_count_spec(upper)
    spec.validate()
    options = LoweringOptions(scheduler=scheduler, attrs={"profiler": profiler_on})
    build = build_runtime_kernel(spec, options)
    lib = tvm.compile(build.module, tvm.target.Target("cuda"), tir_pipeline="tirx")

    dev = tvm.cuda(0)
    count_host = np.zeros((1,), dtype=np.int32)
    count_dev = tvm.runtime.tensor(count_host, dev)
    out_dev = tvm.runtime.tensor(np.zeros((upper,), dtype=np.int32), dev)
    workspace_dev = tvm.runtime.tensor(np.zeros((build.event_workspace_size,), dtype=np.int32), dev)
    args = [count_dev, out_dev, workspace_dev]
    queue = {}
    if scheduler == "dynamic":
        queue = {
            "tasks": tvm.runtime.tensor(build.queue_tasks.copy(), dev),
            "head": tvm.runtime.tensor(build.queue_head.copy(), dev),
            "tail": tvm.runtime.tensor(build.queue_tail.copy(), dev),
        }
        args += [queue["tasks"], queue["head"], queue["tail"]]
    else:
        queue = {"exec": tvm.runtime.tensor(build.exec_queue.copy(), dev)}
        args.append(queue["exec"])
    if profiler_on:
        args.append(
            tvm.runtime.tensor(
                np.zeros((MegaKernelWrapper.PROFILER_BUFFER_SIZE,), dtype=np.uint64),
                dev,
            )
        )
    kernel = lib[spec.name]

    checks = []
    for count in counts:
        # Reset everything the device mutates: queue arrays, workspace, output.
        count_dev.copyfrom(np.array([count], dtype=np.int32))
        out_dev.copyfrom(np.zeros((upper,), dtype=np.int32))
        workspace_dev.copyfrom(np.zeros((build.event_workspace_size,), dtype=np.int32))
        if scheduler == "dynamic":
            queue["tasks"].copyfrom(build.queue_tasks)
            queue["head"].copyfrom(build.queue_head)
            queue["tail"].copyfrom(build.queue_tail)
        else:
            queue["exec"].copyfrom(build.exec_queue)
        kernel(*args)
        dev.sync()
        expected = np.zeros((upper,), dtype=np.int32)
        expected[:count] = np.arange(1, count + 1, dtype=np.int32)
        np.testing.assert_array_equal(out_dev.numpy(), expected)
        checks.append({"count": count, "ok": True})
    return {
        "upper": upper,
        "checks": checks,
        "seed_tasks": len(build.central_tasks),
        "event_workspace_size": build.event_workspace_size,
    }


def run_case_b(
    in_pairs=((3, 5), (20, 17)),
    upper: int = 64,
) -> dict:
    """Run the case-B demo: the pusher tile itself produces the count.

    Proves the post-run push placement end-to-end: each launch computes the
    dispatch count on device from ``in_vals`` and must produce
    ``out[m] = m + 1`` for ``m < in_vals[0] + in_vals[1]``.
    """

    from .dynamic import build_case_b_spec

    spec = build_case_b_spec(upper)
    spec.validate()
    build = build_runtime_kernel(spec, LoweringOptions(scheduler="dynamic"))
    lib = tvm.compile(build.module, tvm.target.Target("cuda"), tir_pipeline="tirx")

    dev = tvm.cuda(0)
    in_dev = tvm.runtime.tensor(np.zeros((2,), dtype=np.int32), dev)
    count_dev = tvm.runtime.tensor(np.zeros((1,), dtype=np.int32), dev)
    out_dev = tvm.runtime.tensor(np.zeros((upper,), dtype=np.int32), dev)
    workspace_dev = tvm.runtime.tensor(np.zeros((build.event_workspace_size,), dtype=np.int32), dev)
    tasks_dev = tvm.runtime.tensor(build.queue_tasks.copy(), dev)
    head_dev = tvm.runtime.tensor(build.queue_head.copy(), dev)
    tail_dev = tvm.runtime.tensor(build.queue_tail.copy(), dev)
    args = [in_dev, count_dev, out_dev, workspace_dev, tasks_dev, head_dev, tail_dev]
    kernel = lib[spec.name]

    checks = []
    for pair in in_pairs:
        count = pair[0] + pair[1]
        in_dev.copyfrom(np.array(pair, dtype=np.int32))
        out_dev.copyfrom(np.zeros((upper,), dtype=np.int32))
        workspace_dev.copyfrom(np.zeros((build.event_workspace_size,), dtype=np.int32))
        tasks_dev.copyfrom(build.queue_tasks)
        head_dev.copyfrom(build.queue_head)
        tail_dev.copyfrom(build.queue_tail)
        kernel(*args)
        dev.sync()
        expected = np.zeros((upper,), dtype=np.int32)
        expected[:count] = np.arange(1, count + 1, dtype=np.int32)
        np.testing.assert_array_equal(out_dev.numpy(), expected)
        checks.append({"count": count, "ok": True})
    return {"upper": upper, "checks": checks}


if __name__ == "__main__":
    for size in (256, 1024):
        report = run_two_stage_reduce(m=size)
        print(
            f"M={report['m']}: OK (max abs err {report['max_abs_err']:.3e}, "
            f"{report['central_tasks']} central tasks)"
        )
    dynamic_report = run_dynamic_count()
    for check in dynamic_report["checks"]:
        print(
            f"dynamic count={check['count']}: OK "
            f"({dynamic_report['seed_tasks']} seed tasks, "
            f"ws {dynamic_report['event_workspace_size']})"
        )
    static_report = run_dynamic_count(scheduler="static")
    for check in static_report["checks"]:
        print(f"static-guard count={check['count']}: OK")
    case_b_report = run_case_b()
    for check in case_b_report["checks"]:
        print(f"case-B count={check['count']}: OK")
