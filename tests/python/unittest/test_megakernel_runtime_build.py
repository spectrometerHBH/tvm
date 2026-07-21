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
"""CPU tests for the runtime-based static megakernel builder.

Everything except the final GPU gate runs without a device: emission
structure of the runtime-built static kernel, host-side central-queue
derivation, and ``LoweringOptions.scheduler`` routing.
"""

from enum import Enum
from typing import ClassVar

import numpy as np
import pytest

import tvm
from tvm.megakernel.dsl import KernelSpec, TileImpl
from tvm.megakernel.runtime import HardwareConfig, StaticTileScheduler, unpack_from_32bit_host
from tvm.megakernel.transform import (
    LoweringOptions,
    build_runtime_kernel,
    lower_static_queue_init_to_tirx,
    lower_to_tirx,
    lower_to_tirx_module,
)
from tvm.megakernel.transform.prepare import (
    DEFAULT_END_JOB_ID,
    INIT_EVENT_JOB_ID,
    WAIT_EVENT_INIT_JOB_ID,
)
from tvm.megakernel.transform.runtime_build import (
    _prepare_runtime_plan,
    _resolve_options,
    build_static_queues,
    derive_static_central_tasks,
)
from tvm.script import tirx as T


class _MarkerTile(TileImpl):
    """Tile emitting a distinctive intrinsic per hook for script assertions."""

    @classmethod
    def init_shared_resources(cls, smem_manager):
        T.evaluate(T.cuda.cta_sync())

    @classmethod
    def finalize_shared_resources(cls, smem_manager):
        T.evaluate(T.cuda.nano_sleep(9999))

    def prefetch(self, m_idx, n_idx, k_idx):
        T.evaluate(T.cuda.warp_sync())

    def run(self, m_idx, n_idx, k_idx):
        T.evaluate(T.cuda.thread_fence())


class _WarpScopedTile(_MarkerTile):
    wait_level: ClassVar[str] = "warp"
    wait_mask: ClassVar[int] = 0x3
    notify_scope: ClassVar[tuple[str, int]] = ("warpgroup", 0)


class _WarpgroupWaitTile(_MarkerTile):
    wait_level: ClassVar[str] = "warpgroup"


class _ProfileEvent(Enum):
    STAGE = 7


class _ProfiledTile(_MarkerTile):
    profile_event: ClassVar[Enum] = _ProfileEvent.STAGE


def _chain_kernel(name="chain"):
    """producer -> consumer (warp wait/warpgroup notify) -> sink event chain."""

    kernel = KernelSpec(name)
    src = kernel.tensor("src", (4,), "float32")
    dst = kernel.tensor("dst", (4,), "float32")
    ready = kernel.event("ready", (2,), 1)
    done = kernel.event("done", (2,), 1)
    kernel.tile("producer", _MarkerTile(), (2, 1, 1), reads=[src], writes=[dst]).notify(
        ready, lambda m, n, k: (m,)
    )
    kernel.tile("consumer", _WarpScopedTile(), (2, 1, 1), reads=[dst]).wait(
        ready, lambda m, n, k: (m,)
    ).notify(done, lambda m, n, k: (m,))
    kernel.tile("sink", _MarkerTile(), (2, 1, 1), reads=[dst]).wait(done, lambda m, n, k: (m,))
    return kernel


def _static_options(attrs=None):
    return LoweringOptions(scheduler="static", attrs=attrs or {})


# ---------------------------------------------------------------------------
# Emission structure
# ---------------------------------------------------------------------------


def test_runtime_build_params_and_event_workspace():
    kernel = _chain_kernel()
    build = build_runtime_kernel(kernel, _static_options())
    func = build.module[kernel.name]

    assert [param.name for param in func.params] == [
        "src_handle",
        "dst_handle",
        "event_workspace_handle",
        "exec_queue_handle",
    ]
    buffers = [func.buffer_map[param] for param in func.params]
    # Two 2-cell events plus the completion cell.
    assert build.event_workspace_size == 2 + 2 + 1
    assert tuple(buffers[2].shape) == (build.event_workspace_size,)
    hardware = HardwareConfig()
    assert tuple(buffers[3].shape) == (hardware.sm_count, StaticTileScheduler.MAX_TASKS)
    assert build.exec_queue.shape == (hardware.sm_count, StaticTileScheduler.MAX_TASKS)
    assert build.exec_queue.dtype == np.int32


def test_runtime_build_emits_runtime_scheduler_and_scoped_endpoints():
    kernel = _chain_kernel()
    build = build_runtime_kernel(kernel, _static_options())
    script = build.module[kernel.name].script()

    # Central-queue scheduler decode and the dispatch fallthrough trap.
    assert "unpack_from_32bit" in script
    assert "trap_when_assert_failed(T.bool(False))" in script
    # Class resource lifecycle hooks from the wrapper.
    assert "T.cuda.cta_sync()" in script
    assert "nano_sleep(9999)" in script
    # The warp-scoped wait participates through the declared mask, and the
    # warpgroup notify syncs through a named barrier.
    assert "T.shift_right(3," in script
    assert "T.ptx.bar.sync" in script
    # Within the consumer's dispatch branch: prefetch -> scoped wait -> run
    # -> scoped notify (anchored on the producer's run before the branch).
    wait_at = script.index("T.shift_right(3,")
    prefetch_at = script.rindex("T.cuda.warp_sync()", 0, wait_at)
    prev_run_at = script.rindex("T.cuda.thread_fence()", 0, wait_at)
    run_at = script.index("T.cuda.thread_fence()", wait_at)
    notify_at = script.index("T.ptx.bar.sync", wait_at)
    assert prev_run_at < prefetch_at < wait_at < run_at < notify_at
    # The profiler is off unless requested.
    assert "timer_start" not in script


def test_runtime_build_profiler_hooks_when_enabled():
    kernel = KernelSpec("profiled")
    src = kernel.tensor("src", (4,), "float32")
    kernel.tile("stage", _ProfiledTile(), (2, 1, 1), reads=[src])
    # A tile without profile_event builds unprofiled under the same options.
    kernel.tile("plain", _MarkerTile(), (2, 1, 1), reads=[src])

    build = build_runtime_kernel(kernel, _static_options(attrs={"profiler": True}))
    func = build.module[kernel.name]
    assert build.profiler_on
    assert func.params[-1].name == "profiler_buffer_handle"
    script = func.script()
    for marker in ("timer_init", "timer_start", "timer_end", "timer_finalize"):
        assert marker in script


def test_runtime_build_without_events_skips_event_machinery():
    kernel = KernelSpec("no_events")
    src = kernel.tensor("src", (4,), "float32")
    kernel.tile("stage", _MarkerTile(), (2, 1, 1), reads=[src])

    build = build_runtime_kernel(kernel, _static_options())
    func = build.module[kernel.name]
    assert [param.name for param in func.params] == ["src_handle", "exec_queue_handle"]
    assert build.event_workspace_size == 0
    assert all(
        task[3] not in (INIT_EVENT_JOB_ID, WAIT_EVENT_INIT_JOB_ID) for task in build.central_tasks
    )


def test_runtime_build_rejects_unsupported_wait_level():
    kernel = KernelSpec("bad_wait")
    kernel.tensor("src", (4,), "float32")
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("producer", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("consumer", _WarpgroupWaitTile(), (1, 1, 1)).wait(ready, (0,))

    with pytest.raises(ValueError, match="wait_level"):
        build_runtime_kernel(kernel, _static_options())


def test_runtime_build_keeps_the_runtime_scalar_guard():
    kernel = KernelSpec("scalar_grid")
    counter = kernel.tensor("counter", (1,), "int32")
    routed = kernel.scalar("routed_rows", source=(counter, (0,)), range=(1, 128))
    kernel.tile("stage", _MarkerTile(), (routed, 1, 1))

    with pytest.raises(ValueError, match="runtime scalar"):
        build_runtime_kernel(kernel, _static_options())


def test_runtime_build_callable_init_count():
    kernel = KernelSpec("callable_count")
    kernel.tensor("src", (4,), "float32")
    ready = kernel.event("ready", (2,), lambda coord: 1)
    kernel.tile("producer", _MarkerTile(), (2, 1, 1)).notify(ready, lambda m, n, k: (m,))
    kernel.tile("consumer", _MarkerTile(), (2, 1, 1)).wait(ready, lambda m, n, k: (m,))

    build = build_runtime_kernel(kernel, _static_options())
    assert build.event_workspace_size == 3


# ---------------------------------------------------------------------------
# Host queue derivation
# ---------------------------------------------------------------------------


def _queue_kernel(name="queue_spec"):
    kernel = KernelSpec(name)
    kernel.tensor("t", (4,), "float32")
    ready = kernel.event("ready", (3,), 2)
    kernel.tile("producer", _MarkerTile(), (2, 2, 1)).notify(ready, lambda m, n, k: (m,))
    kernel.tile("consumer", _MarkerTile(), (3, 1, 1)).wait(ready, lambda m, n, k: (m,))
    return kernel


def _queue_plan(kernel, hardware):
    options = _resolve_options(_static_options())
    return _prepare_runtime_plan(kernel, options, hardware)


def test_derive_static_central_tasks_order():
    hardware = HardwareConfig(sm_count=2)
    plan = _queue_plan(_queue_kernel(), hardware)
    central = derive_static_central_tasks(plan)

    assert central == [
        # One init task per etensor: one user event plus the completion cell.
        (0, 0, 0, INIT_EVENT_JOB_ID),
        (1, 0, 0, INIT_EVENT_JOB_ID),
        # Entry tile tasks (m-major then n).
        (0, 0, 0, 0),
        (0, 1, 0, 0),
        (1, 0, 0, 0),
        (1, 1, 0, 0),
        # One event-init wait task per SM.
        (0, 0, 0, WAIT_EVENT_INIT_JOB_ID),
        (1, 0, 0, WAIT_EVENT_INIT_JOB_ID),
        # Waiting tile tasks after the wait barrier.
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (2, 0, 0, 1),
    ]


def test_build_static_queues_round_robin_and_end_padding():
    hardware = HardwareConfig(sm_count=2)
    plan = _queue_plan(_queue_kernel(), hardware)
    central, queue = build_static_queues(plan, hardware)

    assert queue.shape == (2, StaticTileScheduler.MAX_TASKS)
    assert queue.dtype == np.int32
    # Round-robin deal: task i lands at row i // sm_count, SM i % sm_count.
    for index, (m_idx, n_idx, k_idx, job_id) in enumerate(central):
        row, sm = divmod(index, 2)
        assert unpack_from_32bit_host(queue[sm, row]) == (job_id, m_idx, n_idx, k_idx)
    # The depleted SM of the last partial row and the full trailing row are END.
    last_row = (len(central) - 1) // 2
    assert unpack_from_32bit_host(queue[1, last_row])[0] == DEFAULT_END_JOB_ID
    assert unpack_from_32bit_host(queue[0, last_row + 1])[0] == DEFAULT_END_JOB_ID
    assert unpack_from_32bit_host(queue[1, last_row + 1])[0] == DEFAULT_END_JOB_ID


def test_derive_static_central_tasks_topological_order():
    kernel = KernelSpec("dag")
    kernel.tensor("t", (4,), "float32")
    event_a = kernel.event("event_a", (1,), 1)
    event_b = kernel.event("event_b", (1,), 1)
    # Spec order interleaves entry and waiting tiles on purpose.
    kernel.tile("t_entry", _MarkerTile(), (1, 1, 1)).notify(event_a, (0,))
    kernel.tile("t_mid", _MarkerTile(), (1, 1, 1)).wait(event_a, (0,)).notify(event_b, (0,))
    kernel.tile("t_entry_2", _MarkerTile(), (1, 1, 1))
    kernel.tile("t_last", _MarkerTile(), (1, 1, 1)).wait(event_b, (0,))

    hardware = HardwareConfig(sm_count=1)
    plan = _queue_plan(kernel, hardware)
    central = derive_static_central_tasks(plan)

    jobs = [task[3] for task in central]
    init_count = 2 + 1  # two user events plus the completion cell
    assert jobs[:init_count] == [INIT_EVENT_JOB_ID] * init_count
    body = jobs[init_count:]
    # Entry tiles in spec order, then the wait barrier, then waiting tiles in
    # topological order (t_mid before t_last regardless of spec order).
    assert body == [0, 2, WAIT_EVENT_INIT_JOB_ID, 1, 3]


def test_derive_static_central_tasks_rejects_cycles():
    kernel = KernelSpec("cyclic")
    kernel.tensor("t", (4,), "float32")
    event_a = kernel.event("event_a", (1,), 1)
    event_b = kernel.event("event_b", (1,), 1)
    kernel.tile("t1", _MarkerTile(), (1, 1, 1)).wait(event_b, (0,)).notify(event_a, (0,))
    kernel.tile("t2", _MarkerTile(), (1, 1, 1)).wait(event_a, (0,)).notify(event_b, (0,))

    with pytest.raises(ValueError, match="acyclic"):
        build_runtime_kernel(kernel, _static_options())


def test_derive_static_central_tasks_symbolic_var_values():
    kernel = KernelSpec("symbolic")
    rows = kernel.var("rows", "int32", range=(1, 16))
    kernel.tensor("t", (rows,), "float32")
    kernel.tile("stage", _MarkerTile(), (rows, 1, 1))

    hardware = HardwareConfig(sm_count=2)
    plan = _queue_plan(kernel, hardware)
    with pytest.raises(ValueError, match="var_values"):
        derive_static_central_tasks(plan)
    with pytest.raises(ValueError, match="unknown var"):
        derive_static_central_tasks(plan, {"nope": 1})

    central = derive_static_central_tasks(plan, {"rows": 4})
    assert central == [(m_idx, 0, 0, 0) for m_idx in range(4)]

    # The richer entry point derives the same queue alongside the module.
    build = build_runtime_kernel(kernel, _static_options(), var_values={"rows": 4})
    assert build.central_tasks == tuple(central)
    assert [param.name for param in build.module[kernel.name].params] == [
        "rows",
        "t_handle",
        "exec_queue_handle",
    ]


def test_build_static_queues_capacity_error():
    kernel = KernelSpec("overflow")
    kernel.tensor("t", (256,), "float32")
    kernel.tile("stage", _MarkerTile(), (200, 1, 1))

    with pytest.raises(ValueError, match="exceeding"):
        build_runtime_kernel(kernel, LoweringOptions(scheduler="static", attrs={"sm_count": 1}))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def test_scheduler_none_keeps_legacy_behavior():
    kernel = _chain_kernel("legacy_chain")
    default_module = lower_to_tirx_module(kernel)
    explicit_module = lower_to_tirx_module(kernel, LoweringOptions(scheduler=None))
    assert {gv.name_hint for gv in default_module.functions} == {
        "legacy_chain",
        "legacy_chain_init_queue",
    }
    assert default_module.script() == explicit_module.script()


def test_scheduler_static_routes_to_runtime_builder():
    kernel = _chain_kernel("routed_chain")
    module = lower_to_tirx_module(kernel, _static_options())
    assert [gv.name_hint for gv in module.functions] == ["routed_chain"]
    assert "exec_queue" in module["routed_chain"].script()

    func = lower_to_tirx(kernel, _static_options())
    assert isinstance(func, tvm.tirx.PrimFunc)
    assert func.attrs["global_symbol"] == "routed_chain"


def test_scheduler_static_has_no_queue_init_kernel():
    kernel = _chain_kernel("no_queue_init")
    with pytest.raises(ValueError, match="exec queue"):
        lower_static_queue_init_to_tirx(kernel, _static_options())


def test_scheduler_dynamic_not_implemented():
    kernel = _chain_kernel("dynamic_chain")
    with pytest.raises(NotImplementedError, match="not yet"):
        lower_to_tirx_module(kernel, LoweringOptions(scheduler="dynamic"))


def test_scheduler_garbage_rejected():
    kernel = _chain_kernel("garbage_chain")
    with pytest.raises(ValueError, match="unsupported scheduler"):
        lower_to_tirx_module(kernel, LoweringOptions(scheduler="round-robin"))


def test_build_runtime_kernel_requires_static_scheduler():
    kernel = _chain_kernel("misrouted_chain")
    with pytest.raises(ValueError, match="scheduler='static'"):
        build_runtime_kernel(kernel, LoweringOptions())


# ---------------------------------------------------------------------------
# GPU numerical gate
# ---------------------------------------------------------------------------


def _require_cuda_sm100():
    try:
        import torch
    except ImportError:
        pytest.skip("torch is required for the megakernel GPU gate")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the megakernel GPU gate")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("the megakernel GPU gate requires SM100 or newer")
    if not tvm.cuda(0).exist:
        pytest.skip("TVM CUDA device 0 is not available")


@pytest.mark.parametrize("m", [256, 1024])
def test_demo_two_stage_reduce_gpu(m):
    _require_cuda_sm100()
    from tvm.megakernel.demo.runner import run_two_stage_reduce

    report = run_two_stage_reduce(m=m)
    assert report["max_abs_err"] < 1e-3
