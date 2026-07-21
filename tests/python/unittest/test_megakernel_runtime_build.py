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

import re
from enum import Enum
from typing import ClassVar

import numpy as np
import pytest

import tvm
from tvm.megakernel.dsl import KernelSpec, TileImpl
from tvm.megakernel.runtime import (
    DynamicTileScheduler,
    HardwareConfig,
    StaticTileScheduler,
    unpack_from_32bit_host,
)
from tvm.megakernel.transform import (
    LoweringOptions,
    build_runtime_kernel,
    lower_to_tirx,
    lower_to_tirx_module,
)
from tvm.megakernel.transform.prepare import (
    DEFAULT_END_JOB_ID,
    INIT_EVENT_JOB_ID,
    WAIT_EVENT_INIT_JOB_ID,
)
from tvm.megakernel.transform.runtime_build import (
    _hardware_from_options,
    _lower_runtime_expr,
    _prepare_dynamic_plan,
    _prepare_runtime_plan,
    _resolve_options,
    _RuntimeDynamicKernelBuilder,
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


def test_runtime_build_auto_generates_the_scalar_grid_guard():
    kernel = KernelSpec("scalar_grid")
    counter = kernel.tensor("counter", (1,), "int32")
    out = kernel.tensor("out", (128,), "int32")
    routed = kernel.scalar("routed_rows", source=(counter, (0,)), range=(1, 128))
    kernel.tile("stage", _MarkerTile(), (routed, 1, 1), writes=[out])

    # No declared predicate: the static builder gates on the runtime extent.
    build = build_runtime_kernel(kernel, _static_options())
    func = build.module[kernel.name]
    assert isinstance(func, tvm.tirx.PrimFunc)
    assert "counter" in func.script()  # the guard loads the scalar buffer


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


def test_scheduler_default_is_the_runtime_static_builder():
    kernel = _chain_kernel("default_chain")
    default_module = lower_to_tirx_module(kernel)
    explicit_module = lower_to_tirx_module(kernel, LoweringOptions(scheduler="static"))
    assert [gv.name_hint for gv in default_module.functions] == ["default_chain"]
    assert "exec_queue" in default_module["default_chain"].script()
    assert default_module.script() == explicit_module.script()


def test_scheduler_none_is_rejected():
    kernel = _chain_kernel("none_chain")
    with pytest.raises(ValueError, match="unsupported scheduler"):
        lower_to_tirx_module(kernel, LoweringOptions(scheduler=None))


def test_scheduler_static_routes_to_runtime_builder():
    kernel = _chain_kernel("routed_chain")
    module = lower_to_tirx_module(kernel, _static_options())
    assert [gv.name_hint for gv in module.functions] == ["routed_chain"]
    assert "exec_queue" in module["routed_chain"].script()

    func = lower_to_tirx(kernel, _static_options())
    assert isinstance(func, tvm.tirx.PrimFunc)
    assert func.attrs["global_symbol"] == "routed_chain"


def test_scheduler_dynamic_routes_to_runtime_builder():
    kernel = _chain_kernel("dynamic_chain")
    module = lower_to_tirx_module(kernel, LoweringOptions(scheduler="dynamic"))
    assert [gv.name_hint for gv in module.functions] == ["dynamic_chain"]
    assert "exec_task" in module["dynamic_chain"].script()

    func = lower_to_tirx(kernel, LoweringOptions(scheduler="dynamic"))
    assert isinstance(func, tvm.tirx.PrimFunc)
    assert func.attrs["global_symbol"] == "dynamic_chain"


def test_scheduler_garbage_rejected():
    kernel = _chain_kernel("garbage_chain")
    with pytest.raises(ValueError, match="unsupported scheduler"):
        lower_to_tirx_module(kernel, LoweringOptions(scheduler="round-robin"))


def test_build_runtime_kernel_rejects_none_scheduler():
    kernel = _chain_kernel("misrouted_chain")
    with pytest.raises(ValueError, match="scheduler="):
        build_runtime_kernel(kernel, LoweringOptions(scheduler=None))


class _MixedPolicyTile(_MarkerTile):
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        smem_manager.alloc((16,), "uint8", policy="shared")
        smem_manager.alloc((16,), "uint8", policy="exclusive")


class _OversizedSmemTile(_MarkerTile):
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        smem_manager.alloc((40000,), "uint8")


def test_runtime_build_rejects_mixed_smem_policies():
    kernel = KernelSpec("mixed_smem_policy")
    kernel.tile("managed", _MixedPolicyTile(), (1, 1, 1))
    with pytest.raises(tvm.error.DiagnosticError, match="Cannot use both"):
        build_runtime_kernel(kernel, _static_options())


def test_runtime_build_rejects_smem_overflow():
    kernel = KernelSpec("smem_overflow")
    kernel.tile("managed", _OversizedSmemTile(), (1, 1, 1))
    with pytest.raises(tvm.error.DiagnosticError, match="exceeds the chunked region"):
        build_runtime_kernel(kernel, _static_options(attrs={"max_dynamic_smem": 32768}))


# ---------------------------------------------------------------------------
# Dynamic scheduling: emission structure
# ---------------------------------------------------------------------------


class _DynScopedTile(_MarkerTile):
    wait_level: ClassVar[str] = "warp"
    wait_mask: ClassVar[int] = 0xF
    notify_scope: ClassVar[tuple[str, int]] = ("warpgroup", 0)


def _dynamic_options(attrs=None):
    return LoweringOptions(scheduler="dynamic", attrs=attrs or {})


def _dynamic_spec(name="dynamic_mark", scoped=True):
    kernel = KernelSpec(name)
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _MarkerTile(), (1, 1, 1), reads=[count_buf]).notify(ready, (0,))
    kernel.tile(
        "mark", _DynScopedTile() if scoped else _MarkerTile(), (n_tiles, 1, 1), writes=[out]
    ).wait(ready, (0,))
    return kernel


def test_dynamic_build_params_and_drain_layout():
    kernel = _dynamic_spec()
    build = build_runtime_kernel(kernel, _dynamic_options())
    func = build.module[kernel.name]

    assert [param.name for param in func.params] == [
        "count_handle",
        "out_handle",
        "event_workspace_handle",
        "exec_task",
        "exec_head",
        "exec_tail",
    ]
    buffers = [func.buffer_map[param] for param in func.params]
    # One user event cell plus the synthesized drain cell; no completion cell.
    assert build.event_workspace_size == 2
    assert tuple(buffers[2].shape) == (2,)
    assert tuple(buffers[3].shape) == (DynamicTileScheduler.MAX_TASKS,)
    assert tuple(buffers[4].shape) == tuple(buffers[5].shape) == (1,)
    (drain,) = build.drain_events
    assert drain.name == "__drain_mark__"
    assert drain.workspace_offset == 1
    assert drain.static_count is None
    assert drain.runtime_initialized


def test_dynamic_build_emits_mpmc_scheduler_and_push_structure():
    kernel = _dynamic_spec()
    build = build_runtime_kernel(kernel, _dynamic_options())
    script = build.module[kernel.name].script()

    # MPMC fetch/decode and the dispatch fallthrough trap.
    assert "while_ld_global_acquire" in script
    assert "unpack_from_32bit" in script
    assert "trap_when_assert_failed(T.bool(False))" in script
    # Pre-notify trigger (old % base == 1) and the queue push.
    assert "% 65536 == 1" in script
    assert "stg_local" in script
    # The pusher's pre-notify uses its notify scope (cta), the terminal's
    # drain pre-notify uses the warpgroup scope metadata.
    assert "T.ptx.bar.sync" in script
    # The terminal END push targets the reserved END job id with one task per SM.
    assert "task_type: T.int32 = 31" in script
    assert "enqueue_num: T.int32 = 148" in script
    # The drain event is runtime-initialized with (base + 1) * count.
    assert "65537 *" in script
    # The mark tile waits at warp level through its declared mask.
    assert "T.shift_right(15," in script
    # Profiler stays off by default.
    assert "timer_start" not in script


def test_dynamic_build_profiler_hooks_when_enabled():
    kernel = KernelSpec("dyn_profiled")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _ProfiledTile(), (1, 1, 1), reads=[count_buf]).notify(ready, (0,))
    kernel.tile("mark", _ProfiledTile(), (n_tiles, 1, 1), writes=[out]).wait(ready, (0,))

    build = build_runtime_kernel(kernel, _dynamic_options(attrs={"profiler": True}))
    func = build.module[kernel.name]
    assert build.profiler_on
    assert func.params[-1].name == "profiler_buffer_handle"
    script = func.script()
    for marker in ("timer_init", "timer_start", "timer_end", "timer_finalize"):
        assert marker in script


def test_dynamic_build_seed_queue_contents():
    kernel = _dynamic_spec()
    build = build_runtime_kernel(kernel, _dynamic_options())

    # Two etensors (user event + drain) plus the single entry task.
    assert build.central_tasks == (
        (0, 0, 0, INIT_EVENT_JOB_ID),
        (1, 0, 0, INIT_EVENT_JOB_ID),
        (0, 0, 0, 0),
    )
    assert build.queue_tasks.shape == (DynamicTileScheduler.MAX_TASKS,)
    for index, (m_idx, n_idx, k_idx, job_id) in enumerate(build.central_tasks):
        assert unpack_from_32bit_host(build.queue_tasks[index]) == (job_id, m_idx, n_idx, k_idx)
    assert build.queue_head.tolist() == [0]
    assert build.queue_tail.tolist() == [len(build.central_tasks)]
    # Untouched slots stay at the -1 empty marker for the device spin.
    assert build.queue_tasks[len(build.central_tasks)] == -1


# ---------------------------------------------------------------------------
# Dynamic scheduling: synthesis decisions
# ---------------------------------------------------------------------------


def _dynamic_plan(kernel, attrs=None):
    options = _resolve_options(_dynamic_options(attrs))
    return _prepare_dynamic_plan(kernel, options, _hardware_from_options(options))


def test_dynamic_synthesis_derives_pusher_count_and_indices():
    plan = _dynamic_plan(_dynamic_spec())
    (rule,) = plan.dispatch_rules.values()
    assert (rule.source.name, rule.target.name) == ("plant", "mark")
    assert rule.event.name == "ready"
    # Constant coord on both sides: every mark axis is free.
    assert rule.free_axes == (0, 1, 2)
    assert rule.pinned == ()
    assert rule.free_extents == plan.scheduled[id(rule.target)]
    assert rule.pre_scope == ("cta", 0)
    assert rule.push_level == "cta"


def test_dynamic_synthesis_pinned_axis_push():
    kernel = KernelSpec("pinned")
    kernel.tensor("t", (8,), "float32")
    ready = kernel.event("ready", (4,), 1)
    kernel.tile("producer", _MarkerTile(), (4, 2, 1)).notify(ready, lambda m, n, k: (m,))
    kernel.tile("consumer", _MarkerTile(), (4, 3, 1)).wait(ready, lambda m, n, k: (m,))

    plan = _dynamic_plan(kernel)
    (rule,) = plan.dispatch_rules.values()
    # The m axis is pinned to the producer's m; n and k stay free.
    assert rule.pinned == ((0, ("axis", 0)),)
    assert rule.free_axes == (1, 2)
    assert rule.free_extents == (3, 1)


def test_dynamic_synthesis_rejects_multi_producer_event():
    kernel = KernelSpec("ambiguous")
    kernel.tensor("t", (4,), "float32")
    ready = kernel.event("ready", (1,), 2)
    kernel.tile("p1", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("p2", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("consumer", _MarkerTile(), (1, 1, 1)).wait(ready, (0,))

    with pytest.raises(ValueError, match="producers"):
        _dynamic_plan(kernel)


def test_dynamic_synthesis_rejects_multi_wait_tile():
    kernel = KernelSpec("multi_wait")
    kernel.tensor("t", (4,), "float32")
    event_a = kernel.event("event_a", (1,), 1)
    event_b = kernel.event("event_b", (1,), 1)
    kernel.tile("producer", _MarkerTile(), (1, 1, 1)).notify(event_a, (0,)).notify(event_b, (0,))
    kernel.tile("consumer", _MarkerTile(), (1, 1, 1)).wait(event_a, (0,)).wait(event_b, (0,))

    with pytest.raises(ValueError, match="waits"):
        _dynamic_plan(kernel)


def test_dynamic_synthesis_rejects_fan_out():
    kernel = KernelSpec("fan_out")
    kernel.tensor("t", (4,), "float32")
    event_a = kernel.event("event_a", (1,), 1)
    event_b = kernel.event("event_b", (1,), 1)
    event_c = kernel.event("event_c", (1,), 2)
    kernel.tile("producer", _MarkerTile(), (1, 1, 1)).notify(event_a, (0,)).notify(event_b, (0,))
    kernel.tile("c1", _MarkerTile(), (1, 1, 1)).wait(event_a, (0,)).notify(event_c, (0,))
    kernel.tile("c2", _MarkerTile(), (1, 1, 1)).wait(event_b, (0,)).notify(event_c, (0,))
    kernel.tile("sink", _MarkerTile(), (1, 1, 1)).wait(event_c, (0,))

    with pytest.raises(ValueError, match="more than one tile"):
        _dynamic_plan(kernel)


def test_dynamic_synthesis_rejects_multiple_terminal_tiles():
    kernel = KernelSpec("two_terminals")
    kernel.tensor("t", (4,), "float32")
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("producer", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("c1", _MarkerTile(), (1, 1, 1)).wait(ready, (0,))
    kernel.tile("c2", _MarkerTile(), (1, 1, 1)).wait(ready, (0,))

    with pytest.raises(ValueError, match="exactly one terminal"):
        _dynamic_plan(kernel)


def test_dynamic_synthesis_rejects_cycles():
    kernel = KernelSpec("cyclic_dynamic")
    kernel.tensor("t", (4,), "float32")
    event_a = kernel.event("event_a", (1,), 1)
    event_b = kernel.event("event_b", (1,), 1)
    kernel.tile("t1", _MarkerTile(), (1, 1, 1)).wait(event_b, (0,)).notify(event_a, (0,))
    kernel.tile("t2", _MarkerTile(), (1, 1, 1)).wait(event_a, (0,)).notify(event_b, (0,))

    with pytest.raises(ValueError, match="acyclic"):
        _dynamic_plan(kernel)


def test_dynamic_synthesis_rejects_scalar_entry_tile():
    kernel = KernelSpec("scalar_entry")
    count_buf = kernel.tensor("count", (1,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    kernel.tile("entry", _MarkerTile(), (n_tiles, 1, 1))

    with pytest.raises(ValueError, match="entry tile"):
        _dynamic_plan(kernel)


def test_dynamic_synthesis_rejects_oversized_queue_and_packing():
    kernel = KernelSpec("too_big")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 33))
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("mark", _MarkerTile(), (n_tiles, 1000, 1), writes=[out]).wait(ready, (0,))

    with pytest.raises(ValueError, match="capacity"):
        _dynamic_plan(kernel)

    kernel = KernelSpec("too_wide")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 9000))
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("mark", _MarkerTile(), (n_tiles, 1, 1), writes=[out]).wait(ready, (0,))

    with pytest.raises(ValueError, match="packed-task limit"):
        _dynamic_plan(kernel)


def test_dynamic_coalescing_math_and_validation():
    kernel = KernelSpec("coalesced")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("mark", _MarkerTile(), (n_tiles, 16, 1), writes=[out]).wait(ready, (0,))

    plan = _dynamic_plan(kernel, {"tile_coalescing": {"mark": 4}})
    mark = next(tile for tile in plan.kernel.tiles if tile.name == "mark")
    # The scheduled grid carries the divided n extent.
    assert plan.scheduled[id(mark)][1] == 4
    assert plan.drain.count_extents[1] == 4
    (rule,) = plan.dispatch_rules.values()
    assert rule.free_extents[1] == 4

    with pytest.raises(ValueError, match="divisible"):
        _dynamic_plan(kernel, {"tile_coalescing": {"mark": 3}})
    with pytest.raises(ValueError, match="host-seeded"):
        _dynamic_plan(kernel, {"tile_coalescing": {"plant": 2}})
    with pytest.raises(ValueError, match="unknown tile"):
        _dynamic_plan(kernel, {"tile_coalescing": {"ghost": 2}})


def test_dynamic_coalesced_run_repeats_with_expanded_n():
    kernel = KernelSpec("coalesced_emit")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("mark", _MarkerTile(), (n_tiles, 16, 1), writes=[out]).wait(ready, (0,))

    build = build_runtime_kernel(kernel, _dynamic_options(attrs={"tile_coalescing": {"mark": 4}}))
    script = build.module[kernel.name].script()
    # One serial run loop per task (the production per-task loop form) with
    # the run hook inside; the push count carries the divided extent.
    assert script.count("T.cuda.thread_fence()") == 2
    assert "in range(4):" in script
    assert "count * 4" in script


def test_dynamic_escape_hatch_declares_push_rule():
    kernel = KernelSpec("hatched")
    kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile(
        "mark",
        _MarkerTile(),
        (8, 1, 1),
        writes=[out],
        attrs={
            "megakernel.dispatch": {
                "source": "plant",
                "count": 8,
                "indices": lambda push_idx, m, n, k: (push_idx, 0, 0),
                "pre_scope": ("warp", 0),
                "push_level": "warp",
            }
        },
    ).wait(ready, (0,))

    plan = _dynamic_plan(kernel)
    (rule,) = plan.dispatch_rules.values()
    assert rule.custom_count is not None
    assert rule.custom_indices is not None
    assert rule.pre_scope == ("warp", 0)
    assert rule.push_level == "warp"
    assert rule.count_upper == 8
    build = build_runtime_kernel(kernel, _dynamic_options())
    assert build.event_workspace_size == 2


@pytest.mark.parametrize(
    "spec,error",
    [
        ({"ghost": 1, "source": "plant", "count": 1, "indices": (0, 0, 0)}, "unknown keys"),
        ({"count": 1, "indices": (0, 0, 0)}, "requires key 'source'"),
        ({"source": "ghost", "count": 1, "indices": (0, 0, 0)}, "unknown source"),
        (
            {"source": "plant", "count": 1, "indices": (0, 0, 0), "event": "ghost"},
            "unknown event",
        ),
        (
            {"source": "plant", "count": 1, "indices": (0, 0, 0), "pre_scope": ("warp",)},
            "pre_scope",
        ),
        (
            {
                "source": "plant",
                "count": 1,
                "indices": (0, 0, 0),
                "pre_scope": ("warp", 0),
                "push_level": "cta",
            },
            "cannot push at",
        ),
    ],
)
def test_dynamic_escape_hatch_validation(spec, error):
    kernel = KernelSpec("bad_hatch")
    kernel.tensor("t", (4,), "float32")
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("mark", _MarkerTile(), (1, 1, 1), attrs={"megakernel.dispatch": spec}).wait(
        ready, (0,)
    )

    with pytest.raises((TypeError, ValueError), match=error):
        _dynamic_plan(kernel)


def test_dynamic_drain_writer_selection():
    # Host-planted scalar: the terminal tile's pusher initializes the drain.
    plan = _dynamic_plan(_dynamic_spec())
    assert plan.drain.writer.name == "plant"

    # Produced scalar: the producing tile initializes the drain.
    kernel = KernelSpec("produced_scalar")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ready = kernel.event("ready", (1,), 1)
    ready2 = kernel.event("ready2", (1,), 1)
    kernel.tile("plant", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("mid", _MarkerTile(), (1, 1, 1), writes=[count_buf]).wait(ready, (0,)).notify(
        ready2, (0,)
    )
    kernel.tile("mark", _MarkerTile(), (n_tiles, 1, 1), writes=[out]).wait(ready2, (0,))

    plan = _dynamic_plan(kernel)
    assert plan.drain.writer.name == "mid"


def test_dynamic_var_only_drain_uses_seed_init():
    kernel = KernelSpec("var_drain")
    rows = kernel.var("rows", "int32", range=(1, 16))
    kernel.tensor("t", (rows,), "float32")
    ready = kernel.event("ready", (1,), 2)
    kernel.tile("plant", _MarkerTile(), (2, 1, 1)).notify(ready, (0,))
    kernel.tile("mark", _MarkerTile(), (rows, 1, 1)).wait(ready, (0,))

    build = build_runtime_kernel(kernel, _dynamic_options())
    (drain,) = build.drain_events
    assert drain.static_count is None
    assert not drain.runtime_initialized
    # Two etensors (user event + drain) plus the two plant tasks.
    assert len(build.central_tasks) == 2 + 2
    script = build.module[kernel.name].script()
    assert "65537" in script


def test_dynamic_static_terminal_drain():
    kernel = _chain_kernel("static_terminal")
    build = build_runtime_kernel(kernel, _dynamic_options())
    (drain,) = build.drain_events
    assert drain.static_count == 2
    assert not drain.runtime_initialized
    # Three etensors (ready, done, drain) plus two producer tasks.
    assert len(build.central_tasks) == 3 + 2


def test_dynamic_scalar_expr_lowering():
    kernel = _dynamic_spec()
    options = _resolve_options(_dynamic_options())
    plan = _prepare_dynamic_plan(kernel, options, _hardware_from_options(options))
    builder = _RuntimeDynamicKernelBuilder(plan, _hardware_from_options(options))
    n_tiles = next(iter(kernel.scalars.values()))
    loads = []

    def scalar_load(scalar):
        loads.append(scalar.name)
        return 7

    value = _lower_runtime_expr(n_tiles * 12 + 1, {}, scalar_load, "test")
    assert value == 7 * 12 + 1
    assert loads == ["n_tiles"]
    with pytest.raises(ValueError, match="unbound symbolic variable"):
        _lower_runtime_expr(kernel.var("ghost", range=(1, 4)) + 1, {}, scalar_load, "test")
    del builder


# ---------------------------------------------------------------------------
# Static scalar grids and run predicates
# ---------------------------------------------------------------------------


def _scalar_static_spec(name="scalar_static", predicate="default"):
    """Scalar-grid static spec; ``predicate`` is "default", None, or an attr value."""

    kernel = KernelSpec(name)
    counter = kernel.tensor("counter", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    routed = kernel.scalar("routed_rows", source=(counter, (0,)), range=(1, 8))
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("entry", _MarkerTile(), (1, 1, 1), reads=[counter]).notify(ready, (0,))
    attrs = {}
    if predicate == "default":
        attrs["megakernel.run_predicate"] = (0, "lt", routed)
    elif predicate is not None:
        attrs["megakernel.run_predicate"] = predicate
    kernel.tile("stage", _MarkerTile(), (routed, 1, 1), writes=[out], attrs=attrs).wait(ready, (0,))
    return kernel


def _buffer_load_buffers(func, name):
    from tvm.tirx import BufferLoad
    from tvm.tirx.stmt_functor import post_order_visit

    loads = []

    def visit(node):
        if isinstance(node, BufferLoad) and node.buffer.name == name:
            loads.append(node)

    post_order_visit(func.body, visit)
    return loads


def test_static_scalar_grid_enumerates_upper_bound_with_run_predicate():
    build = build_runtime_kernel(_scalar_static_spec(), _static_options())
    # Two INIT tasks (event + completion), one entry task, one WAIT per SM,
    # then the stage grid at the scalar upper bound.
    hardware = HardwareConfig()
    stage_tasks = [task for task in build.central_tasks if task[3] == 1]
    assert stage_tasks == [(m_idx, 0, 0, 1) for m_idx in range(8)]
    assert len(build.central_tasks) == 2 + 1 + hardware.sm_count + 8
    # The run is guarded by the scalar load comparison; no push synthesis.
    func = build.module["scalar_static"]
    assert _buffer_load_buffers(func, "counter")
    script = func.script()
    assert script.count("T.cuda.thread_fence()") == 2


def test_static_scalar_grid_run_predicate_validation():
    with pytest.raises(ValueError, match="axis"):
        build_runtime_kernel(_scalar_static_spec(predicate=(3, "lt", 1)), _static_options())
    with pytest.raises(ValueError, match="lt"):
        build_runtime_kernel(_scalar_static_spec(predicate=(0, "gt", 1)), _static_options())
    with pytest.raises(TypeError, match="expr"):
        build_runtime_kernel(_scalar_static_spec(predicate=(0, "lt", "x")), _static_options())
    with pytest.raises(TypeError, match="triple"):
        build_runtime_kernel(_scalar_static_spec(predicate=(0, "lt")), _static_options())


def test_dynamic_builder_emits_vacuous_run_predicate():
    kernel = KernelSpec("predicate_dynamic")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile(
        "mark",
        _MarkerTile(),
        (n_tiles, 1, 1),
        writes=[out],
        attrs={"megakernel.run_predicate": (0, "lt", n_tiles)},
    ).wait(ready, (0,))

    build = build_runtime_kernel(kernel, _dynamic_options())
    script = build.module[kernel.name].script()
    # Dynamically dispatched tasks already match the runtime count, so the
    # guard is emitted in its vacuously-true production form.
    assert "T.Or(T.bool(True)" in script
    assert script.count("T.cuda.thread_fence()") == 2


# ---------------------------------------------------------------------------
# Job id pinning and reserved id overrides
# ---------------------------------------------------------------------------


def test_job_id_pinning_and_reserved_id_overrides():
    kernel = _chain_kernel("pinned_chain")
    for tile, job_id in zip(kernel.tiles, (18, 19, 20)):
        tile.attrs["megakernel.job_id"] = job_id
    build = build_runtime_kernel(
        kernel,
        _static_options(attrs={"init_event_job_id": 28, "wait_event_init_job_id": 29}),
    )
    assert {task[3] for task in build.central_tasks} == {18, 19, 20, 28, 29}
    assert build.init_event_job_id == 28
    assert build.wait_event_init_job_id == 29
    assert build.end_task_type == 31
    written_columns = (len(build.central_tasks) + build.sm_count - 1) // build.sm_count + 1
    packed_types = {
        unpack_from_32bit_host(int(cell))[0]
        for cell in build.exec_queue[:, :written_columns].reshape(-1)
    }
    assert packed_types == {18, 19, 20, 28, 29, 31}


def test_job_id_pinning_validation():
    kernel = _chain_kernel("duplicate_ids")
    kernel.tiles[0].attrs["megakernel.job_id"] = 7
    kernel.tiles[1].attrs["megakernel.job_id"] = 7
    with pytest.raises(ValueError, match="duplicate"):
        build_runtime_kernel(kernel, _static_options())

    kernel = _chain_kernel("reserved_collision")
    kernel.tiles[0].attrs["megakernel.job_id"] = 29
    with pytest.raises(ValueError, match="reserved"):
        build_runtime_kernel(kernel, _static_options())

    kernel = _chain_kernel("wide_id")
    kernel.tiles[0].attrs["megakernel.job_id"] = 32
    with pytest.raises(ValueError, match="five-bit"):
        build_runtime_kernel(kernel, _static_options())

    kernel = _chain_kernel("bad_id_type")
    kernel.tiles[0].attrs["megakernel.job_id"] = "18"
    with pytest.raises(ValueError, match="non-negative integer"):
        build_runtime_kernel(kernel, _static_options())

    kernel = _chain_kernel("clashing_reserved")
    with pytest.raises(ValueError, match="distinct"):
        build_runtime_kernel(kernel, _static_options(attrs={"init_event_job_id": 31}))


def test_dynamic_job_id_pinning_and_reserved_overrides():
    kernel = _dynamic_spec("pinned_dynamic")
    kernel.tiles[0].attrs["megakernel.job_id"] = 18
    kernel.tiles[1].attrs["megakernel.job_id"] = 19
    build = build_runtime_kernel(
        kernel, _dynamic_options(attrs={"init_event_job_id": 28, "end_job_id": 30})
    )
    assert build.central_tasks == ((0, 0, 0, 28), (1, 0, 0, 28), (0, 0, 0, 18))
    assert build.init_event_job_id == 28
    assert build.end_task_type == 30
    script = build.module[kernel.name].script()
    assert "task_type: T.int32 = 30" in script

    kernel = _dynamic_spec("dynamic_collision")
    kernel.tiles[0].attrs["megakernel.job_id"] = 28
    with pytest.raises(ValueError, match="reserved"):
        build_runtime_kernel(kernel, _dynamic_options(attrs={"init_event_job_id": 28}))


# ---------------------------------------------------------------------------
# Declared drain events
# ---------------------------------------------------------------------------


def _declared_drain_spec(name="declared_drain", drain_shape=(1,)):
    kernel = KernelSpec(name)
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ready = kernel.event("ready", (1,), 1)
    kernel.event("drain_done", drain_shape, 64, attrs={"megakernel.drain": True})
    kernel.tile("plant", _MarkerTile(), (1, 1, 1), reads=[count_buf]).notify(ready, (0,))
    kernel.tile(
        "mark",
        _MarkerTile(),
        (n_tiles, 1, 1),
        writes=[out],
        # Needed only by the static build (scalar grid guard); the dynamic
        # builder ignores it.
        attrs={"megakernel.run_predicate": (0, "lt", n_tiles)},
    ).wait(ready, (0,))
    return kernel


def test_dynamic_declared_drain_event_layout():
    build = build_runtime_kernel(_declared_drain_spec(), _dynamic_options())
    (drain,) = build.drain_events
    assert drain.name == "drain_done"
    assert drain.workspace_offset == 1
    assert drain.static_count is None
    assert drain.runtime_initialized
    # Two etensors (user event + declared drain) plus the single entry task.
    assert len(build.central_tasks) == 3
    assert build.event_workspace_size == 2
    script = build.module["declared_drain"].script()
    assert "65537 *" in script


def test_static_declared_drain_event_is_an_ordinary_event():
    build = build_runtime_kernel(_declared_drain_spec("declared_static"), _static_options())
    # ready + drain_done + the completion cell; one INIT task per etensor.
    assert build.event_workspace_size == 3
    init_tasks = [task for task in build.central_tasks if task[3] == INIT_EVENT_JOB_ID]
    assert init_tasks == [(idx, 0, 0, INIT_EVENT_JOB_ID) for idx in range(3)]


def test_declared_drain_event_validation():
    kernel = _declared_drain_spec("two_drains")
    kernel.events["ready"].attrs["megakernel.drain"] = True
    with pytest.raises(ValueError, match="at most one"):
        build_runtime_kernel(kernel, _dynamic_options())

    kernel = _declared_drain_spec("wide_drain", drain_shape=(2,))
    with pytest.raises(ValueError, match="shape"):
        build_runtime_kernel(kernel, _dynamic_options())

    kernel = _declared_drain_spec("edged_drain")
    drain = kernel.events["drain_done"]
    kernel.tiles[0].notify(drain, (0,))
    kernel.tiles[1].wait(drain, (0,))
    with pytest.raises(ValueError, match="must not be"):
        build_runtime_kernel(kernel, _dynamic_options())


# ---------------------------------------------------------------------------
# Pre-notify scope overrides and class groups
# ---------------------------------------------------------------------------


class _PreScopedTile(_MarkerTile):
    notify_scope: ClassVar[tuple[str, int]] = ("cta", 0)
    pre_notify_scope: ClassVar[tuple[str, int] | None] = ("warp", 0)


class _GroupedTile(_MarkerTile):
    class_group: ClassVar[type | None] = _MarkerTile


def test_dynamic_pre_notify_scope_override():
    kernel = KernelSpec("pre_scoped")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _PreScopedTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("mark", _MarkerTile(), (n_tiles, 1, 1), writes=[out]).wait(ready, (0,))

    plan = _dynamic_plan(kernel)
    (rule,) = plan.dispatch_rules.values()
    # The pre-notify runs at the override scope, not the notify scope.
    assert rule.pre_scope == ("warp", 0)
    assert rule.push_level == "warp"


def test_pre_notify_scope_metadata_validation():
    class _BadScopeTile(_MarkerTile):
        pre_notify_scope: ClassVar[tuple[str, int] | None] = ("socket", 0)

    kernel = _dynamic_spec("bad_pre_scope")
    kernel.tiles[0].impl = _BadScopeTile()
    with pytest.raises(ValueError, match="pre_notify_scope scope"):
        kernel.validate()

    class _BadScopeIdTile(_MarkerTile):
        pre_notify_scope: ClassVar[tuple[str, int] | None] = ("warp", -1)

    kernel = _dynamic_spec("bad_pre_scope_id")
    kernel.tiles[0].impl = _BadScopeIdTile()
    with pytest.raises(ValueError, match="pre_notify_scope scope_id"):
        kernel.validate()


def test_wrapper_add_tile_honors_class_group():
    from tvm.megakernel.runtime import MegaKernelWrapper

    wrapper = MegaKernelWrapper()
    wrapper._add_tile(_GroupedTile(), None)
    assert wrapper.class_list == {_MarkerTile}
    wrapper._add_tile(_DynScopedTile(), None)
    assert wrapper.class_list == {_MarkerTile, _DynScopedTile}


# ---------------------------------------------------------------------------
# Profiler parameter ABI
# ---------------------------------------------------------------------------


def test_emit_profiler_param_keeps_signature_without_profiler():
    build = build_runtime_kernel(
        _chain_kernel("abi_chain"), _static_options(attrs={"emit_profiler_param": True})
    )
    assert not build.profiler_on
    func = build.module["abi_chain"]
    assert func.params[-1].name == "profiler_buffer_handle"
    script = func.script()
    assert "timer_init" not in script


# ---------------------------------------------------------------------------
# Review findings: F1 scalar-driven push trigger tightening
# ---------------------------------------------------------------------------


def _case_a_spec(name="case_a", pusher_grid=1):
    """entry -> writer (produces scalar) -> pusher -> mark (scalar grid)."""

    kernel = KernelSpec(name)
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    event_a = kernel.event("event_a", (1,), 1)
    event_b = kernel.event("event_b", (1,), 1)
    event_c = kernel.event("event_c", (1,), pusher_grid)
    kernel.tile("entry", _MarkerTile(), (1, 1, 1)).notify(event_a, (0,))
    kernel.tile("writer", _MarkerTile(), (1, 1, 1), writes=[count_buf]).wait(event_a, (0,)).notify(
        event_b, (0,)
    )
    kernel.tile("pusher", _MarkerTile(), (pusher_grid, 1, 1)).wait(event_b, (0,)).notify(
        event_c, (0,)
    )
    kernel.tile("mark", _MarkerTile(), (n_tiles, 1, 1), writes=[out]).wait(event_c, (0,))
    return kernel


def test_f1_scalar_free_push_keeps_started_trigger():
    plan = _dynamic_plan(_dynamic_spec(scoped=False))
    (rule,) = plan.dispatch_rules.values()
    assert rule.trigger == "started"
    assert not rule.post_run
    build = build_runtime_kernel(_dynamic_spec(scoped=False), _dynamic_options())
    script = build.module["dynamic_mark"].script()
    assert "% 65536 == 1" in script
    assert "% 65536 == 0" not in script


def test_f1_upstream_scalar_push_moves_post_run():
    plan = _dynamic_plan(_case_a_spec())
    rule = next(rule for rule in plan.dispatch_rules.values() if rule.target.name == "mark")
    assert rule.source.name == "pusher"
    assert rule.post_run
    assert rule.trigger == "started"
    build = build_runtime_kernel(_case_a_spec(), _dynamic_options())
    script = build.module["case_a"].script()
    # The pusher's push (trigger check) lands after its own run marker, and
    # the trigger keeps the production full-count form.
    assert "% 65536 == 0" not in script
    fences = [match.start() for match in re.finditer(re.escape("T.cuda.thread_fence()"), script)]
    triggers = [match.start() for match in re.finditer(re.escape("% 65536 == 1"), script)]
    assert len(triggers) == 4  # entry, writer, pusher (post-run), drain
    assert triggers[0] < fences[0]  # entry's push stays at task start
    assert fences[2] < triggers[2]  # the scalar-bearing push moved post-run


def test_f1_oversized_pusher_grid_rejected():
    with pytest.raises(ValueError, match="fit the persistent workers"):
        _dynamic_plan(_case_a_spec("case_a_big", pusher_grid=200))


def test_f1_self_produced_scalar_moves_push_post_run():
    kernel = KernelSpec("case_b_post_run")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("writer", _MarkerTile(), (1, 1, 1), writes=[count_buf]).notify(ready, (0,))
    kernel.tile("mark", _MarkerTile(), (n_tiles, 1, 1), writes=[out]).wait(ready, (0,))

    plan = _dynamic_plan(kernel)
    (rule,) = plan.dispatch_rules.values()
    assert rule.post_run
    assert rule.trigger == "started"

    build = build_runtime_kernel(kernel, _dynamic_options())
    script = build.module["case_b_post_run"].script()
    # The push (trigger check) lands after the pusher's run marker.
    assert script.index("T.cuda.thread_fence()") < script.index("% 65536 == 1")


def test_f1_racy_review_example_rejected_or_provably_safe():
    # The review's racy shape: a runtime-count producer whose instances all
    # notify one coordinate with init_count 1 — rejected (F3 cardinality).
    kernel = KernelSpec("racy")
    count_buf = kernel.tensor("count", (1,), "int32")
    count2 = kernel.tensor("count2", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_prod = kernel.scalar("n_prod", source=(count2, (0,)), range=(1, 4))
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ev0 = kernel.event("ev0", (1,), 1)
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("entry", _MarkerTile(), (1, 1, 1)).notify(ev0, (0,))
    kernel.tile("producer", _MarkerTile(), (n_prod, 1, 1), writes=[count_buf]).wait(
        ev0, (0,)
    ).notify(ready, (0,))
    kernel.tile("mark", _MarkerTile(), (n_tiles, 1, 1), writes=[out]).wait(ready, (0,))
    with pytest.raises(ValueError, match="statically provable"):
        _dynamic_plan(kernel)

    # The provably-safe variant (injective coords + upstream producer with a
    # fitting pusher grid) builds with the post-run push placement.
    build = build_runtime_kernel(_case_a_spec("racy_safe"), _dynamic_options())
    assert "% 65536 == 0" not in build.module["racy_safe"].script()


# ---------------------------------------------------------------------------
# Review findings: F3 scalar-dynamic event cardinality
# ---------------------------------------------------------------------------


def test_f3_rejects_unprovable_per_coord_cardinality():
    kernel = KernelSpec("bad_cardinality")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_prod = kernel.scalar("n_prod", source=(count_buf, (0,)), range=(1, 4))
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ev0 = kernel.event("ev0", (1,), 1)
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("entry", _MarkerTile(), (1, 1, 1)).notify(ev0, (0,))
    kernel.tile("producer", _MarkerTile(), (n_prod, 1, 1), writes=[count_buf]).wait(
        ev0, (0,)
    ).notify(ready, (0,))
    kernel.tile("mark", _MarkerTile(), (n_tiles, 1, 1), writes=[out]).wait(ready, (0,))

    with pytest.raises(ValueError, match="statically provable"):
        _dynamic_plan(kernel)


def test_f3_rejects_non_integer_init_count_on_touched_event():
    kernel = KernelSpec("callable_count_dynamic")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ready = kernel.event("ready", (2,), lambda coord: 1)
    kernel.tile("plant", _MarkerTile(), (2, 1, 1)).notify(ready, lambda m, n, k: (m,))
    kernel.tile("mark", _MarkerTile(), (n_tiles, 1, 1), writes=[out]).wait(
        ready, lambda m, n, k: (0,)
    )

    with pytest.raises(ValueError, match="plain integer"):
        _dynamic_plan(kernel)


def test_f3_moe_style_injective_maps_pass():
    # gate_up-style: scalar grid (routed, 12, 1), coord (m,), 12 per cell.
    kernel = KernelSpec("moe_style")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ready = kernel.event("ready", (1,), 1)
    gate_done = kernel.event("gate_done", (64,), 12)
    kernel.tile("plant", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("gate_up", _MarkerTile(), (n_tiles, 12, 1)).wait(ready, (0,)).notify(
        gate_done, lambda m, n, k: (m,)
    )
    kernel.tile("down", _MarkerTile(), (n_tiles, 1, 1), writes=[out]).wait(
        gate_done, lambda m, n, k: (m,)
    )

    plan = _dynamic_plan(kernel)
    assert len(plan.dispatch_rules) == 2


# ---------------------------------------------------------------------------
# Review findings: F4 static scalar-grid run predicates
# ---------------------------------------------------------------------------


def test_f4_declared_predicate_must_match_scalar_axis_and_extent():
    with pytest.raises(ValueError, match="not a scalar-dependent tile_num axis"):
        build_runtime_kernel(_scalar_static_spec(predicate=(1, "lt", 1)), _static_options())
    with pytest.raises(ValueError, match="does not match the axis's extent expression"):
        build_runtime_kernel(_scalar_static_spec(predicate=(0, "lt", 1)), _static_options())


def test_f4_correct_declaration_accepted():
    build = build_runtime_kernel(_scalar_static_spec(), _static_options())
    assert isinstance(build.module["scalar_static"], tvm.tirx.PrimFunc)


# ---------------------------------------------------------------------------
# Review findings: F5 escape hatch capacity and drain proofs
# ---------------------------------------------------------------------------


def _hatched_spec(name, *, count, tile_num=(8, 1, 1), source_grid=(1, 1, 1), scalar_range=(1, 64)):
    kernel = KernelSpec(name)
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=scalar_range)
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _MarkerTile(), source_grid).notify(ready, (0,))
    kernel.tile(
        "mark",
        _MarkerTile(),
        tile_num,
        writes=[out],
        attrs={
            "megakernel.dispatch": {
                "source": "plant",
                "count": count(n_tiles) if callable(count) else count,
                "indices": (0, 0, 0),
            }
        },
    ).wait(ready, (0,))
    return kernel


def test_f5_rejects_unbounded_hatch_count():
    kernel = _hatched_spec(
        "unbounded_hatch",
        count=lambda n_tiles: n_tiles,
        scalar_range=None,
    )
    # A scalar without a range cannot prove a static count upper bound.
    with pytest.raises(ValueError, match="upper bound"):
        _dynamic_plan(kernel)


def test_f5_capacity_uses_declared_count_bound():
    # Source grid 200 x declared count 1000 -> 200000 enqueues, far beyond
    # the target's tile_num volume; the capacity proof must use the declared
    # bound and reject.
    kernel = _hatched_spec(
        "capacity_hatch",
        count=1000,
        tile_num=(8, 1, 1),
        source_grid=(200, 1, 1),
    )
    with pytest.raises(ValueError, match="capacity"):
        _dynamic_plan(kernel)


def test_f5_rejects_drain_mismatch_for_scalar_terminal():
    kernel = KernelSpec("drain_mismatch")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile(
        "mark",
        _MarkerTile(),
        (n_tiles, 1, 1),
        writes=[out],
        attrs={
            "megakernel.dispatch": {
                "source": "plant",
                "count": n_tiles,
                "indices": (0, 0, 0),
            }
        },
    ).wait(ready, (0,))
    with pytest.raises(ValueError, match="underivable"):
        _dynamic_plan(kernel)


def test_f5_scalar_count_hatch_with_static_terminal_builds():
    kernel = _hatched_spec("good_hatch", count=lambda n_tiles: n_tiles)
    plan = _dynamic_plan(kernel)
    (rule,) = plan.dispatch_rules.values()
    assert rule.count_upper == 64
    build = build_runtime_kernel(kernel, _dynamic_options())
    (drain,) = build.drain_events
    assert drain.runtime_initialized


# ---------------------------------------------------------------------------
# Round-2 review findings
# ---------------------------------------------------------------------------


class _MaskedWarpTile(_MarkerTile):
    """The reviewer's narrow-wait pusher: warp wait, mask 0x1, thread-32 push."""

    wait_level: ClassVar[str] = "warp"
    wait_mask: ClassVar[int] = 0x1
    notify_scope: ClassVar[tuple[str, int]] = ("thread", 32)
    pre_notify_scope: ClassVar[tuple[str, int]] = ("thread", 32)


class _CoveredWarpTile(_MarkerTile):
    """Warp wait whose mask covers the pushing warp (mask 0x3, warp 0 push)."""

    wait_level: ClassVar[str] = "warp"
    wait_mask: ClassVar[int] = 0x3
    notify_scope: ClassVar[tuple[str, int]] = ("warp", 0)
    pre_notify_scope: ClassVar[tuple[str, int]] = ("warp", 0)


def _r21_spec(name, pusher_impl):
    kernel = KernelSpec(name)
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    event_a = kernel.event("event_a", (1,), 1)
    event_b = kernel.event("event_b", (1,), 1)
    event_c = kernel.event("event_c", (1,), 1)
    kernel.tile("entry", _MarkerTile(), (1, 1, 1)).notify(event_a, (0,))
    kernel.tile("writer", _MarkerTile(), (1, 1, 1), writes=[count_buf]).wait(event_a, (0,)).notify(
        event_b, (0,)
    )
    kernel.tile("pusher", pusher_impl, (1, 1, 1)).wait(event_b, (0,)).notify(event_c, (0,))
    kernel.tile("mark", _MarkerTile(), (n_tiles, 1, 1), writes=[out]).wait(event_c, (0,))
    return kernel


def _pusher_branch_order(script, fence_ord=3):
    """(fence, cta_sync, trigger) indices inside the pusher's dispatch branch."""

    fences = [match.start() for match in re.finditer(re.escape("T.cuda.thread_fence()"), script)]
    fence = fences[fence_ord - 1]
    sync = (
        script.index("T.cuda.cta_sync()", fence) if "T.cuda.cta_sync()" in script[fence:] else None
    )
    trigger = script.index("% 65536 == 1", fence)
    return fence, sync, trigger


def test_r21_masked_wait_gets_cta_barrier_before_push():
    plan = _dynamic_plan(_r21_spec("r21_masked", _MaskedWarpTile()))
    rule = next(rule for rule in plan.dispatch_rules.values() if rule.target.name == "mark")
    assert rule.post_run

    build = build_runtime_kernel(_r21_spec("r21_masked", _MaskedWarpTile()), _dynamic_options())
    script = build.module["r21_masked"].script()
    _, sync, trigger = _pusher_branch_order(script)
    # The uncovered pushing scope is joined by a CTA barrier before the push.
    assert sync is not None
    assert sync < trigger


def test_r21_cta_wait_needs_no_barrier():
    build = build_runtime_kernel(_r21_spec("r21_cta", _MarkerTile()), _dynamic_options())
    script = build.module["r21_cta"].script()
    _, sync, _ = _pusher_branch_order(script)
    assert sync is None


def test_r21_covering_warp_wait_needs_no_barrier():
    build = build_runtime_kernel(_r21_spec("r21_covered", _CoveredWarpTile()), _dynamic_options())
    script = build.module["r21_covered"].script()
    _, sync, _ = _pusher_branch_order(script)
    assert sync is None


def test_r21_wait_free_pusher_gets_barrier():
    kernel = KernelSpec("r21_no_wait")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("writer", _MarkerTile(), (1, 1, 1), writes=[count_buf]).notify(ready, (0,))
    kernel.tile("mark", _MarkerTile(), (n_tiles, 1, 1), writes=[out]).wait(ready, (0,))

    build = build_runtime_kernel(kernel, _dynamic_options())
    script = build.module["r21_no_wait"].script()
    fence = script.index("T.cuda.thread_fence()")
    sync = script.index("T.cuda.cta_sync()", fence)
    trigger = script.index("% 65536 == 1", fence)
    assert fence < sync < trigger


def test_r22_stateful_count_callable_rejected():
    kernel = KernelSpec("r22_stateful")
    kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    calls = {"n": 0}

    def stateful_count(m, n, k):
        calls["n"] += 1
        return 1 if calls["n"] == 1 else 40000

    kernel.tile(
        "mark",
        _MarkerTile(),
        (8, 1, 1),
        writes=[out],
        attrs={
            "megakernel.dispatch": {
                "source": "plant",
                "count": stateful_count,
                "indices": (0, 0, 0),
            }
        },
    ).wait(ready, (0,))

    with pytest.raises(ValueError, match="pure deterministic"):
        _dynamic_plan(kernel)


def test_r22_stateful_indices_callable_rejected():
    kernel = KernelSpec("r22_stateful_idx")
    kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    calls = {"n": 0}

    def stateful_indices(push_idx, m, n, k):
        calls["n"] += 1
        return (push_idx, 0, 0) if calls["n"] == 1 else (40000, 0, 0)

    kernel.tile(
        "mark",
        _MarkerTile(),
        (8, 1, 1),
        writes=[out],
        attrs={
            "megakernel.dispatch": {
                "source": "plant",
                "count": 8,
                "indices": stateful_indices,
            }
        },
    ).wait(ready, (0,))

    with pytest.raises(ValueError, match="pure deterministic"):
        _dynamic_plan(kernel)


def test_r22_legitimate_callable_evaluated_once():
    kernel = KernelSpec("r22_pure")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, 64))
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("plant", _MarkerTile(), (1, 1, 1)).notify(ready, (0,))
    calls = {"n": 0}

    def pure_count(m, n, k):
        calls["n"] += 1
        return n_tiles

    kernel.tile(
        "mark",
        _MarkerTile(),
        (8, 1, 1),
        writes=[out],
        attrs={
            "megakernel.dispatch": {
                "source": "plant",
                "count": pure_count,
                "indices": lambda push_idx, m, n, k: (push_idx, 0, 0),
            }
        },
    ).wait(ready, (0,))

    build = build_runtime_kernel(kernel, _dynamic_options())
    # Exactly the purity double-call at the same probe args; never re-called.
    assert calls["n"] == 2
    plan = _dynamic_plan(kernel)
    (rule,) = plan.dispatch_rules.values()
    assert rule.count_upper == 64
    script = build.module["r22_pure"].script()
    # Codegen lowered the captured tree (the scalar load), not a re-call.
    assert "count" in script


def test_r23_rejects_static_scalar_cardinality_mismatch():
    kernel = KernelSpec("r23_repro")
    counter = kernel.tensor("counter", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    routed = kernel.scalar("routed", source=(counter, (0,)), range=(1, 4))
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("producer", _MarkerTile(), (routed, 1, 1), writes=[out]).notify(ready, (0,))
    kernel.tile("consumer", _MarkerTile(), (1, 1, 1)).wait(ready, (0,))

    with pytest.raises(ValueError, match="provides 4 notifications per coordinate"):
        build_runtime_kernel(kernel, _static_options())


def test_r23_moe_style_enumerated_fiber_passes():
    kernel = KernelSpec("r23_moe_style")
    counter = kernel.tensor("counter", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    routed = kernel.scalar("routed", source=(counter, (0,)), range=(1, 4))
    ready = kernel.event("ready", (4,), 12)
    kernel.tile("producer", _MarkerTile(), (routed, 12, 1), writes=[out]).notify(
        ready, lambda m, n, k: (m,)
    )
    kernel.tile("consumer", _MarkerTile(), (routed, 1, 1)).wait(ready, lambda m, n, k: (m,))

    build = build_runtime_kernel(kernel, _static_options())
    assert isinstance(build.module["r23_moe_style"], tvm.tirx.PrimFunc)


def test_r23_rejects_scalar_reading_init_count():
    kernel = KernelSpec("r23_scalar_init")
    counter = kernel.tensor("counter", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    routed = kernel.scalar("routed", source=(counter, (0,)), range=(1, 4))
    ready = kernel.event("ready", (1,), lambda coord: routed)
    kernel.tile("producer", _MarkerTile(), (routed, 1, 1), writes=[out]).notify(ready, (0,))
    kernel.tile("consumer", _MarkerTile(), (1, 1, 1)).wait(ready, (0,))

    # Spec validation already requires plain positive integers from init
    # count callables, so a scalar-reading callable never reaches the
    # builder; the builder-side branch is defense in depth.
    with pytest.raises(ValueError, match="positive integer"):
        build_runtime_kernel(kernel, _static_options())


def test_r23_rejects_fiber_mismatched_callable_init_count():
    kernel = KernelSpec("r23_bad_callable")
    counter = kernel.tensor("counter", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    routed = kernel.scalar("routed", source=(counter, (0,)), range=(1, 4))
    ready = kernel.event("ready", (1,), lambda coord: 2)
    kernel.tile("producer", _MarkerTile(), (routed, 1, 1), writes=[out]).notify(ready, (0,))
    kernel.tile("consumer", _MarkerTile(), (1, 1, 1)).wait(ready, (0,))

    with pytest.raises(ValueError, match="provides 4 notifications per coordinate"):
        build_runtime_kernel(kernel, _static_options())


def test_r23_callable_init_count_plain_int_passes():
    kernel = KernelSpec("r23_callable")
    counter = kernel.tensor("counter", (1,), "int32")
    out = kernel.tensor("out", (64,), "int32")
    routed = kernel.scalar("routed", source=(counter, (0,)), range=(1, 4))
    ready = kernel.event("ready", (1,), lambda coord: 4)
    kernel.tile("producer", _MarkerTile(), (routed, 1, 1), writes=[out]).notify(ready, (0,))
    kernel.tile("consumer", _MarkerTile(), (1, 1, 1)).wait(ready, (0,))

    build = build_runtime_kernel(kernel, _static_options())
    assert isinstance(build.module["r23_callable"], tvm.tirx.PrimFunc)


def test_r24_declared_axis_plus_auto_axis():
    kernel = KernelSpec("r24")
    count_r = kernel.tensor("count_r", (1,), "int32")
    count_c = kernel.tensor("count_c", (1,), "int32")
    out = kernel.tensor("out", (8, 8), "int32")
    rows = kernel.scalar("rows", source=(count_r, (0,)), range=(1, 8))
    cols = kernel.scalar("cols", source=(count_c, (0,)), range=(1, 8))
    kernel.tile(
        "t",
        _MarkerTile(),
        (rows, cols, 1),
        writes=[out],
        attrs={"megakernel.run_predicate": (0, "lt", rows)},
    )

    build = build_runtime_kernel(kernel, _static_options())
    script = build.module["r24"].script()
    # The declared rows guard and the auto-generated cols guard both load
    # their scalar buffers.
    assert "count_r" in script
    assert "count_c" in script


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


def test_demo_dynamic_count_gpu():
    _require_cuda_sm100()
    from tvm.megakernel.demo.runner import run_dynamic_count

    report = run_dynamic_count(counts=(1, 8, 37, 64))
    assert all(check["ok"] for check in report["checks"])


def test_demo_dynamic_count_gpu_profiler():
    _require_cuda_sm100()
    from tvm.megakernel.demo.runner import run_dynamic_count

    report = run_dynamic_count(counts=(8,), profiler_on=True)
    assert all(check["ok"] for check in report["checks"])


def test_demo_static_count_auto_guard_gpu():
    _require_cuda_sm100()
    from tvm.megakernel.demo.runner import run_dynamic_count

    # F4: the static path enumerates the scalar upper bound and the
    # auto-generated guard gates execution on the runtime count.
    report = run_dynamic_count(counts=(1, 8, 37, 64), scheduler="static")
    assert all(check["ok"] for check in report["checks"])


def test_demo_case_b_post_run_push_gpu():
    _require_cuda_sm100()
    from tvm.megakernel.demo.runner import run_case_b

    # F1 case B: the pusher tile itself computes the dispatch count on
    # device; the push must land after its run.
    report = run_case_b(in_pairs=((3, 5), (20, 17), (1, 0), (63, 1)))
    assert all(check["ok"] for check in report["checks"])


def test_r24_two_scalar_axes_gpu():
    _require_cuda_sm100()
    import numpy as np

    class FillTile(TileImpl):
        def __init__(self, out):
            super().__init__()
            self.out = out

        def run(self, m_idx, n_idx, k_idx):
            T.buffer_store(self.out, m_idx * 8 + n_idx + 1, [m_idx, n_idx])

    kernel = KernelSpec("r24_gpu")
    count_r = kernel.tensor("count_r", (1,), "int32")
    count_c = kernel.tensor("count_c", (1,), "int32")
    out_spec = kernel.tensor("out", (8, 8), "int32")
    rows = kernel.scalar("rows", source=(count_r, (0,)), range=(1, 8))
    cols = kernel.scalar("cols", source=(count_c, (0,)), range=(1, 8))
    kernel.tile(
        "fill",
        FillTile(out_spec),
        (rows, cols, 1),
        writes=[out_spec],
        attrs={"megakernel.run_predicate": (0, "lt", rows)},
    )
    kernel.validate()
    build = build_runtime_kernel(kernel, LoweringOptions(scheduler="static"))
    lib = tvm.compile(build.module, tvm.target.Target("cuda"), tir_pipeline="tirx")

    dev = tvm.cuda(0)
    cr_dev = tvm.runtime.tensor(np.zeros((1,), dtype=np.int32), dev)
    cc_dev = tvm.runtime.tensor(np.zeros((1,), dtype=np.int32), dev)
    out_dev = tvm.runtime.tensor(np.zeros((8, 8), dtype=np.int32), dev)
    queue_dev = tvm.runtime.tensor(build.exec_queue.copy(), dev)
    func = lib["r24_gpu"]
    for rows_v, cols_v in ((3, 5), (8, 1), (1, 8)):
        cr_dev.copyfrom(np.array([rows_v], dtype=np.int32))
        cc_dev.copyfrom(np.array([cols_v], dtype=np.int32))
        out_dev.copyfrom(np.zeros((8, 8), dtype=np.int32))
        func(cr_dev, cc_dev, out_dev, queue_dev)
        dev.sync()
        expected = np.zeros((8, 8), dtype=np.int32)
        expected[:rows_v, :cols_v] = np.arange(1, 65, dtype=np.int32).reshape(8, 8)[
            :rows_v, :cols_v
        ]
        np.testing.assert_array_equal(out_dev.numpy(), expected)
