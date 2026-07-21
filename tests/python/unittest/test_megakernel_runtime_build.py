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
from tvm.megakernel.runtime import (
    DynamicTileScheduler,
    HardwareConfig,
    StaticTileScheduler,
    unpack_from_32bit_host,
)
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


def test_build_runtime_kernel_requires_runtime_scheduler():
    kernel = _chain_kernel("misrouted_chain")
    with pytest.raises(ValueError, match="scheduler="):
        build_runtime_kernel(kernel, LoweringOptions())


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
        "exec_task_handle",
        "exec_head_handle",
        "exec_tail_handle",
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
    # Four run hook invocations per task with n = n_idx * 4 + step.
    assert script.count("T.cuda.thread_fence()") == 1 + 4
    # The push count carries the divided extent.
    assert "count * 4" in script


def test_dynamic_escape_hatch_declares_push_rule():
    kernel = KernelSpec("hatched")
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
