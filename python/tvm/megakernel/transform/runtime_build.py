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
"""Runtime-based static builder for logical megakernel specifications.

This is the ``scheduler="static"`` backend: it lowers any validated
``KernelSpec`` to a production-structure persistent kernel assembled from the
``tvm.megakernel.runtime`` building blocks, generalizing the hand-written
``tirx_kernels.megakernel.moe`` ``get_func_static`` skeleton.  Nothing here is
MoE-specific.

Result surface
--------------
``lower_to_tirx_module``/``lower_to_tirx`` keep their historical return types
(an ``IRModule``/``PrimFunc``); they route here when
``LoweringOptions.scheduler == "static"`` and expose only the module.  The
host-side products live behind the richer entry point::

    build = build_runtime_kernel(spec, LoweringOptions(scheduler="static"),
                                 var_values={"rows": 12})
    build.module                # IRModule with one device kernel
    build.exec_queue            # (sm_count, max_tasks) int32 numpy central queue
    build.event_workspace_size  # int32 cells to allocate and ZERO before launch

``var_values`` provides concrete integers for symbolic ``VarSpec``\ s; it is
only needed by the host queue derivation (tile grids must be enumerable) and
may be omitted when every ``tile_num`` is already concrete.

Kernel parameter order
----------------------
1. one scalar ``T.Var`` per registered ``VarSpec`` (registry order),
2. one buffer per registered base tensor (registry order, symbolic dims
   lowered against the scalar vars),
3. ``event_workspace``: ``int32[event_workspace_size]`` when the spec has
   events (upper-bound event shapes plus one completion cell; must be zeroed
   before launch),
4. ``exec_queue``: ``int32[sm_count, StaticTileScheduler.MAX_TASKS]``,
5. ``profiler_buffer``: ``uint64[MegaKernelWrapper.PROFILER_BUFFER_SIZE]``
   when ``options.attrs["profiler"]`` is truthy.

Emitted body order (mirrors ``moe.py`` ``fused_body``)
------------------------------------------------------
wrapper reset -> register tiles -> ``host_init_all`` -> ``T.device_entry`` ->
cta/warp/warpgroup/thread ids from ``HardwareConfig`` -> local allocs ->
profiler init -> dynamic smem declaration (``max_dynamic_smem``) -> wrapper
smem-manager construction -> ``device_init_all`` -> ``class_init_all`` -> per
spec event ``add_etensor`` -> ``set_events_complete`` -> static central-queue
scheduler init -> ``smem_manager.init`` -> ``while scheduler.valid():``
dispatch chain (fallthrough trap) -> profiler finalize -> class finalize.

Dispatch chain (one ``If``/``Then``/``Else`` per job id, tiles in spec order,
then the reserved event-init jobs) emits per tile instance ``(m, n, k)``:

``enter_tile_runtime`` -> ``prefetch`` -> scoped waits (``scheduler.wait`` at
``impl.wait_level``/``impl.wait_mask``) -> ``run`` (with profiler start/stop
when enabled) -> scoped notifies (``scheduler.notify`` at
``impl.notify_scope``) -> ``exit_tile_runtime``.

Tile implementation metadata
----------------------------
Endpoint scopes come from the ``TileImpl`` class attributes ``wait_level``,
``wait_mask``, and ``notify_scope`` (PR-3).  The static runtime semaphore
implements ``cta`` and ``warp`` waits, so those are the only ``wait_level``
values this builder accepts; all four notify scopes are supported.

Profiler wiring is duck-typed: a tile implementation may define a
``profile_event`` attribute (an ``Enum`` member or int consumed by
``T.cuda.timer_start``).  When the profiler is enabled, tiles with a
``profile_event`` are wrapped in start/stop pairs; tiles without one run
unprofiled.  The attribute is ignored when the profiler is off.

Private-copy notes
------------------
The event-workspace layout, job ids, static phase order, and all static
safety guards are shared with the legacy emitter through
``transform.prepare`` (no copy).  ``_replace_tensor_specs`` and the
coord-map handling of ``transform.lower`` are re-implemented here as small
private copies (extended to lower symbolic coord entries through the bound
``VarSpec`` values); importing the legacy emitter module from this one would
needlessly couple the two backends.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from tvm.ir import IRModule
from tvm.script import tirx as T
from tvm.tirx import PrimFunc

from ..dsl import ExprSpec, KernelSpec, ScalarSpec, TensorSpec, VarSpec, eval_expr_like
from ..runtime import (
    HardwareConfig,
    MegaKernelWrapper,
    StaticSemaphore,
    StaticTileScheduler,
    TaskPacking,
    build_static_exec_queue,
)
from ..runtime.device import f_init_const
from .prepare import (
    DEFAULT_END_JOB_ID,
    INIT_EVENT_JOB_ID,
    WAIT_EVENT_INIT_JOB_ID,
    TIRXLoweringPlan,
    _upper_bound_shape_extents,
    lower_expr_like,
    lower_shape,
    prepare_tirx_lowering_plan,
)
from .validate import validate_kernel

if TYPE_CHECKING:
    from .lower import LoweringOptions

#: Wait levels implemented by the static runtime semaphore.
_STATIC_WAIT_LEVELS = ("cta", "warp")


@dataclass(frozen=True)
class RuntimeKernelBuild:
    """Host-visible products of one runtime static build."""

    module: IRModule
    exec_queue: np.ndarray
    central_tasks: tuple[tuple[int, int, int, int], ...]
    event_workspace_size: int
    sm_count: int
    max_tasks: int
    end_task_type: int
    init_event_job_id: int
    wait_event_init_job_id: int
    profiler_on: bool


def _resolve_options(options: LoweringOptions | None) -> LoweringOptions:
    from .lower import LoweringOptions  # local import avoids a module cycle

    if options is None:
        return LoweringOptions(scheduler="static")
    if not isinstance(options, LoweringOptions):
        raise TypeError("options must be a LoweringOptions instance or None")
    if options.scheduler != "static":
        raise ValueError(
            "the runtime builder requires LoweringOptions(scheduler='static'), "
            f"got scheduler={options.scheduler!r}"
        )
    if options.schedule != "static":
        raise ValueError(
            "the runtime static builder requires options.schedule to remain "
            f"'static', got {options.schedule!r}"
        )
    return options


_HARDWARE_ATTR_KEYS = (
    "sm_count",
    "num_threads",
    "warps_per_warpgroup",
    "warpgroup_count",
    "warp_size",
    "max_dynamic_smem",
)


def _hardware_from_options(options: LoweringOptions) -> HardwareConfig:
    overrides = {key: options.attrs[key] for key in _HARDWARE_ATTR_KEYS if key in options.attrs}
    return HardwareConfig(**overrides)


def _prepare_runtime_plan(
    kernel: KernelSpec, options: LoweringOptions, hardware: HardwareConfig
) -> TIRXLoweringPlan:
    """Derive the shared lowering plan, pinning it to the runtime scheduler."""

    attrs = dict(options.attrs)
    attrs["sm_count"] = hardware.sm_count
    # The runtime StaticTileScheduler stops at its fixed end task type and
    # loads exactly MAX_TASKS queue entries per SM; pin the plan to both.
    attrs["end_job_id"] = DEFAULT_END_JOB_ID
    attrs["max_tasks"] = StaticTileScheduler.MAX_TASKS
    return prepare_tirx_lowering_plan(kernel, replace(options, attrs=attrs))


def _validate_runtime_static_tiles(plan: TIRXLoweringPlan) -> None:
    for tile in plan.kernel.tiles:
        if tile.impl.wait_level not in _STATIC_WAIT_LEVELS:
            raise ValueError(
                f"tile {tile.name!r}: the static runtime builder supports "
                f"wait_level {_STATIC_WAIT_LEVELS} only, got {tile.impl.wait_level!r}"
            )


def _replace_tensor_specs(value: Any, buffers: dict[int, Any]) -> Any:
    """Private copy of the legacy emitter's TensorSpec-to-buffer rewrite."""

    if isinstance(value, TensorSpec) and id(value.base_tensor) in buffers:
        return buffers[id(value.base_tensor)]
    if isinstance(value, tuple):
        return tuple(_replace_tensor_specs(item, buffers) for item in value)
    if isinstance(value, list):
        return [_replace_tensor_specs(item, buffers) for item in value]
    if isinstance(value, dict):
        return {
            _replace_tensor_specs(key, buffers): _replace_tensor_specs(item, buffers)
            for key, item in value.items()
        }
    return value


class _RuntimeKernelBuilder:
    """Emit one static persistent kernel from a spec via the runtime library."""

    def __init__(self, plan: TIRXLoweringPlan, hardware: HardwareConfig):
        self.plan = plan
        self.hardware = hardware
        self.options = plan.options
        self.profiler_on = bool(plan.attrs.get("profiler", False))
        self.var_values: dict[int, Any] = {}
        self.tensor_buffers: dict[int, Any] = {}
        self.tensor_patches: list[tuple[Any, str, Any]] = []
        self.event_sems: dict[int, Any] = {}
        self.event_workspace = None
        self.queue = None
        self.profiler_buffer = None
        self.wrapper = MegaKernelWrapper(profiler_on=self.profiler_on, hardware=hardware)

    def emit(self) -> None:
        kernel = self.plan.kernel
        hardware = self.hardware
        wrapper = self.wrapper
        T.func_attr({"global_symbol": kernel.name})
        self._emit_var_args()
        self._emit_tensor_args()
        self._patch_tensor_specs()
        self._emit_special_args()

        wrapper.reset()
        for tile in kernel.tiles:
            wrapper._add_tile(tile.impl, getattr(tile.impl, "profile_event", None))
        wrapper.host_init_all()

        T.device_entry()
        T.cta_id([hardware.sm_count])
        T.warp_id([hardware.warp_count])
        T.warpgroup_id([hardware.warpgroup_count])
        T.thread_id([hardware.num_threads])
        T.thread_id_in_wg([hardware.warpgroup_size])
        T.lane_id([hardware.warp_size])
        T.alloc_buffer([1], "uint32", scope="local", align=8)
        T.alloc_buffer([1], "uint64", scope="local", align=8)
        wrapper.init_profiler(self.profiler_buffer)
        smem = T.alloc_buffer([hardware.max_dynamic_smem], "uint8", scope="shared.dyn")
        wrapper.set_smem_manager(hardware.max_dynamic_smem, self.options.smem_chunk_size, smem.data)
        wrapper.device_init_all(wrapper.smem_manager)
        wrapper.class_init_all(wrapper.smem_manager)
        self._emit_events()
        wrapper.init_tile_scheduler(
            False,
            StaticTileScheduler,
            kernel.name,
            self.queue,
            wrapper.smem_manager,
            self.plan.attrs.get("debug_scheduler", False),
        )
        wrapper.smem_manager.init()

        with T.While(wrapper.tile_scheduler.valid()):
            self._emit_dispatch()
            wrapper.tile_scheduler.next_tile()

        if self.profiler_on:
            wrapper.profiler.finalize(T.lane_id([hardware.warp_size]) == 0)
        wrapper.class_finalize_all(wrapper.smem_manager)

    def restore_tensor_specs(self) -> None:
        for impl, name, value in reversed(self.tensor_patches):
            setattr(impl, name, value)

    def _emit_var_args(self) -> None:
        for binding in self.plan.var_bindings:
            name = binding.param_name
            self.var_values[id(binding.var)] = T.arg(name, T.Var(name, binding.var.dtype))

    def _shape(self, shape, label: str) -> tuple[Any, ...]:
        return lower_shape(shape, self.var_values, label)

    def _emit_tensor_args(self) -> None:
        for binding in self.plan.tensor_bindings:
            shape = self._shape(binding.tensor.shape, f"tensor {binding.tensor.name!r}")
            buffer = T.arg(binding.param_name, T.Buffer(shape, binding.tensor.dtype))
            self.tensor_buffers[id(binding.tensor)] = buffer

    def _emit_special_args(self) -> None:
        if self.plan.event_workspace_size:
            self.event_workspace = T.arg(
                "event_workspace",
                T.Buffer((self.plan.event_workspace_size,), "int32"),
            )
        self.queue = T.arg(
            "exec_queue",
            T.Buffer((self.hardware.sm_count, StaticTileScheduler.MAX_TASKS), "int32"),
        )
        if self.profiler_on:
            self.profiler_buffer = T.arg(
                "profiler_buffer",
                T.Buffer((MegaKernelWrapper.PROFILER_BUFFER_SIZE,), "uint64"),
            )

    def _patch_tensor_specs(self) -> None:
        for tile in self.plan.kernel.tiles:
            impl = tile.impl
            for name, value in vars(impl).items():
                replaced = _replace_tensor_specs(value, self.tensor_buffers)
                if replaced is not value:
                    self.tensor_patches.append((impl, name, value))
                    setattr(impl, name, replaced)

    def _emit_events(self) -> None:
        plan = self.plan
        if not plan.event_layouts:
            return
        wrapper = self.wrapper
        for layout in plan.event_layouts:
            event = layout.event
            shape = list(_upper_bound_shape_extents(event.shape, f"event {event.name!r} shape"))
            if callable(event.init_count):

                def f_init(*coord, event=event):
                    return event.init_count(tuple(coord))

            else:
                f_init = f_init_const(event.init_count)
            semaphore = wrapper.add_etensor(StaticSemaphore, self.event_workspace, shape, f_init)
            # add_etensor constructs semaphores with default hardware; bind the
            # configured one so warp-level waits see the right warp geometry.
            semaphore.hardware = self.hardware
            self.event_sems[id(event)] = semaphore
        wrapper.set_events_complete(False, StaticSemaphore, self.event_workspace)
        wrapper.evt_etensor_init_complete.hardware = self.hardware
        if wrapper.etensor_workspace_offset != plan.event_workspace_size:
            raise ValueError(
                "event workspace layout diverged from its static plan: "
                f"{wrapper.etensor_workspace_offset} != {plan.event_workspace_size}"
            )

    def _event_coord(self, coord_map, indices, label: str) -> tuple[Any, ...]:
        coord = coord_map(*indices) if callable(coord_map) else coord_map
        if not isinstance(coord, tuple | list):
            raise TypeError(f"{label} event coordinate map must return a tuple or list")
        return tuple(
            lower_expr_like(entry, self.var_values, label)
            if isinstance(entry, int | VarSpec | ExprSpec | ScalarSpec)
            else entry
            for entry in coord
        )

    def _emit_dispatch(self) -> None:
        plan = self.plan
        wrapper = self.wrapper
        entries: list[tuple[int, Any]] = [
            (tile_plan.job_id, tile_plan.tile) for tile_plan in plan.tile_plans
        ]
        if plan.event_layouts:
            entries.extend(
                [
                    (INIT_EVENT_JOB_ID, "init_event"),
                    (WAIT_EVENT_INIT_JOB_ID, "wait_event_init"),
                ]
            )
        task_type = wrapper.tile_scheduler.task_type
        if_frames = [T.If(task_type == job_id) for job_id, _ in entries]
        then_frames = [T.Then() for _ in entries]
        else_frames = [T.Else() for _ in entries]
        for index, (_, entry) in enumerate(entries):
            if_frames[index].__enter__()
            with then_frames[index]:
                if entry == "init_event":
                    wrapper.task_impl_init_etensor(False)
                elif entry == "wait_event_init":
                    wrapper.task_impl_wait_etensor_init_complete(False)
                else:
                    self._emit_tile(entry)
            else_frames[index].__enter__()
        T.evaluate(T.cuda.trap_when_assert_failed(False))
        for index in range(len(entries) - 1, -1, -1):
            else_frames[index].__exit__(None, None, None)
            if_frames[index].__exit__(None, None, None)

    def _emit_tile(self, tile) -> None:
        wrapper = self.wrapper
        scheduler = wrapper.tile_scheduler
        smem_manager = wrapper.smem_manager
        impl = tile.impl
        indices = (scheduler.m_idx, scheduler.n_idx, scheduler.k_idx)
        smem_manager.enter_tile_runtime(impl)
        wrapper.run_tile_prefetch(impl, *indices)
        for event, coord_map in tile.waits:
            coord = self._event_coord(coord_map, indices, f"tile {tile.name!r} wait")
            scheduler.wait(
                self.event_sems[id(event)],
                *coord,
                wait_level=impl.wait_level,
                mask=impl.wait_mask,
            )
        if self.profiler_on and getattr(impl, "profile_event", None) is not None:
            wrapper.run_tile(impl, *indices)
        else:
            impl.run(*indices)
        for event, coord_map in tile.notifies:
            coord = self._event_coord(coord_map, indices, f"tile {tile.name!r} notify")
            semaphore = self.event_sems[id(event)]

            def notify_func(_notify_idx, coord=coord):
                return (1, -1, *coord)

            scope, scope_id = impl.notify_scope
            scheduler.notify(semaphore, notify_func, scope=scope, scope_id=scope_id, release=True)
        smem_manager.exit_tile_runtime()


@T.jit(check_well_formed=False)
def _runtime_kernel_entry(*, emitter: T.constexpr):
    emitter.emit()


def _emit_runtime_func(
    kernel: KernelSpec, options: LoweringOptions
) -> tuple[PrimFunc, TIRXLoweringPlan, HardwareConfig]:
    validate_kernel(kernel)
    hardware = _hardware_from_options(options)
    plan = _prepare_runtime_plan(kernel, options, hardware)
    _validate_runtime_static_tiles(plan)
    builder = _RuntimeKernelBuilder(plan, hardware)
    try:
        func = _runtime_kernel_entry.specialize(emitter=builder)
    finally:
        builder.restore_tensor_specs()
    return func, plan, hardware


def emit_runtime_module(kernel: KernelSpec, options: LoweringOptions) -> IRModule:
    """Lower a validated spec to its runtime-built static device kernel module."""

    resolved = _resolve_options(options)
    func, _, _ = _emit_runtime_func(kernel, resolved)
    return IRModule({kernel.name: func})


def _var_env(kernel: KernelSpec, var_values: dict[str, int] | None) -> dict[VarSpec, int] | None:
    if var_values is None:
        return None
    if not isinstance(var_values, dict):
        raise TypeError("var_values must be a dict mapping variable names to integers")
    env = {}
    for name, value in var_values.items():
        var = kernel.vars.get(name)
        if var is None:
            raise ValueError(f"queue derivation got a value for unknown var {name!r}")
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"var {name!r} queue value must be an integer")
        env[var] = value
    return env


def derive_static_central_tasks(
    plan: TIRXLoweringPlan, var_values: dict[str, int] | None = None
) -> list[tuple[int, int, int, int]]:
    """Enumerate the static central task list ``(m, n, k, job_id)`` in order.

    The phase order comes from the shared lowering plan: event-init tasks,
    entry tiles (no waits), event-init wait tasks, then waiting tiles in
    stable topological order over the event DAG.  The END marker is not
    listed; the host queue builder pads it.
    """

    schedule = plan.static_schedule
    if schedule is None:
        raise ValueError("static queue derivation requires a static schedule plan")
    env = _var_env(plan.kernel, var_values)
    packing = TaskPacking()
    limits = (packing.max_m_idx, packing.max_n_idx, packing.max_k_idx)
    tasks = []
    for phase in schedule.phases:
        if phase.job_id == schedule.end_job_id:
            continue
        extents = []
        for axis, extent in enumerate(phase.tile_num):
            value = eval_expr_like(extent, env)
            if value is None:
                raise ValueError(
                    f"static queue derivation for phase {phase.label!r} needs a "
                    "concrete value for every symbolic tile_num variable; pass "
                    "var_values"
                )
            if value > limits[axis]:
                raise ValueError(
                    f"static phase {phase.label!r} axis {axis} extent {value} "
                    f"exceeds the packed-task limit {limits[axis]}"
                )
            extents.append(value)
        for m_idx in range(extents[0]):
            for n_idx in range(extents[1]):
                for k_idx in range(extents[2]):
                    tasks.append((m_idx, n_idx, k_idx, phase.job_id))
    return tasks


def build_static_queues(
    plan: TIRXLoweringPlan,
    hardware: HardwareConfig,
    var_values: dict[str, int] | None = None,
) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """Deal the central task list into the per-SM static exec queue array."""

    central = derive_static_central_tasks(plan, var_values)
    sm_count = hardware.sm_count
    columns = (len(central) + sm_count - 1) // sm_count + 1
    if columns > StaticTileScheduler.MAX_TASKS:
        raise ValueError(
            f"static central queue needs {columns} columns ({len(central)} tasks "
            f"on {sm_count} SMs plus the END row), exceeding the scheduler "
            f"capacity {StaticTileScheduler.MAX_TASKS}"
        )
    queue = build_static_exec_queue(
        central,
        sm_count=sm_count,
        max_tasks=StaticTileScheduler.MAX_TASKS,
        end_task_type=plan.static_schedule.end_job_id,
    )
    return central, queue


def build_runtime_kernel(
    kernel: KernelSpec,
    options: LoweringOptions | None = None,
    var_values: dict[str, int] | None = None,
) -> RuntimeKernelBuild:
    """Lower a spec with the runtime static builder and derive its host queue.

    ``var_values`` maps symbolic ``VarSpec`` names to concrete integers; it is
    required only when some ``tile_num`` is symbolic (the device kernel itself
    is symbolic-safe, but the host queue must enumerate concrete grids).
    """

    resolved = _resolve_options(options)
    func, plan, hardware = _emit_runtime_func(kernel, resolved)
    module = IRModule({kernel.name: func})
    central, queue = build_static_queues(plan, hardware, var_values)
    return RuntimeKernelBuild(
        module=module,
        exec_queue=queue,
        central_tasks=tuple(central),
        event_workspace_size=plan.event_workspace_size,
        sm_count=hardware.sm_count,
        max_tasks=StaticTileScheduler.MAX_TASKS,
        end_task_type=plan.static_schedule.end_job_id,
        init_event_job_id=INIT_EVENT_JOB_ID,
        wait_event_init_job_id=WAIT_EVENT_INIT_JOB_ID,
        profiler_on=bool(plan.attrs.get("profiler", False)),
    )


__all__ = [
    "RuntimeKernelBuild",
    "build_runtime_kernel",
    "build_static_queues",
    "derive_static_central_tasks",
    "emit_runtime_module",
]
