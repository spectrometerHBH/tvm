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
"""Runtime-based static/dynamic builder for logical megakernel specifications.

This is the ``scheduler="static"``/``scheduler="dynamic"`` backend: it lowers
any validated ``KernelSpec`` to a production-structure persistent kernel
assembled from the ``tvm.megakernel.runtime`` building blocks, generalizing
the hand-written ``tirx_kernels.megakernel.moe`` ``get_func_static`` /
``get_func_dynamic`` skeletons.  Nothing here is MoE-specific.

Result surface
--------------
``lower_to_tirx_module``/``lower_to_tirx`` keep their historical return types
(an ``IRModule``/``PrimFunc``); they route here when
``LoweringOptions.scheduler`` is ``"static"`` or ``"dynamic"`` and expose only
the module.  The host-side products live behind the richer entry point::

    build = build_runtime_kernel(spec, LoweringOptions(scheduler="static"),
                                 var_values={"rows": 12})
    build.module                # IRModule with one device kernel
    build.exec_queue            # static: (sm_count, max_tasks) int32 queue
    build.queue_tasks/head/tail # dynamic: MPMC seed arrays (int32)
    build.event_workspace_size  # int32 cells to allocate and ZERO before launch

``var_values`` provides concrete integers for symbolic ``VarSpec``\ s; it is
only needed by the host queue derivation (seed/static grids must be
enumerable) and may be omitted when every seeded ``tile_num`` is concrete.
Between launches the host must re-upload the queue arrays and re-zero the
event workspace (the device mutates both).

Dynamic scheduling (``scheduler="dynamic"``)
--------------------------------------------
The dynamic path emits the production MPMC persistent scheduler: params
``exec_task``/``exec_head``/``exec_tail`` (``DynamicTileScheduler.MAX_TASKS``
slots) replace the static exec queue, events use the two-phase dynamic
semaphore, and only the event-init tasks and the entry tiles (no waits) are
seeded on the host.  Every other tile is *dynamically dispatched*: when the
last producer task for an event cell starts (its pre-notify observes
``old % base == 1``), it pushes the consumer's tasks into the MPMC queue via
``pre_notify_and_push``.

Dispatch synthesis derives, for each non-entry tile, its pusher from the
event edge (the single producer of the single event it waits on), the push
count and ``(m, n, k)`` index map from the tile's ``tile_num`` (with runtime
scalars lowered to device loads), and the pre-notify scope from the pusher's
``notify_scope`` metadata.  Wait/notify coord maps must be tuples (or
callables returning tuples) whose entries are integer constants or bare
``m``/``n``/``k`` axis references; anything richer needs the escape hatch.
Exactly one terminal tile (no notifies) is required: a drain event is
synthesized for it (shape ``(1,)``, after the user events in the workspace,
no completion cell in dynamic mode), runtime-initialized with
``(base + 1) * grid_count`` by the tile producing the scalar's source tensor
(or by the terminal tile's pusher when the scalar is host-planted), and its
last pre-notify pushes one END task per SM.  ``tile_coalescing`` in
``options.attrs`` maps a tile name to an integer ``q`` grouping ``q`` n-axis
slices per pushed task (the run hook is repeated with ``n = n_idx * q + i``).

Escape hatch: a tile may declare its push rule explicitly with
``tile.attrs["megakernel.dispatch"] = {...}``; see ``_ESCAPE_HATCH_KEY`` for
the schema.  The dynamic persistent queue also requires the tile-level event
dependency graph to be acyclic (persistent workers block on waits), matching
the production policy check.

Kernel parameter order
----------------------
1. one scalar ``T.Var`` per registered ``VarSpec`` (registry order),
2. one buffer per registered base tensor (registry order, symbolic dims
   lowered against the scalar vars),
3. ``event_workspace``: ``int32[event_workspace_size]`` when the spec has
   events (upper-bound event shapes; static adds one completion cell, dynamic
   adds the drain cell; must be zeroed before launch),
4. static: ``exec_queue`` ``int32[sm_count, StaticTileScheduler.MAX_TASKS]``;
   dynamic: ``exec_task`` ``int32[DynamicTileScheduler.MAX_TASKS]`` and
   ``exec_head``/``exec_tail`` ``int32[1]``,
5. ``profiler_buffer``: ``uint64[MegaKernelWrapper.PROFILER_BUFFER_SIZE]``
   when ``options.attrs["profiler"]`` is truthy.

Emitted body order (mirrors ``moe.py`` ``fused_body``)
------------------------------------------------------
wrapper reset -> register tiles -> ``host_init_all`` -> ``T.device_entry`` ->
cta/warp/warpgroup/thread ids from ``HardwareConfig`` -> local allocs ->
profiler init -> dynamic smem declaration (``max_dynamic_smem``) -> wrapper
smem-manager construction -> ``device_init_all`` -> ``class_init_all`` -> per
spec event ``add_etensor`` -> ``set_events_complete`` -> scheduler init
(static central queue / dynamic MPMC with first-task prefetch) ->
``smem_manager.init`` -> ``while scheduler.valid():`` dispatch chain
(fallthrough trap) -> profiler finalize -> class finalize.

Dispatch chain (one ``If``/``Then``/``Else`` per job id, tiles in spec order,
then the reserved event-init jobs) emits per tile instance ``(m, n, k)``:

``enter_tile_runtime`` -> [dynamic only: ``pre_notify_and_push``] ->
``prefetch`` -> scoped waits (``scheduler.wait`` at
``impl.wait_level``/``impl.wait_mask``) -> ``run`` (repeated
``tile_coalescing`` times with the n index expanded, with profiler start/stop
when enabled) -> [dynamic only: drain-event runtime init in the writer tile]
-> scoped notifies (``scheduler.notify`` at ``impl.notify_scope``) ->
``exit_tile_runtime``.

Tile implementation metadata
----------------------------
Endpoint scopes come from the ``TileImpl`` class attributes ``wait_level``,
``wait_mask``, and ``notify_scope`` (PR-3).  Both runtime semaphores
implement ``cta`` and ``warp`` waits, so those are the only ``wait_level``
values this builder accepts; all four notify scopes are supported.  In
dynamic mode a pusher's ``notify_scope`` additionally drives its pre-notify
and push scope.

Profiler wiring is duck-typed: a tile implementation may define a
``profile_event`` attribute (an ``Enum`` member or int consumed by
``T.cuda.timer_start``).  When the profiler is enabled, tiles with a
``profile_event`` are wrapped in start/stop pairs; tiles without one run
unprofiled.  The attribute is ignored when the profiler is off.  In dynamic
mode the scheduler itself emits FETCH/PUSH events.

Private-copy notes
------------------
The static event-workspace layout, job ids, static phase order, and all
static safety guards are shared with the legacy emitter through
``transform.prepare`` (no copy); the dynamic path derives its own plan
(bindings, layouts, dispatch rules) from the same prepare helpers because the
legacy plan rejects runtime scalars by construction.  ``_replace_tensor_specs``
and the coord-map handling of ``transform.lower`` are re-implemented here as
small private copies (extended to lower symbolic and runtime-scalar coord
entries); importing the legacy emitter module from this one would needlessly
couple the two backends.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from tvm.ir import IRModule
from tvm.script import tirx as T
from tvm.tirx import PrimFunc

from ..dsl import (
    EventSpec,
    ExprSpec,
    KernelSpec,
    ScalarSpec,
    TensorSpec,
    TileSpec,
    VarSpec,
    eval_expr_like,
    expr_bounds,
    expr_vars,
)
from ..dsl.spec import expr_scalars
from ..runtime import (
    DynamicSemaphore,
    DynamicTileScheduler,
    HardwareConfig,
    MegaKernelWrapper,
    MPMCQueueHost,
    SemaphoreBase,
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
    TensorBinding,
    TileLoweringPlan,
    TIRXLoweringPlan,
    VarBinding,
    _sanitize_identifier,
    _upper_bound_shape_extents,
    _validate_event_counter_encoding,
    lower_expr_like,
    lower_shape,
    prepare_tirx_lowering_plan,
    upper_bound_shape_product,
)
from .validate import validate_kernel

if TYPE_CHECKING:
    from .lower import LoweringOptions

#: Wait levels implemented by the static runtime semaphore.
_STATIC_WAIT_LEVELS = ("cta", "warp")


@dataclass(frozen=True)
class DrainEventInfo:
    """Host-visible description of one synthesized dynamic drain event."""

    name: str
    workspace_offset: int
    static_count: int | None
    runtime_initialized: bool


@dataclass(frozen=True)
class RuntimeKernelBuild:
    """Host-visible products of one runtime build."""

    module: IRModule
    scheduler: str
    exec_queue: np.ndarray | None
    queue_tasks: np.ndarray | None
    queue_head: np.ndarray | None
    queue_tail: np.ndarray | None
    central_tasks: tuple[tuple[int, int, int, int], ...]
    event_workspace_size: int
    sm_count: int
    max_tasks: int
    end_task_type: int
    init_event_job_id: int
    wait_event_init_job_id: int
    profiler_on: bool
    drain_events: tuple[DrainEventInfo, ...] = ()


def _resolve_options(options: LoweringOptions | None) -> LoweringOptions:
    from .lower import LoweringOptions  # local import avoids a module cycle

    if options is None:
        return LoweringOptions(scheduler="static")
    if not isinstance(options, LoweringOptions):
        raise TypeError("options must be a LoweringOptions instance or None")
    if options.scheduler not in ("static", "dynamic"):
        raise ValueError(
            "the runtime builder requires LoweringOptions(scheduler='static' or "
            f"'dynamic'), got scheduler={options.scheduler!r}"
        )
    if options.schedule != "static":
        raise ValueError(
            "the runtime builder requires options.schedule to remain "
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


def _lower_runtime_expr(value, var_values, scalar_load, label: str):
    """Lower a logical integer expression, resolving runtime scalar loads."""

    if isinstance(value, bool):
        raise TypeError(f"{label} must not contain boolean values")
    if isinstance(value, int):
        return value
    if isinstance(value, VarSpec):
        if id(value) not in var_values:
            raise ValueError(f"{label} contains an unbound symbolic variable")
        return var_values[id(value)]
    if isinstance(value, ScalarSpec):
        return scalar_load(value)
    if not isinstance(value, ExprSpec):
        raise TypeError(f"{label} must contain only int, VarSpec, ScalarSpec, or ExprSpec values")
    args = [_lower_runtime_expr(arg, var_values, scalar_load, label) for arg in value.args]
    if value.op == "add":
        return args[0] + args[1]
    if value.op == "sub":
        return args[0] - args[1]
    if value.op == "mul":
        return args[0] * args[1]
    if value.op == "floordiv":
        return args[0] // args[1]
    if value.op == "mod":
        return args[0] % args[1]
    if value.op == "neg":
        return -args[0]
    if value.op == "ceildiv":
        return -((-args[0]) // args[1])
    raise ValueError(f"{label} uses unsupported ExprSpec op {value.op!r}")


class _RuntimeKernelBuilder:
    """Emit one persistent kernel from a spec via the runtime library."""

    is_dynamic = False

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
        self._init_scheduler()
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
        self._emit_profiler_arg()

    def _emit_profiler_arg(self) -> None:
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

    def _scalar_load(self, scalar: ScalarSpec):
        tensor, index = scalar.source
        buffer = self.tensor_buffers.get(id(tensor))
        if buffer is None:
            raise ValueError(
                f"scalar {scalar.name!r} source tensor {tensor.name!r} is not a kernel tensor"
            )
        label = f"scalar {scalar.name!r} source index"
        lowered = tuple(lower_expr_like(entry, self.var_values, label) for entry in index)
        return buffer[lowered]

    def _lower_expr(self, value, label: str):
        return _lower_runtime_expr(value, self.var_values, self._scalar_load, label)

    def _lower_event_coord(self, coord_map, indices, label: str) -> tuple[Any, ...]:
        coord = coord_map(*indices) if callable(coord_map) else coord_map
        if not isinstance(coord, tuple | list):
            raise TypeError(f"{label} event coordinate map must return a tuple or list")
        return tuple(
            self._lower_expr(entry, label)
            if isinstance(entry, int | VarSpec | ExprSpec | ScalarSpec)
            else entry
            for entry in coord
        )

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

    def _init_scheduler(self) -> None:
        self.wrapper.init_tile_scheduler(
            False,
            StaticTileScheduler,
            self.plan.kernel.name,
            self.queue,
            self.wrapper.smem_manager,
            self.plan.attrs.get("debug_scheduler", False),
        )

    def _dispatch_extra_entries(self) -> list[tuple[int, Any]]:
        if self.plan.event_layouts:
            return [
                (INIT_EVENT_JOB_ID, "init_event"),
                (WAIT_EVENT_INIT_JOB_ID, "wait_event_init"),
            ]
        return []

    def _emit_dispatch(self) -> None:
        plan = self.plan
        wrapper = self.wrapper
        entries: list[tuple[int, Any]] = [
            (tile_plan.job_id, tile_plan.tile) for tile_plan in plan.tile_plans
        ]
        entries.extend(self._dispatch_extra_entries())
        task_type = wrapper.tile_scheduler.task_type
        if_frames = [T.If(task_type == job_id) for job_id, _ in entries]
        then_frames = [T.Then() for _ in entries]
        else_frames = [T.Else() for _ in entries]
        for index, (_, entry) in enumerate(entries):
            if_frames[index].__enter__()
            with then_frames[index]:
                if entry == "init_event":
                    wrapper.task_impl_init_etensor(self.is_dynamic)
                elif entry == "wait_event_init":
                    wrapper.task_impl_wait_etensor_init_complete(self.is_dynamic)
                else:
                    self._emit_tile(entry)
            else_frames[index].__enter__()
        T.evaluate(T.cuda.trap_when_assert_failed(False))
        for index in range(len(entries) - 1, -1, -1):
            else_frames[index].__exit__(None, None, None)
            if_frames[index].__exit__(None, None, None)

    def _run_repeat_count(self, tile) -> int:
        return 1

    def _emit_tile_pre_steps(self, tile, indices) -> None:
        """Dynamic-only steps emitted before the tile's prefetch."""

    def _emit_tile_post_run_steps(self, tile, indices) -> None:
        """Dynamic-only steps emitted after the run, before the notifies."""

    def _emit_run(self, tile, indices) -> None:
        impl = tile.impl
        repeat = self._run_repeat_count(tile)
        for step in range(repeat):
            run_indices = (
                (indices[0], indices[1] * repeat + step, indices[2]) if repeat != 1 else indices
            )
            if self.profiler_on and getattr(impl, "profile_event", None) is not None:
                self.wrapper.run_tile(impl, *run_indices)
            else:
                impl.run(*run_indices)

    def _emit_tile(self, tile) -> None:
        wrapper = self.wrapper
        scheduler = wrapper.tile_scheduler
        smem_manager = wrapper.smem_manager
        impl = tile.impl
        indices = (scheduler.m_idx, scheduler.n_idx, scheduler.k_idx)
        smem_manager.enter_tile_runtime(impl)
        self._emit_tile_pre_steps(tile, indices)
        wrapper.run_tile_prefetch(impl, *indices)
        for event, coord_map in tile.waits:
            coord = self._lower_event_coord(coord_map, indices, f"tile {tile.name!r} wait")
            scheduler.wait(
                self.event_sems[id(event)],
                *coord,
                wait_level=impl.wait_level,
                mask=impl.wait_mask,
            )
        self._emit_run(tile, indices)
        self._emit_tile_post_run_steps(tile, indices)
        for event, coord_map in tile.notifies:
            coord = self._lower_event_coord(coord_map, indices, f"tile {tile.name!r} notify")
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
) -> tuple[PrimFunc, Any, HardwareConfig]:
    validate_kernel(kernel)
    hardware = _hardware_from_options(options)
    if options.scheduler == "dynamic":
        plan = _prepare_dynamic_plan(kernel, options, hardware)
        _validate_runtime_static_tiles(plan)
        builder: _RuntimeKernelBuilder = _RuntimeDynamicKernelBuilder(plan, hardware)
    else:
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


# ---------------------------------------------------------------------------
# Dynamic scheduling: plan, dispatch synthesis, emission, host seed queues
# ---------------------------------------------------------------------------

#: Tile attribute key for the explicit dynamic dispatch rule (escape hatch).
#: The value is a dict with keys:
#:   - ``source`` (required): name of the pusher tile.
#:   - ``count`` (required): per-push enqueue count, an int/ExprLike or a
#:     ``callable(m, n, k)`` over the source task indices.
#:   - ``indices`` (required): ``callable(push_idx, m, n, k) -> (m, n, k)``
#:     mapping one pushed index to a target task index.
#:   - ``event``: name of the trigger event (default: the tile's single
#:     waited event).
#:   - ``pre_scope``: ``(scope, scope_id)`` of the pre-notify (default: the
#:     source impl's ``notify_scope``).
#:   - ``push_level``: enqueue granularity (default: the pre-notify scope).
_ESCAPE_HATCH_KEY = "megakernel.dispatch"

_PUSH_SCOPES = ("thread", "warp", "warpgroup", "cta")
_PUSH_SCOPE_ORDER = {scope: order for order, scope in enumerate(_PUSH_SCOPES)}


@dataclass(frozen=True)
class _DynamicEventLayout:
    """One dynamic event-workspace region (user event or synthesized drain)."""

    event: EventSpec | None
    name: str
    shape: tuple[int, ...]
    workspace_offset: int
    is_drain: bool


@dataclass(frozen=True)
class _DrainPlan:
    """Synthesized drain event for the single terminal tile."""

    name: str
    terminal: TileSpec
    static_count: int | None
    count_extents: tuple
    writer: TileSpec | None


@dataclass(frozen=True)
class _DispatchRule:
    """One dynamic push rule attached to its source tile's dispatch branch."""

    source: TileSpec
    target: TileSpec
    event: EventSpec
    coord_map: Any
    pinned: tuple
    free_axes: tuple[int, ...]
    free_extents: tuple
    push_level: str
    pre_scope: tuple[str, int]
    custom_count: Any = None
    custom_indices: Any = None


@dataclass(frozen=True)
class _DynamicPlan:
    """Builder-private dynamic lowering state (mirrors the static plan API)."""

    kernel: KernelSpec
    options: LoweringOptions
    attrs: dict[str, Any]
    var_bindings: tuple[VarBinding, ...]
    tensor_bindings: tuple[TensorBinding, ...]
    event_layouts: tuple[_DynamicEventLayout, ...]
    drain: _DrainPlan
    tile_plans: tuple[TileLoweringPlan, ...]
    dispatch_rules: dict[int, _DispatchRule]
    terminal: TileSpec
    entry_tiles: tuple[TileSpec, ...]
    scheduled: dict[int, tuple]
    coalescing: dict[str, int]
    event_workspace_size: int


class _AxisProbe:
    """Sentinel standing in for one task axis while probing a coord map."""

    def __init__(self, axis: int):
        self.axis = axis


def _probe_coord_map(coord_map, label: str) -> tuple[tuple[str, Any], ...]:
    """Classify coord-map entries as constants or bare axis references."""

    hint = f"declare tile.attrs[{_ESCAPE_HATCH_KEY!r}] to override dispatch synthesis"
    if callable(coord_map):
        probes = (_AxisProbe(0), _AxisProbe(1), _AxisProbe(2))
        try:
            coord = coord_map(*probes)
        except Exception as err:  # pylint: disable=broad-exception-caught
            raise ValueError(
                f"{label}: coord_map must use only integer constants or bare "
                f"m/n/k axis references to be dispatchable; {hint}"
            ) from err
    else:
        coord = coord_map
    if not isinstance(coord, tuple | list):
        raise ValueError(f"{label}: coord_map must return a tuple or list; {hint}")
    entries = []
    for entry in coord:
        if isinstance(entry, _AxisProbe):
            entries.append(("axis", entry.axis))
        elif isinstance(entry, int) and not isinstance(entry, bool):
            entries.append(("const", entry))
        else:
            raise ValueError(
                f"{label}: coord entry {entry!r} is not an integer constant or a "
                f"bare axis reference; {hint}"
            )
    return tuple(entries)


def _event_tile_producers(kernel: KernelSpec) -> dict[int, list[TileSpec]]:
    producers: dict[int, list[TileSpec]] = {id(event): [] for event in kernel.events.values()}
    for tile in kernel.tiles:
        for event, _ in tile.notifies:
            producers.setdefault(id(event), []).append(tile)
    return producers


def _tensor_producers(kernel: KernelSpec) -> dict[int, list[TileSpec]]:
    producers: dict[int, list[TileSpec]] = {}
    for tile in kernel.tiles:
        for tensor in tile.writes:
            producers.setdefault(id(tensor.base_tensor), []).append(tile)
    return producers


def _event_ancestors(kernel: KernelSpec) -> dict[int, set[int]]:
    """Transitive producer ancestors of each tile through event edges."""

    producers = _event_tile_producers(kernel)
    parents: dict[int, set[int]] = {id(tile): set() for tile in kernel.tiles}
    for tile in kernel.tiles:
        for event, _ in tile.waits:
            for producer in producers.get(id(event), ()):
                parents[id(tile)].add(id(producer))
    ancestors: dict[int, set[int]] = {tile_id: set() for tile_id in parents}

    def visit(tile_id: int) -> set[int]:
        if ancestors[tile_id]:
            return ancestors[tile_id]
        for parent in parents[tile_id]:
            ancestors[tile_id].add(parent)
            ancestors[tile_id].update(visit(parent))
        return ancestors[tile_id]

    for tile_id in parents:
        visit(tile_id)
    return ancestors


def _validate_dynamic_acyclic(kernel: KernelSpec) -> None:
    """The dynamic persistent queue requires a tile-level dependency DAG."""

    producers = _event_tile_producers(kernel)
    adjacency: dict[int, set[int]] = {id(tile): set() for tile in kernel.tiles}
    for tile in kernel.tiles:
        for event, _ in tile.waits:
            for producer in producers.get(id(event), ()):
                adjacency[id(producer)].add(id(tile))
    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(tile_id: int) -> None:
        if tile_id in visiting:
            raise ValueError(
                "the dynamic persistent queue requires logical event dependencies to be acyclic"
            )
        if tile_id in visited:
            return
        visiting.add(tile_id)
        for consumer in adjacency[tile_id]:
            visit(consumer)
        visiting.remove(tile_id)
        visited.add(tile_id)

    for tile_id in adjacency:
        visit(tile_id)


def _validate_dynamic_event_encoding(kernel: KernelSpec, options: LoweringOptions) -> None:
    """Prove user-event counters fit the int32 semaphore encoding (as static)."""

    probe = TIRXLoweringPlan(
        kernel=kernel,
        options=options,
        attrs={},
        var_bindings=(),
        tensor_bindings=(),
        event_layouts=(),
        event_init_complete_layout=None,
        tile_plans=(),
        static_schedule=None,
    )
    _validate_event_counter_encoding(probe)


def _parse_coalescing(kernel: KernelSpec, attrs: dict[str, Any]) -> dict[str, int]:
    raw = attrs.get("tile_coalescing")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise TypeError("attrs['tile_coalescing'] must be a dict mapping tile names to ints")
    names = {tile.name for tile in kernel.tiles}
    coalescing = {}
    for name, value in raw.items():
        if name not in names:
            raise ValueError(f"tile_coalescing names unknown tile {name!r}")
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"tile_coalescing[{name!r}] must be a positive integer")
        if value != 1:
            coalescing[name] = value
    return coalescing


def _parse_escape_hatch(tile: TileSpec) -> dict[str, Any]:
    spec = tile.attrs[_ESCAPE_HATCH_KEY]
    label = f"tile {tile.name!r} attrs[{_ESCAPE_HATCH_KEY!r}]"
    if not isinstance(spec, dict):
        raise TypeError(f"{label} must be a dict")
    allowed = {"source", "count", "indices", "event", "pre_scope", "push_level"}
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(f"{label} has unknown keys {sorted(unknown)}; allowed: {sorted(allowed)}")
    for key in ("source", "count", "indices"):
        if key not in spec:
            raise ValueError(f"{label} requires key {key!r}")
    if not isinstance(spec["source"], str):
        raise TypeError(f"{label}['source'] must be a tile name")
    if "event" in spec and not isinstance(spec["event"], str):
        raise TypeError(f"{label}['event'] must be an event name")
    count = spec["count"]
    if not (callable(count) or isinstance(count, int | VarSpec | ExprSpec | ScalarSpec)):
        raise TypeError(f"{label}['count'] must be an int, ExprLike, or callable")
    indices = spec["indices"]
    if not callable(indices):
        if not isinstance(indices, tuple | list) or len(indices) != 3:
            raise TypeError(f"{label}['indices'] must be a callable or a 3-tuple")
        for entry in indices:
            if not isinstance(entry, int | VarSpec | ExprSpec | ScalarSpec):
                raise TypeError(f"{label}['indices'] entries must be int or ExprLike")
    scope = spec.get("pre_scope")
    if scope is not None:
        if (
            not isinstance(scope, tuple)
            or len(scope) != 2
            or scope[0] not in _PUSH_SCOPES
            or not isinstance(scope[1], int)
            or scope[1] < 0
        ):
            raise ValueError(f"{label}['pre_scope'] must be a (scope, scope_id) tuple")
    push_level = spec.get("push_level")
    if push_level is not None and push_level not in _PUSH_SCOPES:
        raise ValueError(f"{label}['push_level'] must be one of {_PUSH_SCOPES}")
    return dict(spec)


def _check_push_scopes(source: TileSpec, pre_scope, push_level) -> None:
    if _PUSH_SCOPE_ORDER[push_level] > _PUSH_SCOPE_ORDER[pre_scope[0]]:
        raise ValueError(
            f"tile {source.name!r} cannot push at {push_level!r} from "
            f"pre-notify scope {pre_scope[0]!r}"
        )


def _synthesize_dispatch_rule(
    tile: TileSpec,
    kernel: KernelSpec,
    event_producers,
    scheduled,
    hatch: dict[str, Any] | None,
    ancestors,
    tensor_producers,
) -> _DispatchRule:
    """Derive (or read out) the push rule for one dynamically dispatched tile."""

    by_name = {other.name: other for other in kernel.tiles}
    hint = f"declare tile.attrs[{_ESCAPE_HATCH_KEY!r}] to override dispatch synthesis"
    if hatch is not None:
        source = by_name.get(hatch["source"])
        if source is None:
            raise ValueError(f"tile {tile.name!r} escape hatch names unknown source tile")
        if not tile.waits:
            raise ValueError(f"tile {tile.name!r} escape hatch requires a waited event")
        event_name = hatch.get("event") or tile.waits[0][0].name
        event = kernel.events.get(event_name)
        if event is None:
            raise ValueError(f"tile {tile.name!r} escape hatch names unknown event")
        if not any(waited is event for waited, _ in tile.waits):
            raise ValueError(
                f"tile {tile.name!r} escape hatch event {event_name!r} is not waited by the tile"
            )
        coord_maps = [coord_map for notified, coord_map in source.notifies if notified is event]
        if len(coord_maps) != 1:
            raise ValueError(
                f"tile {tile.name!r} escape hatch source {source.name!r} must notify "
                f"event {event_name!r} exactly once"
            )
        pre_scope = hatch.get("pre_scope", source.impl.notify_scope)
        push_level = hatch.get("push_level", pre_scope[0])
        _check_push_scopes(source, pre_scope, push_level)
        return _DispatchRule(
            source,
            tile,
            event,
            coord_maps[0],
            (),
            (),
            (),
            push_level,
            pre_scope,
            custom_count=hatch["count"],
            custom_indices=hatch["indices"],
        )

    if len(tile.waits) != 1:
        raise ValueError(
            f"tile {tile.name!r} has {len(tile.waits)} waits; dynamic dispatch "
            f"supports exactly one incoming event edge per tile; {hint}"
        )
    event, coord_map_t = tile.waits[0]
    producers = event_producers.get(id(event), [])
    if len(producers) != 1:
        raise ValueError(
            f"event {event.name!r} waited by tile {tile.name!r} has "
            f"{len(producers)} producers; dynamic dispatch needs exactly one; {hint}"
        )
    source = producers[0]
    coord_maps = [coord_map for notified, coord_map in source.notifies if notified is event]
    if len(coord_maps) != 1:
        raise ValueError(
            f"tile {source.name!r} notifies event {event.name!r} "
            f"{len(coord_maps)} times; dynamic dispatch supports exactly one; {hint}"
        )
    coord_s = _probe_coord_map(coord_maps[0], f"tile {source.name!r} notify")
    coord_t = _probe_coord_map(coord_map_t, f"tile {tile.name!r} wait")
    if len(coord_s) != len(coord_t):
        raise ValueError(f"event {event.name!r}: producer/consumer coord ranks differ; {hint}")
    pinned = []
    for (s_kind, s_value), (t_kind, t_value) in zip(coord_s, coord_t):
        if s_kind == "const" and t_kind == "const":
            if s_value != t_value:
                raise ValueError(
                    f"event {event.name!r}: producer coord constant {s_value} does not "
                    f"match consumer constant {t_value}; {hint}"
                )
        elif s_kind == "const" and t_kind == "axis":
            pinned.append((t_value, ("const", s_value)))
        elif s_kind == "axis" and t_kind == "axis":
            pinned.append((t_value, ("axis", s_value)))
        else:
            raise ValueError(
                f"event {event.name!r}: consumer coord pins a constant where the "
                f"producer varies a tile axis; {hint}"
            )
    pinned_axes = {axis for axis, _ in pinned}
    free_axes = tuple(axis for axis in range(3) if axis not in pinned_axes)
    free_extents = tuple(scheduled[id(tile)][axis] for axis in free_axes)
    for scalar in expr_scalars(free_extents):
        scalar_producers = tensor_producers.get(id(scalar.source[0]), [])
        if len(scalar_producers) > 1:
            raise ValueError(
                f"scalar {scalar.name!r} source tensor has multiple producers; "
                "cannot prove push-time availability"
            )
        if (
            scalar_producers
            and scalar_producers[0] is not source
            and id(scalar_producers[0]) not in ancestors[id(source)]
        ):
            raise ValueError(
                f"push count for tile {tile.name!r} reads scalar {scalar.name!r} "
                f"written by tile {scalar_producers[0].name!r}, which is not "
                f"upstream of pusher {source.name!r}; {hint}"
            )
    pre_scope = source.impl.notify_scope
    push_level = pre_scope[0]
    _check_push_scopes(source, pre_scope, push_level)
    return _DispatchRule(
        source,
        tile,
        event,
        coord_maps[0],
        tuple(pinned),
        free_axes,
        free_extents,
        push_level,
        pre_scope,
    )


def _synthesize_drain(
    kernel: KernelSpec,
    terminal: TileSpec,
    scheduled,
    rules: dict[int, _DispatchRule],
    tensor_producers,
) -> _DrainPlan:
    """Synthesize the terminal tile's drain event and its initialization."""

    del kernel
    extents = scheduled[id(terminal)]
    name = f"__drain_{terminal.name}__"
    scalars = expr_scalars(extents)
    if not scalars and not expr_vars(extents):
        static_count = 1
        for extent in extents:
            static_count *= extent
        return _DrainPlan(name, terminal, static_count, extents, None)
    if not scalars:
        # Var-dependent terminal grid: the INIT seed task lowers the count
        # against the kernel's var parameters.
        return _DrainPlan(name, terminal, None, extents, None)
    producers: set[int] = set()
    producer_tiles: list[TileSpec] = []
    for scalar in scalars:
        scalar_producers = tensor_producers.get(id(scalar.source[0]), [])
        if len(scalar_producers) > 1:
            raise ValueError(
                f"scalar {scalar.name!r} source tensor has multiple producers; "
                "cannot place the drain event initialization"
            )
        for producer in scalar_producers:
            if id(producer) not in producers:
                producers.add(id(producer))
                producer_tiles.append(producer)
    if len(producer_tiles) == 1:
        writer = producer_tiles[0]
    else:
        # Host-planted scalar(s): the terminal tile's pusher initializes the
        # drain event after its run (idempotent across its tasks).
        pusher_rule = next((rule for rule in rules.values() if rule.target is terminal), None)
        if pusher_rule is None:
            raise ValueError(
                f"terminal tile {terminal.name!r} has a runtime-scalar grid but no "
                "pusher to initialize its drain event"
            )
        writer = pusher_rule.source
    return _DrainPlan(name, terminal, None, extents, writer)


def _validate_dynamic_capacity(
    kernel: KernelSpec, scheduled, entry_tiles, event_layouts, hardware: HardwareConfig
) -> None:
    """Bound the dynamic queue footprint and the packed task fields."""

    packing = TaskPacking()
    limits = (packing.max_m_idx, packing.max_n_idx, packing.max_k_idx)
    total = len(event_layouts) + hardware.sm_count  # INIT seeds + END tasks
    for tile in kernel.tiles:
        volume = 1
        for axis, extent in enumerate(scheduled[id(tile)]):
            try:
                bounds = expr_bounds(extent, require_bounded=True)
            except (TypeError, ValueError) as err:
                raise type(err)(
                    f"tile {tile.name!r} tile_num axis {axis}: {err}; the dynamic "
                    "queue capacity proof needs bounded extents"
                ) from err
            if bounds[0] <= 0:
                raise ValueError(f"tile {tile.name!r} tile_num axis {axis} is not positive")
            if bounds[1] > limits[axis]:
                raise ValueError(
                    f"tile {tile.name!r} tile_num axis {axis} upper bound {bounds[1]} "
                    f"exceeds the packed-task limit {limits[axis]}"
                )
            volume *= bounds[1]
        total += volume
    del entry_tiles
    if total > DynamicTileScheduler.MAX_TASKS:
        raise ValueError(
            f"dynamic queue upper bound {total} exceeds the MPMC capacity "
            f"{DynamicTileScheduler.MAX_TASKS}"
        )


def _prepare_dynamic_plan(
    kernel: KernelSpec, options: LoweringOptions, hardware: HardwareConfig
) -> _DynamicPlan:
    """Derive the dynamic lowering plan: bindings, layouts, dispatch rules."""

    attrs = dict(options.attrs)

    used_names: set[str] = set()
    var_bindings = tuple(
        VarBinding(var, _sanitize_identifier(var.name, used_names, "value"))
        for var in kernel.vars.values()
    )
    used_names.clear()
    tensor_bindings = tuple(
        TensorBinding(tensor, _sanitize_identifier(tensor.name, used_names, "tensor"))
        for tensor in kernel.tensors.values()
    )

    if not kernel.tiles:
        raise ValueError("dynamic scheduling requires at least one tile")
    if len(kernel.tiles) > INIT_EVENT_JOB_ID:
        raise ValueError(
            f"dynamic kernels support at most {INIT_EVENT_JOB_ID} tiles "
            f"(job ids 0..{INIT_EVENT_JOB_ID - 1}), got {len(kernel.tiles)}"
        )
    tile_plans = tuple(TileLoweringPlan(tile, job_id) for job_id, tile in enumerate(kernel.tiles))
    _validate_dynamic_event_encoding(kernel, options)
    _validate_dynamic_acyclic(kernel)
    coalescing = _parse_coalescing(kernel, attrs)

    entry_tiles = tuple(tile for tile in kernel.tiles if not tile.waits)
    terminal_tiles = [tile for tile in kernel.tiles if not tile.notifies]
    if len(terminal_tiles) != 1:
        names = [tile.name for tile in terminal_tiles]
        raise ValueError(
            "dynamic scheduling requires exactly one terminal tile (a tile "
            f"without notifies); found {len(terminal_tiles)}: {names}"
        )
    terminal = terminal_tiles[0]

    scheduled: dict[int, tuple] = {}
    for tile in kernel.tiles:
        factor = coalescing.get(tile.name, 1)
        extents = list(tile.tile_num)
        if factor != 1:
            if not tile.waits:
                raise ValueError(
                    f"entry tile {tile.name!r} cannot be coalesced (it is host-seeded)"
                )
            n_extent = extents[1]
            if not isinstance(n_extent, int) or isinstance(n_extent, bool):
                raise ValueError(
                    f"tile {tile.name!r}: tile_coalescing requires a statically "
                    f"known n extent, got {n_extent!r}"
                )
            if n_extent % factor:
                raise ValueError(
                    f"tile {tile.name!r}: n extent {n_extent} is not divisible by "
                    f"the coalescing factor {factor}"
                )
            extents[1] = n_extent // factor
        scheduled[id(tile)] = tuple(extents)
    for tile in entry_tiles:
        if expr_scalars(scheduled[id(tile)]):
            raise ValueError(
                f"entry tile {tile.name!r} has a runtime-scalar tile_num; a "
                "scalar-dependent tile must wait on an event to be dynamically "
                "dispatched"
            )

    event_producers = _event_tile_producers(kernel)
    tensor_producers = _tensor_producers(kernel)
    ancestors = _event_ancestors(kernel)
    hatches = {
        tile.name: _parse_escape_hatch(tile)
        for tile in kernel.tiles
        if _ESCAPE_HATCH_KEY in tile.attrs
    }

    rules: dict[int, _DispatchRule] = {}
    for tile in kernel.tiles:
        if not tile.waits:
            continue
        rule = _synthesize_dispatch_rule(
            tile,
            kernel,
            event_producers,
            scheduled,
            hatches.get(tile.name),
            ancestors,
            tensor_producers,
        )
        if id(rule.source) in rules:
            raise ValueError(
                f"tile {rule.source.name!r} pushes more than one tile; dynamic "
                "dispatch supports one outgoing rule per tile"
            )
        rules[id(rule.source)] = rule
    for tile in kernel.tiles:
        if tile is not terminal and id(tile) not in rules:
            raise ValueError(
                f"tile {tile.name!r} is not the pusher of any dispatched tile; "
                "fan-out (one event consumed by several tiles) is not supported "
                "by dynamic dispatch"
            )

    drain = _synthesize_drain(kernel, terminal, scheduled, rules, tensor_producers)

    event_layouts: list[_DynamicEventLayout] = []
    offset = 0
    for event in kernel.events.values():
        shape = _upper_bound_shape_extents(event.shape, f"event {event.name!r} shape")
        event_layouts.append(_DynamicEventLayout(event, event.name, shape, offset, False))
        offset += upper_bound_shape_product(
            event.shape, f"event {event.name!r} shape", require_bounded=True
        )
    event_layouts.append(_DynamicEventLayout(None, drain.name, (1,), offset, True))
    offset += 1

    _validate_dynamic_capacity(kernel, scheduled, entry_tiles, event_layouts, hardware)

    return _DynamicPlan(
        kernel=kernel,
        options=options,
        attrs=attrs,
        var_bindings=var_bindings,
        tensor_bindings=tensor_bindings,
        event_layouts=tuple(event_layouts),
        drain=drain,
        tile_plans=tile_plans,
        dispatch_rules=rules,
        terminal=terminal,
        entry_tiles=entry_tiles,
        scheduled=scheduled,
        coalescing=coalescing,
        event_workspace_size=offset,
    )


class _RuntimeDynamicKernelBuilder(_RuntimeKernelBuilder):
    """Emit one dynamic (MPMC) persistent kernel from a spec."""

    is_dynamic = True

    def __init__(self, plan: _DynamicPlan, hardware: HardwareConfig):
        super().__init__(plan, hardware)
        self.drain_sem = None
        self._job_ids = {id(tile_plan.tile): tile_plan.job_id for tile_plan in plan.tile_plans}

    def _emit_special_args(self) -> None:
        if self.plan.event_workspace_size:
            self.event_workspace = T.arg(
                "event_workspace",
                T.Buffer((self.plan.event_workspace_size,), "int32"),
            )
        self.queue_tasks = T.arg("exec_task", T.Buffer((DynamicTileScheduler.MAX_TASKS,), "int32"))
        self.queue_head = T.arg("exec_head", T.Buffer((1,), "int32"))
        self.queue_tail = T.arg("exec_tail", T.Buffer((1,), "int32"))
        self._emit_profiler_arg()

    def _drain_f_init(self, drain: _DrainPlan):
        if drain.static_count is not None:
            return f_init_const(drain.static_count)
        if drain.writer is None:

            def f_init(*_coord):
                value = 1
                for extent in drain.count_extents:
                    value = value * self._lower_expr(extent, "drain event count")
                return value

            return f_init
        return None

    def _emit_events(self) -> None:
        plan = self.plan
        if not plan.event_layouts:
            return
        wrapper = self.wrapper
        for layout in plan.event_layouts:
            if layout.is_drain:
                f_init = self._drain_f_init(plan.drain)
            elif callable(layout.event.init_count):

                def f_init(*coord, event=layout.event):
                    return event.init_count(tuple(coord))

            else:
                f_init = f_init_const(layout.event.init_count)
            semaphore = wrapper.add_etensor(
                DynamicSemaphore, self.event_workspace, list(layout.shape), f_init
            )
            # Bind the configured hardware, as in the static path.
            semaphore.hardware = self.hardware
            if layout.is_drain:
                self.drain_sem = semaphore
            else:
                self.event_sems[id(layout.event)] = semaphore
        wrapper.set_events_complete(True, DynamicSemaphore, self.event_workspace)
        if wrapper.etensor_workspace_offset != plan.event_workspace_size:
            raise ValueError(
                "dynamic event workspace layout diverged from its plan: "
                f"{wrapper.etensor_workspace_offset} != {plan.event_workspace_size}"
            )

    def _init_scheduler(self) -> None:
        self.wrapper.init_tile_scheduler(
            True,
            DynamicTileScheduler,
            self.queue_tasks,
            self.queue_head,
            self.queue_tail,
            self.wrapper.smem_manager,
            self.wrapper.profiler,
            self.plan.attrs.get("debug_scheduler", False),
        )

    def _dispatch_extra_entries(self) -> list[tuple[int, Any]]:
        if self.plan.event_layouts:
            return [(INIT_EVENT_JOB_ID, "init_event")]
        return []

    def _run_repeat_count(self, tile) -> int:
        return self.plan.coalescing.get(tile.name, 1)

    def _emit_tile_pre_steps(self, tile, indices) -> None:
        rule = self.plan.dispatch_rules.get(id(tile))
        if rule is not None:
            self._emit_pre_notify_and_push(rule, indices)
        if tile is self.plan.terminal:
            self._emit_drain_push(tile)

    def _emit_tile_post_run_steps(self, tile, indices) -> None:
        drain = self.plan.drain
        if drain is not None and drain.writer is tile:
            T.evaluate(T.cuda.cta_sync())
            tid = T.thread_id([self.hardware.num_threads])
            with T.If(tid == 0):
                with T.Then():
                    value = 1
                    for extent in drain.count_extents:
                        value = value * self._lower_expr(extent, "drain event count")
                    T.buffer_store(self.drain_sem.sem, (SemaphoreBase.base + 1) * value, [0])

    def _emit_pre_notify_and_push(self, rule: _DispatchRule, indices) -> None:
        scheduler = self.wrapper.tile_scheduler
        semaphore = self.event_sems[id(rule.event)]
        coord = self._lower_event_coord(
            rule.coord_map, indices, f"tile {rule.source.name!r} pre-notify"
        )

        def notify_func(_notify_idx):
            return (1, -1, *coord)

        push_fn = self._make_push_fn(rule, indices)

        def trigger_fn(_trigger_idx):
            return push_fn

        scheduler.pre_notify_and_push(
            semaphore,
            notify_func,
            trigger_fn,
            rule.push_level,
            rule.pre_scope[0],
            scope_id=rule.pre_scope[1],
        )

    def _make_push_fn(self, rule: _DispatchRule, indices):
        target_job = self._job_ids[id(rule.target)]
        if rule.custom_indices is not None:
            count = rule.custom_count
            if callable(count):
                count = count(*indices)
            if isinstance(count, int | VarSpec | ExprSpec | ScalarSpec):
                count = self._lower_expr(count, f"tile {rule.target.name!r} push count")

            def push_fn(push_idx):
                mapped = rule.custom_indices(push_idx, *indices)
                if not isinstance(mapped, tuple | list) or len(mapped) != 3:
                    raise TypeError("escape-hatch indices must return a 3-tuple")
                mapped = tuple(
                    self._lower_expr(entry, "push indices")
                    if isinstance(entry, int | VarSpec | ExprSpec | ScalarSpec)
                    else entry
                    for entry in mapped
                )
                return (target_job, count, *mapped)

            return push_fn

        free_extents = [
            self._lower_expr(extent, f"tile {rule.target.name!r} push count")
            for extent in rule.free_extents
        ]
        count = 1
        for extent in free_extents:
            count = count * extent

        def push_fn(push_idx):
            values: dict[int, Any] = {}
            remaining = push_idx
            for pos, axis in enumerate(rule.free_axes):
                if pos + 1 < len(rule.free_axes):
                    later = 1
                    for extent in free_extents[pos + 1 :]:
                        later = later * extent
                    values[axis] = remaining // later
                    remaining = remaining % later
                else:
                    values[axis] = remaining
            for axis, (kind, payload) in rule.pinned:
                values[axis] = indices[payload] if kind == "axis" else payload
            return (target_job, count, values.get(0, 0), values.get(1, 0), values.get(2, 0))

        return push_fn

    def _emit_drain_push(self, tile) -> None:
        scheduler = self.wrapper.tile_scheduler
        scope, scope_id = tile.impl.notify_scope

        def notify_func(_notify_idx):
            return (1, -1, 0)

        def push_fn(_push_idx):
            return (DEFAULT_END_JOB_ID, self.hardware.sm_count, 0, 0, 0)

        def trigger_fn(_trigger_idx):
            return push_fn

        scheduler.pre_notify_and_push(
            self.drain_sem, notify_func, trigger_fn, scope, scope, scope_id=scope_id
        )


def derive_dynamic_seed_tasks(
    plan: _DynamicPlan, var_values: dict[str, int] | None = None
) -> list[tuple[int, int, int, int]]:
    """Enumerate the dynamic seed tasks: event-init tasks plus entry tiles."""

    env = _var_env(plan.kernel, var_values)
    packing = TaskPacking()
    limits = (packing.max_m_idx, packing.max_n_idx, packing.max_k_idx)
    tasks = []
    for etensor_idx in range(len(plan.event_layouts)):
        tasks.append((etensor_idx, 0, 0, INIT_EVENT_JOB_ID))
    job_ids = {id(tile_plan.tile): tile_plan.job_id for tile_plan in plan.tile_plans}
    for tile in plan.entry_tiles:
        extents = []
        for axis, extent in enumerate(plan.scheduled[id(tile)]):
            value = eval_expr_like(extent, env)
            if value is None:
                raise ValueError(
                    f"dynamic seed derivation for tile {tile.name!r} needs a "
                    "concrete value for every symbolic tile_num variable; pass "
                    "var_values"
                )
            if value > limits[axis]:
                raise ValueError(
                    f"entry tile {tile.name!r} axis {axis} extent {value} exceeds "
                    f"the packed-task limit {limits[axis]}"
                )
            extents.append(value)
        for m_idx in range(extents[0]):
            for n_idx in range(extents[1]):
                for k_idx in range(extents[2]):
                    tasks.append((m_idx, n_idx, k_idx, job_ids[id(tile)]))
    return tasks


def build_dynamic_queues(
    plan: _DynamicPlan, var_values: dict[str, int] | None = None
) -> tuple[list[tuple[int, int, int, int]], MPMCQueueHost]:
    """Build the host-side MPMC seed queue arrays for a dynamic plan."""

    seeds = derive_dynamic_seed_tasks(plan, var_values)
    host = MPMCQueueHost(DynamicTileScheduler.MAX_TASKS)
    for m_idx, n_idx, k_idx, job_id in seeds:
        host.enqueue(job_id, m_idx, n_idx, k_idx)
    return seeds, host


def _drain_event_infos(plan: _DynamicPlan) -> tuple[DrainEventInfo, ...]:
    layout = next(layout for layout in plan.event_layouts if layout.is_drain)
    return (
        DrainEventInfo(
            name=plan.drain.name,
            workspace_offset=layout.workspace_offset,
            static_count=plan.drain.static_count,
            runtime_initialized=plan.drain.writer is not None,
        ),
    )


def build_runtime_kernel(
    kernel: KernelSpec,
    options: LoweringOptions | None = None,
    var_values: dict[str, int] | None = None,
) -> RuntimeKernelBuild:
    """Lower a spec with the runtime builder and derive its host queue.

    ``var_values`` maps symbolic ``VarSpec`` names to concrete integers; it is
    required only when some seeded ``tile_num`` is symbolic (the device kernel
    itself is symbolic-safe, but the host queue must enumerate concrete
    grids).
    """

    resolved = _resolve_options(options)
    func, plan, hardware = _emit_runtime_func(kernel, resolved)
    module = IRModule({kernel.name: func})
    if resolved.scheduler == "dynamic":
        seeds, host = build_dynamic_queues(plan, var_values)
        return RuntimeKernelBuild(
            module=module,
            scheduler="dynamic",
            exec_queue=None,
            queue_tasks=host.tasks,
            queue_head=host.head,
            queue_tail=host.tail,
            central_tasks=tuple(seeds),
            event_workspace_size=plan.event_workspace_size,
            sm_count=hardware.sm_count,
            max_tasks=DynamicTileScheduler.MAX_TASKS,
            end_task_type=DEFAULT_END_JOB_ID,
            init_event_job_id=INIT_EVENT_JOB_ID,
            wait_event_init_job_id=WAIT_EVENT_INIT_JOB_ID,
            profiler_on=bool(plan.attrs.get("profiler", False)),
            drain_events=_drain_event_infos(plan),
        )
    central, queue = build_static_queues(plan, hardware, var_values)
    return RuntimeKernelBuild(
        module=module,
        scheduler="static",
        exec_queue=queue,
        queue_tasks=None,
        queue_head=None,
        queue_tail=None,
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
    "DrainEventInfo",
    "RuntimeKernelBuild",
    "build_dynamic_queues",
    "build_runtime_kernel",
    "build_static_queues",
    "derive_dynamic_seed_tasks",
    "derive_static_central_tasks",
    "emit_runtime_module",
]
