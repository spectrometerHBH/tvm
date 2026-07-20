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
"""Backend-private preparation for the default TIRX lowering backend."""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..dsl import EventSpec, ExprSpec, TensorSpec, VarSpec, expr_bounds
from .model import DeviceRegionPlan, ExecutionPlan, TileProgram, WaitStep
from .semantic import SemanticPlan

if TYPE_CHECKING:
    from .lower import LoweringOptions

INIT_EVENT_JOB_ID = 29
WAIT_EVENT_INIT_JOB_ID = 30
EVENT_INIT_COMPLETE_NAME = "__event_init_complete__"
DEFAULT_END_JOB_ID = 31
DEFAULT_MAX_TASKS = 128


@dataclass(frozen=True)
class VarBinding:
    """One semantic variable's final kernel parameter name."""

    var: VarSpec
    param_name: str


@dataclass(frozen=True)
class TensorBinding:
    """One semantic tensor's final kernel parameter name."""

    tensor: TensorSpec
    param_name: str


@dataclass(frozen=True)
class EventLayout:
    """Statically reserved event-workspace region."""

    event: EventSpec | None
    name: str
    shape: Any
    dtype: str
    workspace_offset: int
    reserved_size: int


@dataclass(frozen=True)
class TileLoweringPlan:
    """Backend job binding for one explicit physical tile program."""

    program: TileProgram
    job_id: int

    @property
    def tile(self):
        return self.program.tile


@dataclass(frozen=True)
class TaskPhase:
    """One grid inserted into the static queue initializer."""

    kind: str
    job_id: int
    tile_num: Any
    label: str


@dataclass(frozen=True)
class StaticSchedulePlan:
    """Static queue phases and their fixed physical capacity."""

    phases: tuple[TaskPhase, ...]
    sm_count: int
    max_tasks: int
    end_job_id: int


@dataclass(frozen=True)
class TIRXLoweringPlan:
    """Prepared lower-private state consumed only by the default TIRX backend."""

    semantic: SemanticPlan
    execution: ExecutionPlan
    options: LoweringOptions
    region: DeviceRegionPlan
    attrs: dict[str, Any]
    var_bindings: tuple[VarBinding, ...]
    tensor_bindings: tuple[TensorBinding, ...]
    event_layouts: tuple[EventLayout, ...]
    event_init_complete_layout: EventLayout | None
    tile_plans: tuple[TileLoweringPlan, ...]
    static_schedule: StaticSchedulePlan | None

    @property
    def kernel(self):
        return self.semantic.kernel

    @property
    def event_workspace_size(self) -> int:
        layout = self.event_init_complete_layout
        return 0 if layout is None else layout.workspace_offset + layout.reserved_size

    def var_binding(self, var: VarSpec) -> VarBinding:
        for binding in self.var_bindings:
            if binding.var is var:
                return binding
        raise KeyError(var.name)

    def tensor_binding(self, tensor: TensorSpec) -> TensorBinding:
        base = tensor.base_tensor
        for binding in self.tensor_bindings:
            if binding.tensor is base:
                return binding
        raise KeyError(base.name)

    def event_layout(self, event: EventSpec) -> EventLayout:
        for layout in self.event_layouts:
            if layout.event is event:
                return layout
        raise KeyError(event.name)


def prepare_tirx_lowering_plan(
    execution: ExecutionPlan, options: LoweringOptions
) -> TIRXLoweringPlan:
    """Prepare and validate the default backend's private lowering state."""

    semantic = execution.resolved_semantic_plan()
    execution.validate()
    if len(execution.device_regions) != 1 or execution.host_regions:
        raise ValueError("the default backend requires exactly one device region")
    region = execution.device_regions[0]
    attrs = {**region.attrs, **options.attrs}

    used_names: set[str] = set()
    var_bindings = tuple(
        VarBinding(var, _sanitize_identifier(var.name, used_names, "value"))
        for var in semantic.vars
    )
    used_names.clear()
    tensor_bindings = tuple(
        TensorBinding(tensor, _sanitize_identifier(tensor.name, used_names, "tensor"))
        for tensor in semantic.tensors
    )

    event_layouts = []
    offset = 0
    for event in semantic.events:
        reserved_size = upper_bound_shape_product(
            event.shape,
            f"event {event.name!r} shape",
            require_bounded=True,
        )
        event_layouts.append(
            EventLayout(
                event,
                event.name,
                event.shape,
                event.dtype,
                offset,
                reserved_size,
            )
        )
        offset += reserved_size
    complete_layout = None
    if event_layouts:
        complete_layout = EventLayout(
            None,
            EVENT_INIT_COMPLETE_NAME,
            (1,),
            "int32",
            offset,
            1,
        )

    tile_plans = tuple(
        TileLoweringPlan(program, job_id) for job_id, program in enumerate(region.tile_programs)
    )
    static_schedule = None
    if options.schedule == "static":
        static_schedule = _build_static_schedule(
            tile_plans,
            len(event_layouts),
            attrs,
        )

    plan = TIRXLoweringPlan(
        semantic=semantic,
        execution=execution,
        options=options,
        region=region,
        attrs=attrs,
        var_bindings=var_bindings,
        tensor_bindings=tensor_bindings,
        event_layouts=tuple(event_layouts),
        event_init_complete_layout=complete_layout,
        tile_plans=tile_plans,
        static_schedule=static_schedule,
    )
    return validate_tirx_lowering_plan(plan)


def validate_tirx_lowering_plan(plan: TIRXLoweringPlan) -> TIRXLoweringPlan:
    """Validate event layout, task encodings, and static queue capacity."""

    _validate_event_layout(plan)
    _validate_job_ids(plan)
    if plan.options.schedule == "static":
        _validate_static_schedule(plan)
    elif plan.static_schedule is not None:
        raise ValueError("non-static lowering cannot carry a static schedule")
    return plan


def _build_static_schedule(tile_plans, event_count: int, attrs) -> StaticSchedulePlan:
    sm_count = _positive_int(attrs.get("sm_count", 1), "sm_count")
    end_job_id = _nonnegative_int(attrs.get("end_job_id", DEFAULT_END_JOB_ID), "end_job_id")
    phases = []
    if event_count:
        phases.append(TaskPhase("grid", INIT_EVENT_JOB_ID, (event_count + 1, 1, 1), "init_events"))
    entry = [tile for tile in tile_plans if not _program_waits(tile.program)]
    waiting = [tile for tile in tile_plans if _program_waits(tile.program)]
    phases.extend(_tile_phase(tile) for tile in entry)
    if event_count:
        phases.append(
            TaskPhase("grid", WAIT_EVENT_INIT_JOB_ID, (sm_count, 1, 1), "wait_event_init")
        )
    phases.extend(_tile_phase(tile) for tile in waiting)
    phases.append(TaskPhase("grid", end_job_id, (sm_count, 1, 1), "end"))
    required_tasks = sum(
        upper_bound_shape_product(
            phase.tile_num,
            f"static phase {phase.label!r}",
            require_bounded=True,
        )
        for phase in phases
    )
    required_per_sm = (required_tasks + sm_count - 1) // sm_count
    max_tasks = _positive_int(
        attrs.get("max_tasks", max(DEFAULT_MAX_TASKS, required_per_sm)),
        "max_tasks",
    )
    return StaticSchedulePlan(tuple(phases), sm_count, max_tasks, end_job_id)


def _program_waits(program: TileProgram) -> bool:
    return any(isinstance(step, WaitStep) for step in program.steps)


def _tile_phase(tile_plan: TileLoweringPlan) -> TaskPhase:
    return TaskPhase("grid", tile_plan.job_id, tile_plan.tile.tile_num, tile_plan.tile.name)


def _validate_event_layout(plan: TIRXLoweringPlan) -> None:
    for layout in plan.event_layouts:
        if layout.dtype != "int32":
            raise ValueError(
                f"default TIRX event storage requires int32, got {layout.dtype!r} "
                f"for event {layout.name!r}"
            )
    regions = [
        (layout.workspace_offset, layout.workspace_offset + layout.reserved_size, layout.name)
        for layout in plan.event_layouts
    ]
    if plan.event_init_complete_layout is not None:
        layout = plan.event_init_complete_layout
        regions.append(
            (layout.workspace_offset, layout.workspace_offset + layout.reserved_size, layout.name)
        )
    for index, (begin, end, name) in enumerate(regions):
        if begin < 0 or end <= begin:
            raise ValueError(f"event workspace region {name!r} is invalid")
        for other_begin, other_end, other_name in regions[index + 1 :]:
            if begin < other_end and other_begin < end:
                raise ValueError(f"event workspace regions {name!r} and {other_name!r} overlap")


def _validate_job_ids(plan: TIRXLoweringPlan) -> None:
    reserved = {INIT_EVENT_JOB_ID, WAIT_EVENT_INIT_JOB_ID}
    if plan.static_schedule is not None:
        reserved.add(plan.static_schedule.end_job_id)
    seen = set()
    semantic_tile_ids = {id(tile) for tile in plan.semantic.tiles}
    for tile_plan in plan.tile_plans:
        if tile_plan.job_id in reserved:
            raise ValueError(
                f"tile {tile_plan.tile.name!r} job id {tile_plan.job_id} collides "
                "with a reserved job id"
            )
        if tile_plan.job_id < 0 or tile_plan.job_id >= 32:
            raise ValueError("static task job ids must fit the five-bit queue field")
        if tile_plan.job_id in seen:
            raise ValueError("lowering plan contains duplicate tile job ids")
        if id(tile_plan.tile) not in semantic_tile_ids:
            raise ValueError("lowering plan references a tile outside its semantic plan")
        seen.add(tile_plan.job_id)


def _validate_static_schedule(plan: TIRXLoweringPlan) -> None:
    schedule = plan.static_schedule
    if schedule is None:
        raise ValueError("static lowering requires a static schedule plan")
    if schedule.end_job_id >= 32:
        raise ValueError("end job id must fit the five-bit queue field")
    if schedule.end_job_id in {INIT_EVENT_JOB_ID, WAIT_EVENT_INIT_JOB_ID}:
        raise ValueError("end job id collides with an event initialization job")
    phase_job_ids = {phase.job_id for phase in schedule.phases}
    for tile_plan in plan.tile_plans:
        if tile_plan.job_id not in phase_job_ids:
            raise ValueError(f"tile {tile_plan.tile.name!r} is missing from the static schedule")

    total_tasks = 0
    for phase in schedule.phases:
        _validate_packed_task_shape(phase.tile_num, phase.label)
        total_tasks += upper_bound_shape_product(
            phase.tile_num,
            f"static phase {phase.label!r}",
            require_bounded=True,
        )
    tasks_per_sm = (total_tasks + schedule.sm_count - 1) // schedule.sm_count
    if tasks_per_sm > schedule.max_tasks:
        raise ValueError(
            f"static schedule needs {tasks_per_sm} queue entries per SM, "
            f"exceeding max_tasks={schedule.max_tasks}"
        )


def _validate_packed_task_shape(shape, label: str) -> None:
    values = tuple(shape)
    if len(values) != 3:
        raise ValueError(f"static phase {label!r} must have exactly three task axes")
    limits = (1 << 13, 1 << 10, 1 << 4)
    for axis, (extent, limit) in enumerate(zip(values, limits)):
        bounds = expr_bounds(extent, require_bounded=True)
        if bounds[0] <= 0:
            raise ValueError(f"static phase {label!r} has a non-positive axis {axis}")
        if bounds[1] > limit:
            raise ValueError(
                f"static phase {label!r} axis {axis} extent {bounds[1]} exceeds "
                f"the packed-task limit {limit}"
            )


def upper_bound_shape_product(shape, label: str, *, require_bounded: bool) -> int | None:
    """Return a shape's conservative upper-bound element count."""

    values = tuple(shape) if isinstance(shape, tuple | list) else (shape,)
    result = 1
    for extent in values:
        try:
            bounds = expr_bounds(extent, require_bounded=require_bounded)
        except (TypeError, ValueError) as err:
            raise type(err)(f"{label}: {err}") from err
        if bounds is None:
            return None
        if bounds[0] <= 0:
            raise ValueError(f"{label} extents must be positive")
        result *= bounds[1]
    return result


def lower_expr_like(value, var_values: dict[int, Any], label: str):
    """Lower a logical integer expression using bound TIRX variables."""

    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, VarSpec):
        if id(value) not in var_values:
            raise ValueError(f"{label} contains an unbound symbolic variable")
        return var_values[id(value)]
    if not isinstance(value, ExprSpec):
        raise TypeError(f"{label} must contain only int, VarSpec, or ExprSpec values")
    args = [lower_expr_like(arg, var_values, label) for arg in value.args]
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


def lower_shape(shape, var_values: dict[int, Any], label: str) -> tuple[Any, ...]:
    """Lower all logical extents in one shape."""

    values = tuple(shape) if isinstance(shape, tuple | list) else (shape,)
    return tuple(lower_expr_like(value, var_values, label) for value in values)


def _sanitize_identifier(name: str, used_names: set[str], prefix: str) -> str:
    candidate = re.sub(r"\W", "_", name)
    if not candidate or candidate[0].isdigit() or keyword.iskeyword(candidate):
        candidate = f"{prefix}_{candidate}"
    base = candidate
    suffix = 1
    while candidate in used_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _positive_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer")
    return value


__all__ = [
    "DEFAULT_END_JOB_ID",
    "DEFAULT_MAX_TASKS",
    "EVENT_INIT_COMPLETE_NAME",
    "INIT_EVENT_JOB_ID",
    "WAIT_EVENT_INIT_JOB_ID",
    "EventLayout",
    "StaticSchedulePlan",
    "TIRXLoweringPlan",
    "TaskPhase",
    "TensorBinding",
    "TileLoweringPlan",
    "VarBinding",
    "lower_expr_like",
    "lower_shape",
    "prepare_tirx_lowering_plan",
    "upper_bound_shape_product",
    "validate_tirx_lowering_plan",
]
