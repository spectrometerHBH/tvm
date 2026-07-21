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
"""Backend-private preparation for the default TIRX lowering backend.

These helpers are consumed only by ``transform.lower``.  They derive the
parameter bindings, event-workspace layout, job IDs, and static queue
schedule for one already-validated ``KernelSpec``; none of this is a public
contract.
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any

from ..dsl import EventSpec, ExprSpec, KernelSpec, TensorSpec, TileSpec, VarSpec, expr_bounds
from .scheduler import TIRXSemaphore
from .validate import _spec_view, _static_event_tile_adjacency

if TYPE_CHECKING:
    from .lower import LoweringOptions

INIT_EVENT_JOB_ID = 29
WAIT_EVENT_INIT_JOB_ID = 30
EVENT_INIT_COMPLETE_NAME = "__event_init_complete__"
DEFAULT_END_JOB_ID = 31
DEFAULT_MAX_TASKS = 128
_EVENT_INIT_COUNT_PROOF_LIMIT = 262_144
_MAX_EVENT_COUNTER_COUNT = ((1 << 31) - 1) // (TIRXSemaphore.base + 1)


@dataclass(frozen=True)
class VarBinding:
    """One logical variable's final kernel parameter name."""

    var: VarSpec
    param_name: str


@dataclass(frozen=True)
class TensorBinding:
    """One logical tensor's final kernel parameter name."""

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
    """Backend job binding for one logical tile."""

    tile: TileSpec
    job_id: int


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

    kernel: KernelSpec
    options: LoweringOptions
    attrs: dict[str, Any]
    var_bindings: tuple[VarBinding, ...]
    tensor_bindings: tuple[TensorBinding, ...]
    event_layouts: tuple[EventLayout, ...]
    event_init_complete_layout: EventLayout | None
    tile_plans: tuple[TileLoweringPlan, ...]
    static_schedule: StaticSchedulePlan | None

    @property
    def event_workspace_size(self) -> int:
        layout = self.event_init_complete_layout
        return 0 if layout is None else layout.workspace_offset + layout.reserved_size


def prepare_tirx_lowering_plan(kernel: KernelSpec, options: LoweringOptions) -> TIRXLoweringPlan:
    """Prepare and validate the default backend's private lowering state."""

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

    event_layouts = []
    offset = 0
    for event in kernel.events.values():
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

    tile_plans = tuple(TileLoweringPlan(tile, job_id) for job_id, tile in enumerate(kernel.tiles))
    static_schedule = None
    if options.schedule == "static":
        static_schedule = _build_static_schedule(
            tile_plans,
            len(event_layouts),
            attrs,
            _static_event_tile_adjacency(_spec_view(kernel)),
        )

    plan = TIRXLoweringPlan(
        kernel=kernel,
        options=options,
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

    _validate_event_counter_encoding(plan)
    _validate_event_layout(plan)
    _validate_job_ids(plan)
    if plan.options.schedule == "static":
        _validate_static_schedule(plan)
    elif plan.static_schedule is not None:
        raise ValueError("non-static lowering cannot carry a static schedule")
    return plan


def _validate_event_counter_encoding(plan: TIRXLoweringPlan) -> None:
    """Prove that every encoded semaphore count fits signed int32 storage."""

    for event in plan.kernel.events.values():
        if isinstance(event.init_count, int) and not isinstance(event.init_count, bool):
            if event.init_count > _MAX_EVENT_COUNTER_COUNT:
                raise ValueError(
                    f"event {event.name!r} init_count {event.init_count} exceeds the "
                    f"int32 semaphore encoding limit {_MAX_EVENT_COUNTER_COUNT}"
                )
            continue

        upper_extents = _upper_bound_shape_extents(
            event.shape,
            f"event {event.name!r} init_count domain",
        )
        point_count = 1
        for extent in upper_extents:
            point_count *= extent
        if point_count > _EVENT_INIT_COUNT_PROOF_LIMIT:
            raise ValueError(
                f"event {event.name!r} callable init_count spans {point_count} coordinates; "
                "the default backend cannot prove its int32 semaphore encoding"
            )
        for coord in product(*(range(extent) for extent in upper_extents)):
            try:
                first = event.init_count(coord)
                second = event.init_count(coord)
            except Exception as err:  # pylint: disable=broad-exception-caught
                raise ValueError(
                    f"event {event.name!r} init_count failed at coord {coord}"
                ) from err
            if first != second:
                raise ValueError(
                    f"event {event.name!r} init_count must be deterministic at coord {coord}"
                )
            if not isinstance(first, int) or isinstance(first, bool) or first <= 0:
                raise ValueError(
                    f"event {event.name!r} init_count at coord {coord} must be positive"
                )
            if first > _MAX_EVENT_COUNTER_COUNT:
                raise ValueError(
                    f"event {event.name!r} init_count {first} at coord {coord} exceeds the "
                    f"int32 semaphore encoding limit {_MAX_EVENT_COUNTER_COUNT}"
                )

    if plan.event_init_complete_layout is not None and plan.static_schedule is not None:
        complete_count = len(plan.event_layouts) + 1 + plan.static_schedule.sm_count
        if complete_count > _MAX_EVENT_COUNTER_COUNT:
            raise ValueError(
                f"event initialization completion count {complete_count} exceeds the "
                f"int32 semaphore encoding limit {_MAX_EVENT_COUNTER_COUNT}"
            )


def _upper_bound_shape_extents(shape, label: str) -> tuple[int, ...]:
    values = tuple(shape) if isinstance(shape, tuple | list) else (shape,)
    result = []
    for extent in values:
        try:
            bounds = expr_bounds(extent, require_bounded=True)
        except (TypeError, ValueError) as err:
            raise type(err)(f"{label}: {err}") from err
        if bounds[0] <= 0:
            raise ValueError(f"{label} extents must be positive")
        result.append(bounds[1])
    return tuple(result)


def _build_static_schedule(
    tile_plans, event_count: int, attrs, tile_adjacency
) -> StaticSchedulePlan:
    sm_count = _positive_int(attrs.get("sm_count", 1), "sm_count")
    end_job_id = _nonnegative_int(attrs.get("end_job_id", DEFAULT_END_JOB_ID), "end_job_id")
    phases = []
    if event_count:
        phases.append(TaskPhase("grid", INIT_EVENT_JOB_ID, (event_count + 1, 1, 1), "init_events"))
    entry = [tile_plan for tile_plan in tile_plans if not tile_plan.tile.waits]
    waiting = [tile_plan for tile_plan in tile_plans if tile_plan.tile.waits]
    waiting = _stable_topological_tile_plans(waiting, tile_adjacency)
    phases.extend(_tile_phase(tile_plan) for tile_plan in entry)
    if event_count:
        phases.append(
            TaskPhase("grid", WAIT_EVENT_INIT_JOB_ID, (sm_count, 1, 1), "wait_event_init")
        )
    phases.extend(_tile_phase(tile_plan) for tile_plan in waiting)
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


def _stable_topological_tile_plans(tile_plans, tile_adjacency):
    """Order the static waiting-tile DAG without disturbing stable ties."""

    if len(tile_plans) < 2:
        return tile_plans
    by_tile_id = {id(tile_plan.tile): tile_plan for tile_plan in tile_plans}
    order = {id(tile_plan.tile): index for index, tile_plan in enumerate(tile_plans)}
    outgoing = {tile_id: set() for tile_id in by_tile_id}
    incoming = {tile_id: 0 for tile_id in by_tile_id}
    for producer_id, consumer_ids in tile_adjacency.items():
        if producer_id not in by_tile_id:
            continue
        for consumer_id in consumer_ids:
            if consumer_id not in by_tile_id or consumer_id in outgoing[producer_id]:
                continue
            outgoing[producer_id].add(consumer_id)
            incoming[consumer_id] += 1

    ready = sorted((tile_id for tile_id, count in incoming.items() if count == 0), key=order.get)
    result = []
    while ready:
        tile_id = ready.pop(0)
        result.append(by_tile_id[tile_id])
        for consumer_id in sorted(outgoing[tile_id], key=order.get):
            incoming[consumer_id] -= 1
            if incoming[consumer_id] == 0:
                ready.append(consumer_id)
                ready.sort(key=order.get)
    if len(result) != len(tile_plans):
        raise ValueError("static event-coordinate dependencies do not form a tile-phase DAG")
    return result


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
    kernel_tile_ids = {id(tile) for tile in plan.kernel.tiles}
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
        if id(tile_plan.tile) not in kernel_tile_ids:
            raise ValueError("lowering plan references a tile outside its kernel")
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
