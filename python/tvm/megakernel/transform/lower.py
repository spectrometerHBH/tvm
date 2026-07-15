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
"""Parser-style action lowering for logical megakernel specifications."""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import Any

import tvm
from tvm.ir import IRModule
from tvm.script import tirx as T
from tvm.tirx import PrimFunc

from ..dsl import EventSpec, KernelSpec, TensorSpec, VarSpec
from .model import (
    Action,
    BarrierAction,
    DeviceRegionPlan,
    EmissionContext,
    ExecutionPlan,
    FetchGuardAction,
    HookAction,
    HostCallAction,
    HostEdgeAction,
    MegakernelBackend,
    MidBodyPortAction,
    NotifyAction,
    ProfileAction,
    QueuePushAction,
    RunAction,
    RuntimeEventInitAction,
    SmemEnterAction,
    SmemExitAction,
    TileActionProgram,
    TileEmitter,
    WaitAction,
    make_static_execution_plan,
)
from .scheduler import StaticTileScheduler, TIRXSemaphore
from .smem import TIRXSmemManager

INIT_EVENT_JOB_ID = 29
WAIT_EVENT_INIT_JOB_ID = 30


@T.inline
def _wait_event_init_complete(buffer, coord, sm_count, warp_count):
    state = T.alloc_buffer((1,), "int32", scope="local", align=4)
    state[0] = -1
    warp_id = T.warp_id([warp_count])
    lane_id = T.lane_id([32])
    while 1:
        if lane_id == 0:
            T.ptx.ld_global_acquire(state[0], T.address_of(buffer[coord]))
        if T.ptx.any_sync(
            0xFFFFFFFF,
            (state[0] <= sm_count * (TIRXSemaphore.base + 1)) & (state[0] > 0),
        ):
            if (lane_id == 0) & (warp_id == 0):
                T.cuda.atomic_add(T.address_of(buffer[coord]), -(TIRXSemaphore.base + 1))
            break
        T.cuda.nano_sleep(40)


@dataclass(frozen=True)
class LoweringOptions:
    """Options for the default single-device static backend."""

    smem_max_bytes: int = 228 * 1024
    smem_chunk_size: int = 16 * 1024
    schedule: str = "static"
    attrs: dict[str, Any] = field(default_factory=dict)
    execution_plan: ExecutionPlan | None = None
    backend: MegakernelBackend | None = None


@dataclass
class _TensorBinding:
    spec: TensorSpec
    name: str
    buffer: Any = None


@dataclass
class _BuildState:
    plan: ExecutionPlan
    options: LoweringOptions
    var_values: dict[int, Any] = field(default_factory=dict)
    tensor_bindings: dict[int, _TensorBinding] = field(default_factory=dict)
    event_buffers: dict[int, Any] = field(default_factory=dict)
    event_sizes: dict[int, Any] = field(default_factory=dict)
    event_workspace: Any = None
    event_complete_coord: Any = None
    queue: Any = None
    scheduler: StaticTileScheduler | None = None
    smem_manager: TIRXSmemManager | None = None
    tensor_patches: list[tuple[Any, str, Any]] = field(default_factory=list)


class _ParserEmitter:
    def __init__(self, backend: TIRXStaticBackend, state: _BuildState):
        self.backend = backend
        self.state = state

    def emit(self) -> None:
        self.backend._emit_kernel(self.state)  # pylint: disable=protected-access


@T.jit(check_well_formed=False)
def _megakernel_entry(*, emitter: T.constexpr):
    emitter.emit()


class TIRXStaticBackend(MegakernelBackend):
    """Default parser backend for one static persistent device region."""

    def __init__(self, options: LoweringOptions):
        self.options = options
        self._visited_actions: list[tuple[Action, EmissionContext]] = []

    def begin_execution(self, plan: ExecutionPlan) -> None:
        if self.options.schedule != "static":
            raise NotImplementedError("the default backend supports only static scheduling")
        if len(plan.device_regions) != 1 or plan.host_regions:
            raise ValueError("the default backend requires exactly one device region")

    def emit_action(self, action: Action, context: EmissionContext) -> None:
        self._visited_actions.append((action, context))

    def end_execution(self, plan: ExecutionPlan) -> PrimFunc:
        state = self._prepare_state(plan)
        try:
            return _megakernel_entry.specialize(emitter=_ParserEmitter(self, state))
        finally:
            self._restore_tensor_specs(state)

    def _prepare_state(self, plan: ExecutionPlan) -> _BuildState:
        used_names: set[str] = set()
        state = _BuildState(plan, self.options)
        for tensor in plan.kernel.tensors.values():
            name = _sanitize_identifier(tensor.name, used_names)
            state.tensor_bindings[id(tensor)] = _TensorBinding(tensor, name)
        return state

    def _emit_kernel(self, state: _BuildState) -> None:
        plan = state.plan
        kernel = plan.kernel
        region = plan.device_regions[0]
        T.func_attr({"global_symbol": kernel.name})
        self._emit_var_args(state)
        self._emit_tensor_args(state)
        self._patch_tensor_specs(state)
        self._emit_event_workspace(state)
        attrs = {**region.attrs, **self.options.attrs}
        sm_count = attrs.get("sm_count", 1)
        num_threads = attrs.get("num_threads", 256)
        max_tasks = attrs.get("max_tasks", StaticTileScheduler.MAX_TASKS)
        state.queue = T.arg("queue", T.Buffer((sm_count, max_tasks), "int32"))
        T.device_entry()

        state.smem_manager = TIRXSmemManager(
            self.options.smem_max_bytes, self.options.smem_chunk_size
        )
        self._bind_event_buffers(state)
        runtime = {"state": state, "region": region, "indices": (None, None, None)}
        for action in region.prologue:
            self._emit_tirx_action(action, runtime)

        state.scheduler = StaticTileScheduler(
            state.queue,
            state.smem_manager,
            debug=attrs.get("debug_scheduler", False),
            sm_count=sm_count,
            num_threads=num_threads,
            max_tasks=max_tasks,
            end_job_id=attrs.get("end_job_id", 31),
            warp_count=attrs.get("warp_count"),
            warpgroup_count=attrs.get("warpgroup_count"),
            warpgroup_size=attrs.get("warpgroup_size", 128),
        )
        state.scheduler.init()

        with T.While(state.scheduler.valid()):
            indices = state.scheduler.indices()
            runtime["indices"] = indices
            self._emit_dispatch(state, region, runtime)
            state.scheduler.next_tile()

        for action in region.epilogue:
            self._emit_tirx_action(action, runtime)

    def _emit_var_args(self, state: _BuildState) -> None:
        used_names: set[str] = set()
        for var in state.plan.kernel.vars.values():
            name = _sanitize_identifier(var.name, used_names)
            state.var_values[id(var)] = T.arg(name, T.Var(name, var.dtype))

    def _lower_extent(self, state: _BuildState, value: Any, label: str):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, VarSpec) and id(value) in state.var_values:
            return state.var_values[id(value)]
        raise ValueError(f"{label} contains an unbound symbolic variable")

    def _shape(self, state: _BuildState, shape, label: str) -> tuple[Any, ...]:
        values = tuple(shape) if isinstance(shape, tuple | list) else (shape,)
        return tuple(self._lower_extent(state, value, label) for value in values)

    def _emit_tensor_args(self, state: _BuildState) -> None:
        for binding in state.tensor_bindings.values():
            shape = self._shape(state, binding.spec.shape, f"tensor {binding.spec.name!r}")
            binding.buffer = T.arg(binding.name, T.Buffer(shape, binding.spec.dtype))

    def _emit_event_workspace(self, state: _BuildState) -> None:
        if not state.plan.kernel.events:
            return
        total = 0
        for event in state.plan.kernel.events.values():
            shape = self._shape(state, event.shape, f"event {event.name!r}")
            size = _shape_product(shape)
            state.event_sizes[id(event)] = size
            total += size
        state.event_complete_coord = total
        state.event_workspace = T.arg("event_workspace", T.Buffer((total + 1,), "int32"))

    def _bind_event_buffers(self, state: _BuildState) -> None:
        if state.event_workspace is None:
            return
        offset = 0
        for event in state.plan.kernel.events.values():
            shape = self._shape(state, event.shape, f"event {event.name!r}")
            state.event_buffers[id(event)] = T.decl_buffer(
                shape,
                event.dtype,
                data=state.event_workspace.data,
                elem_offset=offset,
                scope="global",
            )
            offset += state.event_sizes[id(event)]

    def _emit_dispatch(self, state: _BuildState, region: DeviceRegionPlan, runtime) -> None:
        items: list[tuple[int, str | TileActionProgram]] = []
        if state.event_buffers:
            items.extend(
                [
                    (INIT_EVENT_JOB_ID, "init_event"),
                    (WAIT_EVENT_INIT_JOB_ID, "wait_event_init"),
                ]
            )
        items.extend((job_id, program) for job_id, program in enumerate(region.tile_programs))
        if not items:
            return
        task_type = state.scheduler.task_type[0]
        if_frames = [T.If(task_type == job_id) for job_id, _ in items]
        then_frames = [T.Then() for _ in items]
        else_frames = [T.Else() for _ in items]
        for index, (_, item) in enumerate(items):
            if_frames[index].__enter__()
            with then_frames[index]:
                if item == "init_event":
                    self._emit_init_event_task(state)
                elif item == "wait_event_init":
                    self._emit_wait_event_init_task(state)
                else:
                    for action in item.actions:
                        self._emit_tirx_action(action, runtime)
            else_frames[index].__enter__()
        T.evaluate(T.cuda.trap_when_assert_failed(False))
        for index in range(len(items) - 1, -1, -1):
            else_frames[index].__exit__(None, None, None)
            if_frames[index].__exit__(None, None, None)

    def _event_coord(self, coord_map, indices) -> tuple[Any, ...]:
        coord = coord_map(*indices) if callable(coord_map) else coord_map
        if not isinstance(coord, tuple | list):
            raise TypeError("event coordinate map must return a tuple or list")
        return tuple(coord)

    def _emit_tirx_action(self, action: Action, runtime) -> None:
        state: _BuildState = runtime["state"]
        indices = runtime["indices"]
        smem = state.smem_manager
        scheduler = state.scheduler
        if isinstance(action, SmemEnterAction):
            smem.enter_tile_runtime(action.tile)
        elif isinstance(action, SmemExitAction):
            smem.exit_tile_runtime()
        elif isinstance(action, HookAction):
            if action.hook == "init_shared_resources":
                smem.set_tile(None)
                action.target.init_shared_resources(smem)
            elif action.hook == "finalize_shared_resources":
                action.target.finalize_shared_resources(smem)
            elif action.hook == "device_init":
                smem.set_tile(action.target)
                action.target.device_init(smem, *indices)
            elif action.hook == "prefetch":
                action.target.prefetch(*indices)
            elif action.hook == "smem_commit":
                smem.commit()
            elif callable(action.target):
                action.target(*action.args, **action.kwargs)
            else:
                raise ValueError(f"unsupported hook action {action.hook!r}")
        elif isinstance(action, WaitAction):
            coord = self._event_coord(action.coord_map, indices)
            semaphore = TIRXSemaphore(state.event_buffers[id(action.event)])
            scheduler.wait(semaphore, *coord, wait_level=action.level, mask=action.mask)
        elif isinstance(action, NotifyAction):
            coord = self._event_coord(action.coord_map, indices)
            semaphore = TIRXSemaphore(state.event_buffers[id(action.event)])

            def notify_func(_notify_idx):
                return (action.count, action.rank, *coord)

            scheduler.notify(
                semaphore,
                notify_func,
                scope=action.scope,
                scope_id=action.scope_id,
                release=action.release,
            )
        elif isinstance(action, RunAction):
            self._emit_run(action, indices)
        elif isinstance(action, BarrierAction):
            if action.kind == "cta":
                T.cuda.cta_sync()
            elif action.kind == "warp":
                T.cuda.warp_sync()
            elif callable(action.payload):
                action.payload(runtime)
            else:
                raise ValueError(f"unsupported barrier kind {action.kind!r}")
        elif isinstance(action, RuntimeEventInitAction):
            if callable(action.payload):
                action.payload(runtime)
            elif isinstance(action.event, EventSpec):
                buffer = state.event_buffers[id(action.event)]
                if action.predicate is None:
                    T.buffer_store(buffer, action.value, [0])
                elif action.predicate:
                    T.buffer_store(buffer, action.value, [0])
            else:
                raise ValueError("runtime event init requires an EventSpec or backend payload")
        elif isinstance(action, ProfileAction):
            if callable(action.payload):
                action.payload(runtime)
        elif isinstance(action, QueuePushAction | FetchGuardAction | MidBodyPortAction):
            if not callable(action.payload):
                raise NotImplementedError(f"default backend cannot emit {type(action).__name__}")
            action.payload(runtime)
        elif isinstance(action, HostCallAction | HostEdgeAction):
            raise ValueError("host actions cannot appear in the default device backend")
        else:
            raise TypeError(f"unknown megakernel action {type(action).__name__}")

    def _emit_run(self, action: RunAction, indices) -> None:
        predicate = action.predicate(*indices) if callable(action.predicate) else action.predicate

        def run_once(repeat_idx=0):
            mapped = indices
            if action.index_map is not None:
                mapped = action.index_map(*indices, repeat_idx)
            action.tile.impl.run(*mapped)

        def run_body():
            if action.repeat == 1:
                run_once()
            else:
                with T.serial(action.repeat) as repeat_idx:
                    run_once(repeat_idx)

        if predicate is None or predicate is True:
            run_body()
        elif predicate is not False:
            with T.If(predicate):
                with T.Then():
                    run_body()

    def _emit_init_event_task(self, state: _BuildState) -> None:
        events = list(state.plan.kernel.events.values())
        event_id = state.scheduler.m_idx[0]
        tid = T.thread_id([state.scheduler.num_threads])
        for static_id, event in enumerate(events):
            with T.If(event_id == static_id):
                with T.Then():
                    index = T.alloc_buffer((1,), "int32", scope="local")
                    T.buffer_store(index, tid, [0])
                    shape = self._shape(state, event.shape, f"event {event.name!r}")
                    with T.While(index[0] < state.event_sizes[id(event)]):
                        coord = _linear_index_to_coord(index[0], shape)
                        count = (
                            event.init_count(coord)
                            if callable(event.init_count)
                            else event.init_count
                        )
                        T.buffer_store(
                            state.event_buffers[id(event)],
                            count * (TIRXSemaphore.base + 1),
                            list(coord),
                        )
                        T.buffer_store(
                            index,
                            index[0] + state.scheduler.num_threads,
                            [0],
                        )
                    self._notify_event_init_complete(state)
        with T.If(event_id == len(events)):
            with T.Then():
                complete_init = (len(events) + 1 + state.scheduler.sm_count) * (
                    TIRXSemaphore.base + 1
                )
                T.buffer_store(
                    state.event_workspace,
                    complete_init,
                    [state.event_complete_coord],
                )
                self._notify_event_init_complete(state)

    def _notify_event_init_complete(self, state: _BuildState) -> None:
        semaphore = TIRXSemaphore(state.event_workspace)

        def notify_func(_notify_idx):
            return (1, -1, state.event_complete_coord)

        state.scheduler.notify(semaphore, notify_func, scope="cta")

    def _emit_wait_event_init_task(self, state: _BuildState) -> None:
        _wait_event_init_complete(
            state.event_workspace,
            state.event_complete_coord,
            state.scheduler.sm_count,
            state.scheduler.warp_count,
        )

    def _patch_tensor_specs(self, state: _BuildState) -> None:
        for tile in state.plan.kernel.tiles:
            impl = tile.impl
            for name, value in vars(impl).items():
                replaced = _replace_tensor_specs(value, state.tensor_bindings)
                if replaced is not value:
                    state.tensor_patches.append((impl, name, value))
                    setattr(impl, name, replaced)

    def _restore_tensor_specs(self, state: _BuildState) -> None:
        for impl, name, value in reversed(state.tensor_patches):
            setattr(impl, name, value)


class _QueueInitEmitter:
    def __init__(self, plan: ExecutionPlan, options: LoweringOptions):
        self.plan = plan
        self.options = options

    def emit(self) -> None:
        kernel = self.plan.kernel
        region = self.plan.device_regions[0]
        attrs = {**region.attrs, **self.options.attrs}
        sm_count = attrs.get("sm_count", 1)
        num_threads = attrs.get("num_threads", 256)
        max_tasks = attrs.get("max_tasks", StaticTileScheduler.MAX_TASKS)
        end_job_id = attrs.get("end_job_id", 31)
        T.func_attr({"global_symbol": f"{kernel.name}_init_queue"})
        var_values = {
            id(var): T.arg(var.name, T.Var(var.name, var.dtype)) for var in kernel.vars.values()
        }
        queue = T.arg("queue", T.Buffer((sm_count, max_tasks), "int32"))
        T.device_entry()
        T.cta_id([sm_count])
        tid = T.thread_id([num_threads])
        offset = T.alloc_buffer((1,), "int32", scope="local")
        T.buffer_store(offset, 0, [0])
        phases = []
        events = list(kernel.events.values())
        if events:
            phases.append((INIT_EVENT_JOB_ID, (len(events) + 1, 1, 1)))
        for job_id, program in enumerate(region.tile_programs):
            if not program.tile.waits:
                phases.append((job_id, _lower_shape(program.tile.tile_num, var_values)))
        if events:
            phases.append((WAIT_EVENT_INIT_JOB_ID, (sm_count, 1, 1)))
        for job_id, program in enumerate(region.tile_programs):
            if program.tile.waits:
                phases.append((job_id, _lower_shape(program.tile.tile_num, var_values)))
        phases.append((end_job_id, (sm_count, 1, 1)))
        for job_id, shape in phases:
            count = _shape_product(shape)
            with T.If(tid == 0):
                with T.Then():
                    grid = T.grid(*shape)
                    indices = grid.__enter__()
                    packed = _pack_static_task(*indices, job_id)
                    T.buffer_store(
                        queue,
                        packed,
                        [offset[0] % sm_count, offset[0] // sm_count],
                    )
                    T.buffer_store(offset, offset[0] + 1, [0])
                    grid.__exit__(None, None, None)
                with T.Else():
                    T.buffer_store(offset, offset[0] + count, [0])


@T.jit(check_well_formed=False)
def _queue_init_entry(*, emitter: T.constexpr):
    emitter.emit()


def _sanitize_identifier(name: str, used: set[str]) -> str:
    candidate = re.sub(r"\W", "_", name)
    if not candidate or candidate[0].isdigit() or keyword.iskeyword(candidate):
        candidate = f"value_{candidate}"
    base = candidate
    suffix = 1
    while candidate in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


def _shape_product(shape) -> Any:
    result = 1
    for extent in shape:
        result *= extent
    return result


def _linear_index_to_coord(index, shape) -> tuple[Any, ...]:
    coord = []
    remaining = index
    for extent in reversed(shape):
        coord.append(remaining % extent)
        remaining = remaining // extent
    return tuple(reversed(coord))


def _lower_shape(shape, var_values) -> tuple[Any, ...]:
    values = tuple(shape) if isinstance(shape, tuple | list) else (shape,)
    return tuple(var_values[id(value)] if isinstance(value, VarSpec) else value for value in values)


def _pack_static_task(m_idx, n_idx, k_idx, job_id):
    return T.bitwise_or(
        T.bitwise_or(job_id, T.shift_left(m_idx, 5)),
        T.bitwise_or(T.shift_left(n_idx, 18), T.shift_left(k_idx, 28)),
    )


def _replace_tensor_specs(value: Any, bindings: dict[int, _TensorBinding]) -> Any:
    if isinstance(value, TensorSpec) and id(value) in bindings:
        return bindings[id(value)].buffer
    if isinstance(value, tuple):
        return tuple(_replace_tensor_specs(item, bindings) for item in value)
    if isinstance(value, list):
        return [_replace_tensor_specs(item, bindings) for item in value]
    if isinstance(value, dict):
        return {
            _replace_tensor_specs(key, bindings): _replace_tensor_specs(item, bindings)
            for key, item in value.items()
        }
    return value


def _resolve_options(options: LoweringOptions | None) -> LoweringOptions:
    if options is None:
        return LoweringOptions()
    if not isinstance(options, LoweringOptions):
        raise TypeError("options must be a LoweringOptions instance or None")
    return options


def lower_execution_plan(
    plan: ExecutionPlan,
    backend: MegakernelBackend | None = None,
    options: LoweringOptions | None = None,
):
    """Validate and lower an explicit execution plan through a backend."""

    resolved = _resolve_options(options)
    selected_backend = backend or resolved.backend or TIRXStaticBackend(resolved)
    return TileEmitter(selected_backend).emit(plan)


def lower_to_tirx(kernel: KernelSpec, options: LoweringOptions | None = None) -> PrimFunc:
    """Lower a graph with the default static plan or an explicitly supplied plan."""

    resolved = _resolve_options(options)
    plan = resolved.execution_plan or make_static_execution_plan(kernel)
    if plan.kernel is not kernel:
        raise ValueError("execution plan belongs to a different KernelSpec")
    return lower_execution_plan(plan, options=resolved)


def lower_static_queue_init_to_tirx(
    kernel: KernelSpec, options: LoweringOptions | None = None
) -> PrimFunc:
    """Build the queue initializer matching the default static action plan."""

    resolved = _resolve_options(options)
    plan = resolved.execution_plan or make_static_execution_plan(kernel)
    plan.validate()
    if len(plan.device_regions) != 1 or plan.host_regions:
        raise ValueError("static queue init requires one device region")
    return _queue_init_entry.specialize(emitter=_QueueInitEmitter(plan, resolved))


def lower_to_tirx_module(kernel: KernelSpec, options: LoweringOptions | None = None) -> IRModule:
    """Lower a graph to its device kernel and static queue initializer."""

    resolved = _resolve_options(options)
    return IRModule(
        {
            kernel.name: lower_to_tirx(kernel, resolved),
            f"{kernel.name}_init_queue": lower_static_queue_init_to_tirx(kernel, resolved),
        }
    )


@tvm.transform.module_pass(opt_level=0, name="LowerMegakernelDSL")
class LowerMegakernelDSL:
    """Module pass wrapper around parser-style action lowering."""

    def __init__(self, kernel: KernelSpec, options: LoweringOptions | None = None):
        self.kernel = kernel
        self.options = options

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        lowered = lower_to_tirx_module(self.kernel, self.options)
        return IRModule({**mod.functions, **lowered.functions}, attrs=mod.attrs)


MegakernelLowerer = TIRXStaticBackend

__all__ = [
    "LowerMegakernelDSL",
    "LoweringOptions",
    "MegakernelLowerer",
    "TIRXStaticBackend",
    "lower_execution_plan",
    "lower_static_queue_init_to_tirx",
    "lower_to_tirx",
    "lower_to_tirx_module",
]
