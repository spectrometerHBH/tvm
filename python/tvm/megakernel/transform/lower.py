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
"""Parser-style physical-program lowering for logical megakernel specifications."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tvm
from tvm.ir import IRModule
from tvm.script import tirx as T
from tvm.tirx import PrimFunc

from ..dsl import EventSpec, KernelSpec, TensorSpec
from .model import (
    BarrierStep,
    DeviceRegionPlan,
    ExecutionPlan,
    ExecutionPlanBackend,
    FetchGuardStep,
    HookStep,
    HostCallStep,
    HostSyncStep,
    MidBodyPortStep,
    NotifyStep,
    ProgramStep,
    QueuePushStep,
    RunStep,
    RuntimeEventInitStep,
    TileProgram,
    WaitStep,
    make_static_execution_plan,
)
from .prepare import (
    INIT_EVENT_JOB_ID,
    WAIT_EVENT_INIT_JOB_ID,
    TIRXLoweringPlan,
    lower_shape,
    prepare_tirx_lowering_plan,
)
from .scheduler import StaticTileScheduler, TIRXSemaphore
from .smem import TIRXSmemManager


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
    backend: ExecutionPlanBackend | None = None


@dataclass
class _TensorBinding:
    spec: TensorSpec
    name: str
    buffer: Any = None


@dataclass
class _BuildState:
    plan: ExecutionPlan
    options: LoweringOptions
    lowering_plan: TIRXLoweringPlan
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


class TIRXStaticBackend(ExecutionPlanBackend):
    """Default parser backend for one static persistent device region."""

    def __init__(self, options: LoweringOptions):
        self.options = options

    def lower(self, plan: ExecutionPlan) -> PrimFunc:
        if self.options.schedule != "static":
            raise NotImplementedError("the default backend supports only static scheduling")
        state = self._prepare_state(self.prepare(plan))
        try:
            return _megakernel_entry.specialize(emitter=_ParserEmitter(self, state))
        finally:
            self._restore_tensor_specs(state)

    def prepare(self, plan: ExecutionPlan) -> TIRXLoweringPlan:
        """Prepare the private binding/layout/schedule plan for this backend."""

        return prepare_tirx_lowering_plan(plan, self.options)

    def _prepare_state(self, lowering_plan: TIRXLoweringPlan) -> _BuildState:
        state = _BuildState(lowering_plan.execution, self.options, lowering_plan)
        for binding in lowering_plan.tensor_bindings:
            state.tensor_bindings[id(binding.tensor)] = _TensorBinding(
                binding.tensor, binding.param_name
            )
        return state

    def _emit_kernel(self, state: _BuildState) -> None:
        plan = state.plan
        kernel = plan.kernel
        region = state.lowering_plan.region
        T.func_attr({"global_symbol": kernel.name})
        self._emit_var_args(state)
        self._emit_tensor_args(state)
        self._patch_tensor_specs(state)
        self._emit_event_workspace(state)
        attrs = state.lowering_plan.attrs
        sm_count = attrs.get("sm_count", 1)
        num_threads = attrs.get("num_threads", 256)
        max_tasks = state.lowering_plan.static_schedule.max_tasks
        state.queue = T.arg("queue", T.Buffer((sm_count, max_tasks), "int32"))
        T.device_entry()

        state.smem_manager = TIRXSmemManager(
            self.options.smem_max_bytes,
            self.options.smem_chunk_size,
            num_threads=num_threads,
            warp_count=attrs.get("warp_count"),
        )
        self._bind_event_buffers(state)
        state.smem_manager.init()
        runtime = {"state": state, "region": region, "indices": (None, None, None)}
        for step in region.prologue_steps:
            self._emit_tirx_step(step, runtime)

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

        for step in region.epilogue_steps:
            self._emit_tirx_step(step, runtime)

    def _emit_var_args(self, state: _BuildState) -> None:
        for binding in state.lowering_plan.var_bindings:
            name = binding.param_name
            state.var_values[id(binding.var)] = T.arg(name, T.Var(name, binding.var.dtype))

    def _shape(self, state: _BuildState, shape, label: str) -> tuple[Any, ...]:
        return lower_shape(shape, state.var_values, label)

    def _emit_tensor_args(self, state: _BuildState) -> None:
        for binding in state.tensor_bindings.values():
            shape = self._shape(state, binding.spec.shape, f"tensor {binding.spec.name!r}")
            binding.buffer = T.arg(binding.name, T.Buffer(shape, binding.spec.dtype))

    def _emit_event_workspace(self, state: _BuildState) -> None:
        lowering_plan = state.lowering_plan
        if not lowering_plan.event_layouts:
            return
        for layout in lowering_plan.event_layouts:
            event = layout.event
            shape = self._shape(state, event.shape, f"event {event.name!r}")
            size = _shape_product(shape)
            state.event_sizes[id(event)] = size
        state.event_complete_coord = lowering_plan.event_init_complete_layout.workspace_offset
        state.event_workspace = T.arg(
            "event_workspace",
            T.Buffer((lowering_plan.event_workspace_size,), "int32"),
        )

    def _bind_event_buffers(self, state: _BuildState) -> None:
        if state.event_workspace is None:
            return
        for layout in state.lowering_plan.event_layouts:
            event = layout.event
            shape = self._shape(state, event.shape, f"event {event.name!r}")
            state.event_buffers[id(event)] = T.decl_buffer(
                shape,
                event.dtype,
                data=state.event_workspace.data,
                elem_offset=layout.workspace_offset,
                scope="global",
            )

    def _emit_dispatch(self, state: _BuildState, region: DeviceRegionPlan, runtime) -> None:
        items: list[tuple[int, str | TileProgram]] = []
        if state.event_buffers:
            items.extend(
                [
                    (INIT_EVENT_JOB_ID, "init_event"),
                    (WAIT_EVENT_INIT_JOB_ID, "wait_event_init"),
                ]
            )
        items.extend(
            (tile_plan.job_id, tile_plan.program) for tile_plan in state.lowering_plan.tile_plans
        )
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
                    self._emit_tile_program(item, runtime)
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

    def _emit_tile_program(self, program: TileProgram, runtime) -> None:
        smem = runtime["state"].smem_manager
        entered_smem = False
        if program.smem_scope == "program":
            smem.enter_tile_runtime(program.tile)
            entered_smem = True
        for step in program.steps:
            if (
                program.smem_scope == "run_to_end"
                and isinstance(step, RunStep)
                and not entered_smem
            ):
                smem.enter_tile_runtime(program.tile)
                entered_smem = True
            self._emit_tirx_step(step, runtime, program.tile)
        if entered_smem:
            smem.validate_tile_phase(program.tile)
            smem.exit_tile_runtime()

    def _emit_tirx_step(self, step: ProgramStep, runtime, tile=None) -> None:
        state: _BuildState = runtime["state"]
        indices = runtime["indices"]
        smem = state.smem_manager
        scheduler = state.scheduler
        if isinstance(step, HookStep):
            if step.hook == "init_shared_resources":
                smem.set_tile(None)
                step.target.init_shared_resources(smem)
            elif step.hook == "finalize_shared_resources":
                step.target.finalize_shared_resources(smem)
            elif step.hook == "device_init":
                smem.set_tile(step.target)
                step.target.device_init(smem, *indices)
            elif step.hook == "prefetch":
                step.target.prefetch(*indices)
            elif step.hook == "smem_commit":
                smem.commit()
            elif callable(step.target):
                step.target(*step.args, **step.kwargs)
            else:
                raise ValueError(f"unsupported hook step {step.hook!r}")
        elif isinstance(step, WaitStep):
            coord = self._event_coord(step.coord_map, indices)
            semaphore = TIRXSemaphore(state.event_buffers[id(step.event)])
            scheduler.wait(semaphore, *coord, wait_level=step.level, mask=step.mask)
        elif isinstance(step, NotifyStep):
            coord = self._event_coord(step.coord_map, indices)
            semaphore = TIRXSemaphore(state.event_buffers[id(step.event)])

            def notify_func(_notify_idx):
                return (step.count, step.rank, *coord)

            scheduler.notify(
                semaphore,
                notify_func,
                scope=step.scope,
                scope_id=step.scope_id,
                release=step.release,
            )
        elif isinstance(step, RunStep):
            self._emit_run(step, tile, indices)
        elif isinstance(step, BarrierStep):
            if step.kind == "cta":
                T.cuda.cta_sync()
            elif step.kind == "warp":
                T.cuda.warp_sync()
            else:
                raise ValueError(f"unsupported barrier kind {step.kind!r}")
        elif isinstance(step, RuntimeEventInitStep):
            if not isinstance(step.event, EventSpec):
                raise ValueError("default backend runtime event init requires an EventSpec")
            if step.scope != "thread":
                raise ValueError(
                    f"default backend runtime event init does not support {step.scope!r} scope"
                )
            buffer = state.event_buffers[id(step.event)]
            if step.scope_id == -1:
                T.buffer_store(buffer, step.value, [0])
            else:
                tid = T.thread_id([scheduler.num_threads])
                with T.If(tid == step.scope_id):
                    with T.Then():
                        T.buffer_store(buffer, step.value, [0])
        elif isinstance(step, QueuePushStep | FetchGuardStep | MidBodyPortStep):
            raise NotImplementedError(f"default backend cannot emit {type(step).__name__}")
        elif isinstance(step, HostCallStep | HostSyncStep):
            raise ValueError("host steps cannot appear in the default device backend")
        else:
            raise TypeError(f"unknown megakernel program step {type(step).__name__}")

    def _emit_run(self, step: RunStep, tile, indices) -> None:
        predicate = step.predicate(*indices) if callable(step.predicate) else step.predicate

        def run_once(repeat_idx=0):
            mapped = indices
            if step.index_map is not None:
                mapped = step.index_map(*indices, repeat_idx)
            tile.impl.run(*mapped)

        def run_body():
            if step.repeat == 1:
                run_once()
            else:
                with T.serial(step.repeat) as repeat_idx:
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
    def __init__(self, plan: TIRXLoweringPlan):
        self.plan = plan

    def emit(self) -> None:
        kernel = self.plan.kernel
        schedule = self.plan.static_schedule
        if schedule is None:
            raise ValueError("queue initialization requires a static lowering plan")
        attrs = self.plan.attrs
        sm_count = schedule.sm_count
        num_threads = attrs.get("num_threads", 256)
        max_tasks = schedule.max_tasks
        T.func_attr({"global_symbol": f"{kernel.name}_init_queue"})
        var_values = {
            id(binding.var): T.arg(
                binding.param_name,
                T.Var(binding.param_name, binding.var.dtype),
            )
            for binding in self.plan.var_bindings
        }
        queue = T.arg("queue", T.Buffer((sm_count, max_tasks), "int32"))
        T.device_entry()
        T.cta_id([sm_count])
        tid = T.thread_id([num_threads])
        offset = T.alloc_buffer((1,), "int32", scope="local")
        T.buffer_store(offset, 0, [0])
        for phase in schedule.phases:
            job_id = phase.job_id
            shape = lower_shape(
                phase.tile_num,
                var_values,
                f"static phase {phase.label!r}",
            )
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


def _pack_static_task(m_idx, n_idx, k_idx, job_id):
    return T.bitwise_or(
        T.bitwise_or(job_id, T.shift_left(m_idx, 5)),
        T.bitwise_or(T.shift_left(n_idx, 18), T.shift_left(k_idx, 28)),
    )


def _replace_tensor_specs(value: Any, bindings: dict[int, _TensorBinding]) -> Any:
    if isinstance(value, TensorSpec) and id(value.base_tensor) in bindings:
        return bindings[id(value.base_tensor)].buffer
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
    backend: ExecutionPlanBackend | None = None,
    options: LoweringOptions | None = None,
):
    """Validate and lower an explicit execution plan through a backend."""

    plan.validate()
    resolved = _resolve_options(options)
    selected_backend = backend or resolved.backend or TIRXStaticBackend(resolved)
    return selected_backend.lower(plan)


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
    """Build the queue initializer matching the default static physical plan."""

    resolved = _resolve_options(options)
    plan = resolved.execution_plan or make_static_execution_plan(kernel)
    plan.validate()
    lowering_plan = prepare_tirx_lowering_plan(plan, resolved)
    return _queue_init_entry.specialize(emitter=_QueueInitEmitter(lowering_plan))


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
    """Module pass wrapper around parser-style physical-program lowering."""

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
