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
"""Direct static TIRX lowering for logical megakernel specifications.

The emitter reads a validated ``KernelSpec`` directly.  Per tile it emits
``smem.enter_tile_runtime(tile)`` -> ``device_init`` -> ``prefetch`` ->
waits -> ``run`` -> ``smem.validate_tile_phase(tile)`` -> notifies ->
``exit_tile_runtime`` inline; there is no intermediate step program.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tvm
from tvm.ir import IRModule
from tvm.script import tirx as T
from tvm.tirx import PrimFunc

from ..dsl import KernelSpec, TensorSpec
from .prepare import (
    INIT_EVENT_JOB_ID,
    WAIT_EVENT_INIT_JOB_ID,
    TIRXLoweringPlan,
    lower_shape,
    prepare_tirx_lowering_plan,
)
from .scheduler import StaticTileScheduler, TIRXSemaphore
from .smem import TIRXSmemManager
from .validate import validate_kernel


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
                T.cuda.thread_fence()
                T.cuda.atomic_add(T.address_of(buffer[coord]), -(TIRXSemaphore.base + 1))
            break
        T.cuda.nano_sleep(40)


@dataclass(frozen=True)
class LoweringOptions:
    """Options for the default single-device static backend.

    ``scheduler`` selects the build path: ``None`` keeps the legacy direct
    emitter below, while ``"static"`` and ``"dynamic"`` route to the
    runtime-library builder in ``transform.runtime_build``.  ``schedule`` is
    the legacy emitter's own scheduling knob and is unrelated to routing.
    """

    smem_max_bytes: int = 228 * 1024
    smem_chunk_size: int = 16 * 1024
    schedule: str = "static"
    scheduler: str | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class _TensorBinding:
    spec: TensorSpec
    name: str
    buffer: Any = None


class _StaticKernelBuilder:
    """Reference builder that emits one static persistent kernel from a spec."""

    def __init__(self, plan: TIRXLoweringPlan):
        self.plan = plan
        self.options = plan.options
        self.var_values: dict[int, Any] = {}
        self.tensor_bindings: dict[int, _TensorBinding] = {
            id(binding.tensor): _TensorBinding(binding.tensor, binding.param_name)
            for binding in plan.tensor_bindings
        }
        self.event_buffers: dict[int, Any] = {}
        self.event_sizes: dict[int, Any] = {}
        self.event_workspace = None
        self.event_complete_coord = None
        self.queue = None
        self.scheduler: StaticTileScheduler | None = None
        self.smem_manager: TIRXSmemManager | None = None
        self.tensor_patches: list[tuple[Any, str, Any]] = []

    def emit(self) -> None:
        kernel = self.plan.kernel
        attrs = self.plan.attrs
        sm_count = attrs.get("sm_count", 1)
        num_threads = attrs.get("num_threads", 256)
        max_tasks = self.plan.static_schedule.max_tasks
        T.func_attr({"global_symbol": kernel.name})
        self._emit_var_args()
        self._emit_tensor_args()
        self._patch_tensor_specs()
        self._emit_event_workspace()
        self.queue = T.arg("queue", T.Buffer((sm_count, max_tasks), "int32"))
        for tile in kernel.tiles:
            tile.impl.host_init()
        T.device_entry()

        self.smem_manager = TIRXSmemManager(
            self.options.smem_max_bytes,
            self.options.smem_chunk_size,
            num_threads=num_threads,
            warp_count=attrs.get("warp_count"),
        )
        self._bind_event_buffers()
        self.smem_manager.init()
        for cls in _unique_impl_classes(kernel):
            self.smem_manager.set_tile(cls)
            cls.init_shared_resources(self.smem_manager)

        self.scheduler = StaticTileScheduler(
            self.queue,
            self.smem_manager,
            debug=attrs.get("debug_scheduler", False),
            sm_count=sm_count,
            num_threads=num_threads,
            max_tasks=max_tasks,
            end_job_id=attrs.get("end_job_id", 31),
            warp_count=attrs.get("warp_count"),
            warpgroup_count=attrs.get("warpgroup_count"),
            warpgroup_size=attrs.get("warpgroup_size", 128),
        )
        self.scheduler.init()

        with T.While(self.scheduler.valid()):
            indices = self.scheduler.indices()
            self._emit_dispatch(indices)
            self.scheduler.next_tile()

        for cls in reversed(_unique_impl_classes(kernel)):
            self.smem_manager.set_tile(cls)
            cls.finalize_shared_resources(self.smem_manager)
        self.smem_manager.commit()

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
        for binding in self.tensor_bindings.values():
            shape = self._shape(binding.spec.shape, f"tensor {binding.spec.name!r}")
            binding.buffer = T.arg(binding.name, T.Buffer(shape, binding.spec.dtype))

    def _emit_event_workspace(self) -> None:
        plan = self.plan
        if not plan.event_layouts:
            return
        for layout in plan.event_layouts:
            event = layout.event
            shape = self._shape(event.shape, f"event {event.name!r}")
            size = _shape_product(shape)
            self.event_sizes[id(event)] = size
        self.event_complete_coord = plan.event_init_complete_layout.workspace_offset
        self.event_workspace = T.arg(
            "event_workspace",
            T.Buffer((plan.event_workspace_size,), "int32"),
        )

    def _bind_event_buffers(self) -> None:
        if self.event_workspace is None:
            return
        for layout in self.plan.event_layouts:
            event = layout.event
            shape = self._shape(event.shape, f"event {event.name!r}")
            self.event_buffers[id(event)] = T.decl_buffer(
                shape,
                event.dtype,
                data=self.event_workspace.data,
                elem_offset=layout.workspace_offset,
                scope="global",
            )

    def _emit_dispatch(self, indices) -> None:
        items: list[tuple[int, Any]] = []
        if self.event_buffers:
            items.extend(
                [
                    (INIT_EVENT_JOB_ID, "init_event"),
                    (WAIT_EVENT_INIT_JOB_ID, "wait_event_init"),
                ]
            )
        items.extend((tile_plan.job_id, tile_plan.tile) for tile_plan in self.plan.tile_plans)
        if not items:
            return
        task_type = self.scheduler.task_type[0]
        if_frames = [T.If(task_type == job_id) for job_id, _ in items]
        then_frames = [T.Then() for _ in items]
        else_frames = [T.Else() for _ in items]
        for index, (_, item) in enumerate(items):
            if_frames[index].__enter__()
            with then_frames[index]:
                if item == "init_event":
                    self._emit_init_event_task()
                elif item == "wait_event_init":
                    self._emit_wait_event_init_task()
                else:
                    self._emit_tile(item, indices)
            else_frames[index].__enter__()
        T.evaluate(T.cuda.trap_when_assert_failed(False))
        for index in range(len(items) - 1, -1, -1):
            else_frames[index].__exit__(None, None, None)
            if_frames[index].__exit__(None, None, None)

    def _emit_tile(self, tile, indices) -> None:
        smem = self.smem_manager
        smem.enter_tile_runtime(tile)
        smem.set_tile(tile)
        tile.impl.device_init(smem, *indices)
        tile.impl.prefetch(*indices)
        for event, coord_map in tile.waits:
            coord = _event_coord(coord_map, indices)
            semaphore = TIRXSemaphore(self.event_buffers[id(event)])
            self.scheduler.wait(semaphore, *coord)
        tile.impl.run(*indices)
        smem.validate_tile_phase(tile)
        for event, coord_map in tile.notifies:
            coord = _event_coord(coord_map, indices)
            semaphore = TIRXSemaphore(self.event_buffers[id(event)])

            def notify_func(_notify_idx):
                return (1, -1, *coord)

            self.scheduler.notify(semaphore, notify_func, scope="cta", scope_id=0, release=True)
        smem.exit_tile_runtime()

    def _emit_init_event_task(self) -> None:
        events = list(self.plan.kernel.events.values())
        event_id = self.scheduler.m_idx[0]
        tid = T.thread_id([self.scheduler.num_threads])
        for static_id, event in enumerate(events):
            with T.If(event_id == static_id):
                with T.Then():
                    index = T.alloc_buffer((1,), "int32", scope="local")
                    T.buffer_store(index, tid, [0])
                    shape = self._shape(event.shape, f"event {event.name!r}")
                    with T.While(index[0] < self.event_sizes[id(event)]):
                        coord = _linear_index_to_coord(index[0], shape)
                        count = (
                            event.init_count(coord)
                            if callable(event.init_count)
                            else event.init_count
                        )
                        T.buffer_store(
                            self.event_buffers[id(event)],
                            count * (TIRXSemaphore.base + 1),
                            list(coord),
                        )
                        T.buffer_store(
                            index,
                            index[0] + self.scheduler.num_threads,
                            [0],
                        )
                    self._notify_event_init_complete()
        with T.If(event_id == len(events)):
            with T.Then():
                complete_init = (len(events) + 1 + self.scheduler.sm_count) * (
                    TIRXSemaphore.base + 1
                )
                T.buffer_store(
                    self.event_workspace,
                    complete_init,
                    [self.event_complete_coord],
                )
                self._notify_event_init_complete()

    def _notify_event_init_complete(self) -> None:
        semaphore = TIRXSemaphore(self.event_workspace)

        def notify_func(_notify_idx):
            return (1, -1, self.event_complete_coord)

        self.scheduler.notify(semaphore, notify_func, scope="cta", release=True)

    def _emit_wait_event_init_task(self) -> None:
        _wait_event_init_complete(
            self.event_workspace,
            self.event_complete_coord,
            self.scheduler.sm_count,
            self.scheduler.warp_count,
        )

    def _patch_tensor_specs(self) -> None:
        for tile in self.plan.kernel.tiles:
            impl = tile.impl
            for name, value in vars(impl).items():
                replaced = _replace_tensor_specs(value, self.tensor_bindings)
                if replaced is not value:
                    self.tensor_patches.append((impl, name, value))
                    setattr(impl, name, replaced)


def _unique_impl_classes(kernel: KernelSpec) -> list[type]:
    classes = []
    seen = set()
    for tile in kernel.tiles:
        cls = type(tile.impl)
        if cls not in seen:
            seen.add(cls)
            classes.append(cls)
    return classes


def _event_coord(coord_map, indices) -> tuple[Any, ...]:
    coord = coord_map(*indices) if callable(coord_map) else coord_map
    if not isinstance(coord, tuple | list):
        raise TypeError("event coordinate map must return a tuple or list")
    return tuple(coord)


@T.jit(check_well_formed=False)
def _megakernel_entry(*, emitter: T.constexpr):
    emitter.emit()


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
        block_id = T.cta_id([sm_count])
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
                    with T.If(offset[0] % sm_count == block_id):
                        with T.Then():
                            T.buffer_store(
                                queue,
                                packed,
                                [block_id, offset[0] // sm_count],
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


def _route_scheduler(options: LoweringOptions) -> str | None:
    """Validate the routing knob; return the selected backend name or None."""

    scheduler = options.scheduler
    if scheduler is None:
        return None
    if scheduler not in ("static", "dynamic"):
        raise ValueError(
            f"unsupported scheduler {scheduler!r}; expected None, 'static', or 'dynamic'"
        )
    return scheduler


def _emit_with_runtime_builder(kernel: KernelSpec, options: LoweringOptions) -> IRModule:
    from .runtime_build import emit_runtime_module  # local import avoids a cycle

    return emit_runtime_module(kernel, options)


def _prepare(kernel: KernelSpec, options: LoweringOptions) -> TIRXLoweringPlan:
    if options.schedule != "static":
        raise NotImplementedError("the default backend supports only static scheduling")
    validate_kernel(kernel)
    return prepare_tirx_lowering_plan(kernel, options)


def lower_to_tirx(kernel: KernelSpec, options: LoweringOptions | None = None) -> PrimFunc:
    """Validate a spec and lower it to the default static persistent kernel."""

    resolved = _resolve_options(options)
    if _route_scheduler(resolved) is not None:
        return _emit_with_runtime_builder(kernel, resolved)[kernel.name]
    plan = _prepare(kernel, resolved)
    builder = _StaticKernelBuilder(plan)
    try:
        return _megakernel_entry.specialize(emitter=builder)
    finally:
        builder.restore_tensor_specs()


def lower_static_queue_init_to_tirx(
    kernel: KernelSpec, options: LoweringOptions | None = None
) -> PrimFunc:
    """Build the queue initializer matching the default static kernel."""

    resolved = _resolve_options(options)
    if _route_scheduler(resolved) is not None:
        raise ValueError(
            "the runtime builder (scheduler='static'/'dynamic') derives the "
            "exec queue on the host; use "
            "tvm.megakernel.transform.build_runtime_kernel and its "
            "RuntimeKernelBuild queue arrays instead of a queue-init kernel"
        )
    plan = _prepare(kernel, resolved)
    return _queue_init_entry.specialize(emitter=_QueueInitEmitter(plan))


def lower_to_tirx_module(kernel: KernelSpec, options: LoweringOptions | None = None) -> IRModule:
    """Lower a spec to its device kernel and static queue initializer."""

    resolved = _resolve_options(options)
    if _route_scheduler(resolved) is not None:
        return _emit_with_runtime_builder(kernel, resolved)
    return IRModule(
        {
            kernel.name: lower_to_tirx(kernel, resolved),
            f"{kernel.name}_init_queue": lower_static_queue_init_to_tirx(kernel, resolved),
        }
    )


@tvm.transform.module_pass(opt_level=0, name="LowerMegakernelDSL")
class LowerMegakernelDSL:
    """Module pass wrapper around direct static megakernel lowering."""

    def __init__(self, kernel: KernelSpec, options: LoweringOptions | None = None):
        self.kernel = kernel
        self.options = options

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        lowered = lower_to_tirx_module(self.kernel, self.options)
        return IRModule({**mod.functions, **lowered.functions}, attrs=mod.attrs)


__all__ = [
    "LowerMegakernelDSL",
    "LoweringOptions",
    "lower_static_queue_init_to_tirx",
    "lower_to_tirx",
    "lower_to_tirx_module",
]
