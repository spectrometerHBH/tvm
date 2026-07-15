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
"""Execution-plan and action-program contracts for megakernel lowering."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from ..dsl import CoordMapType, EventSpec, KernelSpec, TileSpec

EdgeLocation = Literal["tile_action", "fetch_guard", "host_runtime", "mid_body_port"]
RegionDependencyKind = Literal["launch_order", "completion"]


@dataclass(frozen=True, eq=False)
class LogicalEdge:
    """One producer-to-consumer edge induced by a logical event."""

    event: EventSpec
    producer: str
    consumer: str

    @property
    def key(self) -> tuple[int, str, str]:
        return (id(self.event), self.producer, self.consumer)

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, LogicalEdge) and self.key == other.key

    def __str__(self) -> str:
        return f"{self.producer}-[{self.event.name}]->{self.consumer}"


class Action:
    """Marker base class for physical lowering actions."""


@dataclass(frozen=True)
class HookAction(Action):
    """Invoke a lifecycle or implementation hook selected by the backend."""

    hook: str
    target: Any = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WaitAction(Action):
    """Wait for one physical event endpoint."""

    edges: tuple[LogicalEdge, ...]
    event: EventSpec
    coord_map: CoordMapType
    level: str = "cta"
    mask: int = 0xFFFFFFFF
    payload: Any = None


@dataclass(frozen=True)
class NotifyAction(Action):
    """Notify one physical event endpoint."""

    edges: tuple[LogicalEdge, ...]
    event: EventSpec
    coord_map: CoordMapType
    scope: str = "cta"
    scope_id: int = 0
    count: Any = 1
    rank: int = -1
    release: bool = False
    payload: Any = None


@dataclass(frozen=True)
class QueuePushAction(Action):
    """Push scheduler work, optionally as part of a logical edge binding."""

    edges: tuple[LogicalEdge, ...] = ()
    payload: Any = None


@dataclass(frozen=True)
class BarrierAction(Action):
    """Emit a barrier at its exact source-order position."""

    kind: str
    payload: Any = None


@dataclass(frozen=True)
class RuntimeEventInitAction(Action):
    """Initialize a runtime-sized physical event."""

    event: Any
    value: Any
    predicate: Any = None
    payload: Any = None


@dataclass(frozen=True)
class ProfileAction(Action):
    """Emit one profiling lifecycle operation."""

    phase: Literal["init", "begin", "end", "finalize"]
    event: Any = None
    predicate: Any = None
    payload: Any = None


@dataclass(frozen=True)
class SmemEnterAction(Action):
    """Enter a tile's managed shared-memory runtime scope."""

    tile: TileSpec


@dataclass(frozen=True)
class SmemExitAction(Action):
    """Exit a tile's managed shared-memory runtime scope."""

    tile: TileSpec


@dataclass(frozen=True)
class RunAction(Action):
    """Run a tile under an optional predicate, repeat, and index mapping."""

    tile: TileSpec
    predicate: Any = None
    repeat: Any = 1
    index_map: Any = None
    payload: Any = None


@dataclass(frozen=True)
class FetchGuardAction(Action):
    """Guard publication of one scheduler fetch result."""

    edges: tuple[LogicalEdge, ...]
    predicate: Any = None
    payload: Any = None


@dataclass(frozen=True)
class MidBodyPortAction(Action):
    """Bind an edge to a backend-approved location inside a tile body."""

    edges: tuple[LogicalEdge, ...]
    port: str
    payload: Any = None


@dataclass(frozen=True)
class HostEdgeAction(Action):
    """Realize a logical edge in a host or runtime region."""

    edges: tuple[LogicalEdge, ...]
    kind: RegionDependencyKind
    payload: Any = None


@dataclass(frozen=True)
class HostCallAction(Action):
    """Invoke a backend-owned host operation."""

    name: str
    payload: Any = None


@dataclass(frozen=True)
class SchedulerFetchProgram:
    """Ordered actions that fetch and publish scheduler work."""

    actions: tuple[Action, ...] = ()


@dataclass(frozen=True)
class TileActionProgram:
    """Ordered physical actions for one logical tile."""

    tile: TileSpec
    actions: tuple[Action, ...]


@dataclass(frozen=True)
class DeviceRegionPlan:
    """One device region with lifecycle, fetch, and per-tile programs."""

    name: str
    prologue: tuple[Action, ...] = ()
    fetch_program: SchedulerFetchProgram = field(default_factory=SchedulerFetchProgram)
    tile_programs: tuple[TileActionProgram, ...] = ()
    epilogue: tuple[Action, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HostRegionPlan:
    """One host/runtime region in the execution DAG."""

    name: str
    actions: tuple[Action, ...]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegionDependencyPlan:
    """Ordering between regions without conflating launch and completion."""

    source: str
    target: str
    kind: RegionDependencyKind
    payload: Any = None


@dataclass(frozen=True)
class EdgeBindingPlan:
    """The single approved physical location for one logical edge."""

    edge: LogicalEdge
    location: EdgeLocation
    region: str
    port: str | None = None


def logical_edges(kernel: KernelSpec) -> tuple[LogicalEdge, ...]:
    """Return logical event edges in stable kernel/tile order."""

    producers: dict[int, list[str]] = defaultdict(list)
    consumers: dict[int, list[str]] = defaultdict(list)
    event_by_id = {id(event): event for event in kernel.events.values()}
    for tile in kernel.tiles:
        for event, _ in tile.notifies:
            if tile.name not in producers[id(event)]:
                producers[id(event)].append(tile.name)
        for event, _ in tile.waits:
            if tile.name not in consumers[id(event)]:
                consumers[id(event)].append(tile.name)

    result = []
    for event in kernel.events.values():
        event_id = id(event)
        for producer in producers[event_id]:
            for consumer in consumers[event_id]:
                result.append(LogicalEdge(event_by_id[event_id], producer, consumer))
    return tuple(result)


def _action_edges(action: Action) -> tuple[LogicalEdge, ...]:
    edges = getattr(action, "edges", ())
    return tuple(edges)


@dataclass(frozen=True)
class ExecutionPlan:
    """Complete physical plan for one logical ``KernelSpec``."""

    kernel: KernelSpec
    device_regions: tuple[DeviceRegionPlan, ...] = ()
    host_regions: tuple[HostRegionPlan, ...] = ()
    region_dependencies: tuple[RegionDependencyPlan, ...] = ()
    edge_bindings: tuple[EdgeBindingPlan, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)

    def regions_in_dependency_order(self) -> tuple[DeviceRegionPlan | HostRegionPlan, ...]:
        """Return a stable topological order of all regions."""

        regions = (*self.device_regions, *self.host_regions)
        by_name = {region.name: region for region in regions}
        order = {region.name: index for index, region in enumerate(regions)}
        incoming = {name: 0 for name in by_name}
        outgoing: dict[str, list[str]] = {name: [] for name in by_name}
        for dependency in self.region_dependencies:
            outgoing[dependency.source].append(dependency.target)
            incoming[dependency.target] += 1
        ready = sorted((name for name, count in incoming.items() if count == 0), key=order.get)
        result = []
        while ready:
            name = ready.pop(0)
            result.append(by_name[name])
            for target in outgoing[name]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
                    ready.sort(key=order.get)
        if len(result) != len(regions):
            raise ValueError("execution region dependencies must be acyclic")
        return tuple(result)

    def validate(self) -> ExecutionPlan:
        """Validate region topology, action placement, and exact edge coverage."""

        self.kernel.validate()
        if not isinstance(self.attrs, dict):
            raise TypeError("execution plan attrs must be a dict")

        regions = (*self.device_regions, *self.host_regions)
        region_names = [region.name for region in regions]
        if len(set(region_names)) != len(region_names):
            raise ValueError("execution region names must be unique")
        region_name_set = set(region_names)
        for dependency in self.region_dependencies:
            if dependency.source not in region_name_set or dependency.target not in region_name_set:
                raise ValueError("region dependency references an unknown region")
            if dependency.source == dependency.target:
                raise ValueError("region dependency cannot be self-referential")
            if dependency.kind not in ("launch_order", "completion"):
                raise ValueError(f"unsupported region dependency kind {dependency.kind!r}")
        self.regions_in_dependency_order()

        expected_edges = logical_edges(self.kernel)
        expected_by_key = {edge.key: edge for edge in expected_edges}
        binding_counts = Counter(binding.edge.key for binding in self.edge_bindings)
        missing = [str(edge) for edge in expected_edges if binding_counts[edge.key] == 0]
        duplicate = [
            str(expected_by_key[key]) for key, count in binding_counts.items() if count > 1
        ]
        unknown = [
            binding.edge
            for binding in self.edge_bindings
            if binding.edge.key not in expected_by_key
        ]
        if missing:
            raise ValueError(f"logical edges are unbound: {', '.join(missing)}")
        if duplicate:
            raise ValueError(f"logical edges have duplicate bindings: {', '.join(duplicate)}")
        if unknown:
            raise ValueError(f"edge binding references an unknown logical edge: {unknown[0]}")

        binding_by_key = {binding.edge.key: binding for binding in self.edge_bindings}
        occurrences: dict[tuple[int, str, str], list[tuple[str, Action, str | None]]] = defaultdict(
            list
        )
        kernel_tile_ids = {id(tile) for tile in self.kernel.tiles}
        for region in self.device_regions:
            if not isinstance(region.attrs, dict):
                raise TypeError(f"device region {region.name!r} attrs must be a dict")
            for action in (*region.prologue, *region.epilogue):
                for edge in _action_edges(action):
                    occurrences[edge.key].append(("region_lifecycle", action, region.name))
            for action in region.fetch_program.actions:
                for edge in _action_edges(action):
                    occurrences[edge.key].append(("fetch_guard", action, region.name))
            seen_tiles: set[int] = set()
            for program in region.tile_programs:
                if id(program.tile) not in kernel_tile_ids:
                    raise ValueError("tile action program references a tile outside this kernel")
                if id(program.tile) in seen_tiles:
                    raise ValueError(
                        f"device region {region.name!r} repeats tile program {program.tile.name!r}"
                    )
                seen_tiles.add(id(program.tile))
                for action in program.actions:
                    if isinstance(action, SmemEnterAction | SmemExitAction | RunAction):
                        if action.tile is not program.tile:
                            raise ValueError(
                                f"tile action in {program.tile.name!r} references another tile"
                            )
                    if isinstance(action, WaitAction):
                        for edge in action.edges:
                            if edge.consumer != program.tile.name:
                                raise ValueError("wait action is not in its logical consumer tile")
                    if isinstance(action, NotifyAction | QueuePushAction | MidBodyPortAction):
                        for edge in _action_edges(action):
                            if edge.producer != program.tile.name:
                                raise ValueError(
                                    "producer-side action is not in its logical producer tile"
                                )
                    placement = (
                        "mid_body_port" if isinstance(action, MidBodyPortAction) else "tile_action"
                    )
                    for edge in _action_edges(action):
                        occurrences[edge.key].append((placement, action, region.name))

        for region in self.host_regions:
            if not isinstance(region.attrs, dict):
                raise TypeError(f"host region {region.name!r} attrs must be a dict")
            for action in region.actions:
                for edge in _action_edges(action):
                    occurrences[edge.key].append(("host_runtime", action, region.name))

        for key, binding in binding_by_key.items():
            if binding.region not in region_name_set:
                raise ValueError(f"edge binding references unknown region {binding.region!r}")
            placed = occurrences.get(key, [])
            if not placed:
                raise ValueError(f"edge binding {binding.edge} has no physical action")
            for placement, action, region_name in placed:
                if placement != binding.location or region_name != binding.region:
                    raise ValueError(
                        f"edge {binding.edge} is placed at {placement} in {region_name!r}, "
                        f"not {binding.location} in {binding.region!r}"
                    )
                if binding.location == "fetch_guard" and not isinstance(action, FetchGuardAction):
                    raise ValueError("fetch-guard edge must use FetchGuardAction")
                if binding.location == "host_runtime" and not isinstance(action, HostEdgeAction):
                    raise ValueError("host/runtime edge must use HostEdgeAction")
                if binding.location == "mid_body_port":
                    if not isinstance(action, MidBodyPortAction) or action.port != binding.port:
                        raise ValueError("mid-body edge must use its approved port")

        for key, placed in occurrences.items():
            if key not in binding_by_key:
                raise ValueError(f"physical action references unbound logical edge: {placed[0][1]}")
        return self


@dataclass(frozen=True)
class EmissionContext:
    """Current traversal position supplied to a backend."""

    plan: ExecutionPlan
    region: DeviceRegionPlan | HostRegionPlan
    program: str
    tile: TileSpec | None = None


class MegakernelBackend:
    """Backend protocol consumed by ``TileEmitter``."""

    def begin_execution(self, plan: ExecutionPlan) -> None:
        """Begin an execution plan."""

    def bind_region_dependency(self, plan: ExecutionPlan, dependency: RegionDependencyPlan) -> None:
        """Bind one launch-order or completion dependency."""

        if dependency.kind == "launch_order":
            self.bind_launch_order(plan, dependency)
        else:
            self.bind_completion(plan, dependency)

    def bind_launch_order(self, plan: ExecutionPlan, dependency: RegionDependencyPlan) -> None:
        """Bind ordering between asynchronous region launches."""

    def bind_completion(self, plan: ExecutionPlan, dependency: RegionDependencyPlan) -> None:
        """Bind a dependency that requires source-region completion."""

    def begin_region(self, plan: ExecutionPlan, region: DeviceRegionPlan | HostRegionPlan) -> None:
        """Begin one region."""

        if isinstance(region, DeviceRegionPlan):
            self.begin_device_region(plan, region)
        else:
            self.begin_host_region(plan, region)

    def begin_device_region(self, plan: ExecutionPlan, region: DeviceRegionPlan) -> None:
        """Begin one device region."""

    def begin_host_region(self, plan: ExecutionPlan, region: HostRegionPlan) -> None:
        """Begin one host/runtime region."""

    def begin_action_program(self, context: EmissionContext) -> None:
        """Begin one ordered action program."""

    def emit_action(self, action: Action, context: EmissionContext) -> None:
        """Emit one action at its exact program position."""

        if isinstance(context.region, DeviceRegionPlan):
            self.emit_device_action(action, context)
        else:
            self.emit_host_action(action, context)

    def emit_device_action(self, action: Action, context: EmissionContext) -> None:
        """Emit an action owned by a device region."""

        raise NotImplementedError

    def emit_host_action(self, action: Action, context: EmissionContext) -> None:
        """Emit an action owned by a host/runtime region."""

        raise NotImplementedError

    def end_action_program(self, context: EmissionContext) -> None:
        """End one ordered action program."""

    def end_region(self, plan: ExecutionPlan, region: DeviceRegionPlan | HostRegionPlan) -> None:
        """End one region."""

        if isinstance(region, DeviceRegionPlan):
            self.end_device_region(plan, region)
        else:
            self.end_host_region(plan, region)

    def end_device_region(self, plan: ExecutionPlan, region: DeviceRegionPlan) -> None:
        """End one device region."""

    def end_host_region(self, plan: ExecutionPlan, region: HostRegionPlan) -> None:
        """End one host/runtime region."""

    def end_execution(self, plan: ExecutionPlan):
        """Finish the plan and return the backend result."""


class TileEmitter:
    """Generic traversal of region lifecycle, fetch, and tile action programs."""

    def __init__(self, backend: MegakernelBackend):
        self.backend = backend

    def emit_program(self, context: EmissionContext, actions: tuple[Action, ...]) -> None:
        """Emit one already-validated program inside a backend-owned control scope."""

        self.backend.begin_action_program(context)
        for action in actions:
            self.backend.emit_action(action, context)
        self.backend.end_action_program(context)

    def emit(self, plan: ExecutionPlan):
        plan.validate()
        self.backend.begin_execution(plan)
        for dependency in plan.region_dependencies:
            self.backend.bind_region_dependency(plan, dependency)
        for region in plan.regions_in_dependency_order():
            self.backend.begin_region(plan, region)
            if isinstance(region, DeviceRegionPlan):
                self.emit_program(EmissionContext(plan, region, "region_prologue"), region.prologue)
                self.emit_program(
                    EmissionContext(plan, region, "scheduler_fetch"),
                    region.fetch_program.actions,
                )
                for tile_program in region.tile_programs:
                    self.emit_program(
                        EmissionContext(plan, region, "tile_action", tile_program.tile),
                        tile_program.actions,
                    )
                self.emit_program(EmissionContext(plan, region, "region_epilogue"), region.epilogue)
            else:
                self.emit_program(EmissionContext(plan, region, "host"), region.actions)
            self.backend.end_region(plan, region)
        return self.backend.end_execution(plan)


def make_static_execution_plan(kernel: KernelSpec) -> ExecutionPlan:
    """Create the default single-device static action program."""

    kernel.validate()
    edges = logical_edges(kernel)
    by_producer_event: dict[tuple[str, int], list[LogicalEdge]] = defaultdict(list)
    by_consumer_event: dict[tuple[str, int], list[LogicalEdge]] = defaultdict(list)
    for edge in edges:
        by_producer_event[(edge.producer, id(edge.event))].append(edge)
        by_consumer_event[(edge.consumer, id(edge.event))].append(edge)

    unique_classes = []
    seen_classes = set()
    for tile in kernel.tiles:
        if type(tile.impl) not in seen_classes:
            seen_classes.add(type(tile.impl))
            unique_classes.append(type(tile.impl))
    prologue = tuple(HookAction("init_shared_resources", target=cls) for cls in unique_classes)
    epilogue = (
        *(HookAction("finalize_shared_resources", target=cls) for cls in reversed(unique_classes)),
        HookAction("smem_commit"),
    )

    programs = []
    for tile in kernel.tiles:
        actions: list[Action] = [
            SmemEnterAction(tile),
            HookAction("device_init", target=tile.impl),
        ]
        for event, coord_map in tile.waits:
            event_edges = tuple(by_consumer_event[(tile.name, id(event))])
            actions.append(WaitAction(event_edges, event, coord_map))
        actions.append(HookAction("prefetch", target=tile.impl))
        actions.append(RunAction(tile))
        for event, coord_map in tile.notifies:
            event_edges = tuple(by_producer_event[(tile.name, id(event))])
            actions.append(NotifyAction(event_edges, event, coord_map))
        actions.append(SmemExitAction(tile))
        programs.append(TileActionProgram(tile, tuple(actions)))

    region_name = "device"
    return ExecutionPlan(
        kernel=kernel,
        device_regions=(
            DeviceRegionPlan(
                region_name,
                prologue=prologue,
                tile_programs=tuple(programs),
                epilogue=epilogue,
                attrs={"schedule": "static"},
            ),
        ),
        edge_bindings=tuple(EdgeBindingPlan(edge, "tile_action", region_name) for edge in edges),
    ).validate()


__all__ = [
    "Action",
    "BarrierAction",
    "DeviceRegionPlan",
    "EdgeBindingPlan",
    "EmissionContext",
    "ExecutionPlan",
    "FetchGuardAction",
    "HookAction",
    "HostCallAction",
    "HostEdgeAction",
    "HostRegionPlan",
    "LogicalEdge",
    "MegakernelBackend",
    "MidBodyPortAction",
    "NotifyAction",
    "ProfileAction",
    "QueuePushAction",
    "RegionDependencyPlan",
    "RunAction",
    "RuntimeEventInitAction",
    "SchedulerFetchProgram",
    "SmemEnterAction",
    "SmemExitAction",
    "TileActionProgram",
    "TileEmitter",
    "WaitAction",
    "logical_edges",
    "make_static_execution_plan",
]
