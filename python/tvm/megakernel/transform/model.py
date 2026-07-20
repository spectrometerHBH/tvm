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
"""Execution-plan and physical-program contracts for megakernel lowering."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Literal

from ..dsl import CoordMapType, EventSpec, KernelSpec, TileSpec
from .semantic import SemanticPlan, build_semantic_plan, validate_semantic_plan

EdgeLocation = Literal["prologue", "fetch", "tile", "host", "epilogue"]
RegionDependencyKind = Literal["launch_order", "completion"]
SmemScope = Literal["none", "program", "run_to_end"]


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


@dataclass(frozen=True, kw_only=True)
class ProgramStep:
    """Base class for one physical lowering step."""

    edges: tuple[LogicalEdge, ...] = ()


@dataclass(frozen=True)
class HookStep(ProgramStep):
    """Invoke a lifecycle or implementation hook selected by the backend."""

    hook: str
    target: Any = None
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WaitStep(ProgramStep):
    """Wait for one physical event endpoint."""

    event: EventSpec
    coord_map: CoordMapType
    level: str = "cta"
    mask: int = 0xFFFFFFFF


@dataclass(frozen=True)
class NotifyStep(ProgramStep):
    """Notify one physical event endpoint."""

    event: EventSpec
    coord_map: CoordMapType
    scope: str = "cta"
    scope_id: int = 0
    count: Any = 1
    rank: int = -1
    release: bool = False


@dataclass(frozen=True)
class QueuePushStep(ProgramStep):
    """Push scheduler work, optionally as part of a logical edge binding."""


@dataclass(frozen=True)
class BarrierStep(ProgramStep):
    """Emit a barrier at its exact source-order position."""

    kind: str


@dataclass(frozen=True)
class RuntimeEventInitStep(ProgramStep):
    """Initialize a runtime-sized physical event."""

    event: Any
    value: Any
    scope: str = "thread"
    scope_id: int = 0


@dataclass(frozen=True)
class RunStep(ProgramStep):
    """Run a tile under an optional predicate, repeat, and index mapping."""

    predicate: Any = None
    repeat: Any = 1
    index_map: Any = None
    profile_event: Any = None


@dataclass(frozen=True)
class FetchGuardStep(ProgramStep):
    """Guard publication of one scheduler fetch result."""

    predicate: Any = None


@dataclass(frozen=True)
class MidBodyPortStep(ProgramStep):
    """Bind an edge to a backend-approved location inside a tile body."""

    port: str


@dataclass(frozen=True)
class HostCallStep(ProgramStep):
    """Invoke a backend-owned host operation."""

    name: str


@dataclass(frozen=True)
class HostSyncStep(ProgramStep):
    """Synchronize host/runtime work at this exact program position."""

    kind: str


@dataclass(frozen=True)
class TileProgram:
    """Ordered physical steps for one logical tile."""

    tile: TileSpec
    steps: tuple[ProgramStep, ...]
    smem_scope: SmemScope = "none"


@dataclass(frozen=True)
class DeviceRegionPlan:
    """One device region with lifecycle, fetch, and per-tile programs."""

    name: str
    prologue_steps: tuple[ProgramStep, ...] = ()
    fetch_steps: tuple[ProgramStep, ...] = ()
    tile_programs: tuple[TileProgram, ...] = ()
    epilogue_steps: tuple[ProgramStep, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HostRegionPlan:
    """One host/runtime region in the execution DAG."""

    name: str
    steps: tuple[ProgramStep, ...]
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RegionDependencyPlan:
    """Ordering between regions without conflating launch and completion."""

    source: str
    target: str
    kind: RegionDependencyKind


@dataclass(frozen=True)
class EdgePlacement:
    """The physical location derived for one logical edge."""

    edge: LogicalEdge
    location: EdgeLocation
    region: str
    port: str | None = None


def logical_edges(kernel: KernelSpec | SemanticPlan) -> tuple[LogicalEdge, ...]:
    """Return logical event edges in stable kernel/tile order."""

    semantic = kernel if isinstance(kernel, SemanticPlan) else build_semantic_plan(kernel)
    return tuple(
        LogicalEdge(edge.event, edge.producer.name, edge.consumer.name)
        for edge in semantic.logical_edges
    )


@dataclass(frozen=True)
class ExecutionPlan:
    """Complete physical plan for one logical ``KernelSpec``."""

    kernel: KernelSpec
    device_regions: tuple[DeviceRegionPlan, ...] = ()
    host_regions: tuple[HostRegionPlan, ...] = ()
    region_dependencies: tuple[RegionDependencyPlan, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)
    semantic_plan: SemanticPlan | None = field(default=None, compare=False)

    def resolved_semantic_plan(self) -> SemanticPlan:
        """Return the validated semantic input consumed by this physical plan."""

        semantic = self.semantic_plan or build_semantic_plan(self.kernel)
        if semantic.kernel is not self.kernel:
            raise ValueError("execution plan semantic plan belongs to a different KernelSpec")
        return validate_semantic_plan(semantic)

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
        """Validate region topology, step placement, and exact edge coverage."""

        semantic = self.resolved_semantic_plan()
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

        self._derive_edge_placements(semantic)
        return self

    def edge_placements(self) -> tuple[EdgePlacement, ...]:
        """Return the validated physical placement derived for every logical edge."""

        semantic = self.resolved_semantic_plan()
        self.validate()
        return self._derive_edge_placements(semantic)

    def _derive_edge_placements(
        self, semantic: SemanticPlan | None = None
    ) -> tuple[EdgePlacement, ...]:
        """Validate physical programs and derive edge placements without storing a copy."""

        expected_edges = logical_edges(semantic or self.resolved_semantic_plan())
        expected_by_key = {edge.key: edge for edge in expected_edges}
        occurrences: dict[tuple[int, str, str], list[tuple[EdgePlacement, ProgramStep]]] = (
            defaultdict(list)
        )
        unknown_edges: list[LogicalEdge] = []

        def place_step(
            step: ProgramStep,
            location: EdgeLocation,
            region_name: str,
            tile: TileSpec | None = None,
        ) -> None:
            if not isinstance(step, ProgramStep):
                raise TypeError(f"physical program contains non-step value {step!r}")
            if not isinstance(step.edges, tuple):
                raise TypeError(f"{type(step).__name__}.edges must be a tuple")
            if isinstance(step, RunStep | WaitStep | NotifyStep | QueuePushStep | MidBodyPortStep):
                if location != "tile" or tile is None:
                    raise ValueError(f"{type(step).__name__} must appear in a tile program")
            if isinstance(step, FetchGuardStep) and location != "fetch":
                raise ValueError("FetchGuardStep must appear in device fetch_steps")
            if isinstance(step, HostCallStep | HostSyncStep) and location != "host":
                raise ValueError(f"{type(step).__name__} must appear in a host region")
            if isinstance(step, HookStep) and step.hook == "host_init" and location != "prologue":
                raise ValueError("HookStep('host_init') must appear in device prologue_steps")
            if isinstance(step, WaitStep):
                for edge in step.edges:
                    if edge.consumer != tile.name:
                        raise ValueError("WaitStep is not in its logical consumer tile")
            if isinstance(step, NotifyStep | QueuePushStep | MidBodyPortStep):
                for edge in step.edges:
                    if edge.producer != tile.name:
                        raise ValueError(
                            f"{type(step).__name__} is not in its logical producer tile"
                        )
            if isinstance(step, MidBodyPortStep):
                if not isinstance(step.port, str) or not step.port:
                    raise ValueError("MidBodyPortStep.port must be a non-empty string")
                port = step.port
            else:
                port = None
            for edge in step.edges:
                if not isinstance(edge, LogicalEdge):
                    raise TypeError(f"{type(step).__name__}.edges must contain LogicalEdge values")
                placement = EdgePlacement(edge, location, region_name, port)
                occurrences[edge.key].append((placement, step))
                if edge.key not in expected_by_key:
                    unknown_edges.append(edge)

        kernel_tile_ids = {id(tile) for tile in self.kernel.tiles}
        for region in self.device_regions:
            if not isinstance(region.attrs, dict):
                raise TypeError(f"device region {region.name!r} attrs must be a dict")
            if not isinstance(region.prologue_steps, tuple):
                raise TypeError(f"device region {region.name!r} prologue_steps must be a tuple")
            if not isinstance(region.fetch_steps, tuple):
                raise TypeError(f"device region {region.name!r} fetch_steps must be a tuple")
            if not isinstance(region.tile_programs, tuple):
                raise TypeError(f"device region {region.name!r} tile_programs must be a tuple")
            if not isinstance(region.epilogue_steps, tuple):
                raise TypeError(f"device region {region.name!r} epilogue_steps must be a tuple")
            for step in region.prologue_steps:
                place_step(step, "prologue", region.name)
            for step in region.fetch_steps:
                place_step(step, "fetch", region.name)
            seen_tiles: set[int] = set()
            for program in region.tile_programs:
                if id(program.tile) not in kernel_tile_ids:
                    raise ValueError("tile program references a tile outside this kernel")
                if id(program.tile) in seen_tiles:
                    raise ValueError(
                        f"device region {region.name!r} repeats tile program {program.tile.name!r}"
                    )
                seen_tiles.add(id(program.tile))
                if not isinstance(program.steps, tuple):
                    raise TypeError(f"tile program {program.tile.name!r} steps must be a tuple")
                if program.smem_scope not in ("none", "program", "run_to_end"):
                    raise ValueError(
                        f"tile program {program.tile.name!r} has invalid smem_scope "
                        f"{program.smem_scope!r}"
                    )
                for step in program.steps:
                    place_step(step, "tile", region.name, program.tile)
            for step in region.epilogue_steps:
                place_step(step, "epilogue", region.name)

        for region in self.host_regions:
            if not isinstance(region.attrs, dict):
                raise TypeError(f"host region {region.name!r} attrs must be a dict")
            if not isinstance(region.steps, tuple):
                raise TypeError(f"host region {region.name!r} steps must be a tuple")
            for step in region.steps:
                place_step(step, "host", region.name)

        if unknown_edges:
            raise ValueError(
                f"physical step references an unknown logical edge: {unknown_edges[0]}"
            )
        missing = [edge for edge in expected_edges if edge.key not in occurrences]
        if missing:
            raise ValueError(
                f"logical edges are missing physical steps: {', '.join(map(str, missing))}"
            )

        result = []
        for edge in expected_edges:
            placed = occurrences[edge.key]
            first = placed[0][0]
            for placement, _ in placed[1:]:
                if (
                    placement.location,
                    placement.region,
                    placement.port,
                ) != (first.location, first.region, first.port):
                    raise ValueError(
                        f"edge {edge} spans physical placements "
                        f"({first.location}, {first.region!r}, {first.port!r}) and "
                        f"({placement.location}, {placement.region!r}, {placement.port!r})"
                    )
            result.append(EdgePlacement(edge, first.location, first.region, first.port))
        return tuple(result)


class ExecutionPlanBackend:
    """Backend that directly lowers a validated physical execution plan."""

    def lower(self, plan: ExecutionPlan):
        """Lower ``plan`` and return the backend-owned result."""

        raise NotImplementedError


def make_static_execution_plan(kernel: KernelSpec) -> ExecutionPlan:
    """Create the default single-device static physical program."""

    semantic = kernel.semantic_plan()
    edges = logical_edges(semantic)
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
    prologue = (
        *(HookStep("host_init", target=tile.impl) for tile in kernel.tiles),
        *(HookStep("init_shared_resources", target=cls) for cls in unique_classes),
    )
    epilogue = (
        *(HookStep("finalize_shared_resources", target=cls) for cls in reversed(unique_classes)),
        HookStep("smem_commit"),
    )

    programs = []
    for tile in kernel.tiles:
        steps: list[ProgramStep] = [HookStep("device_init", target=tile.impl)]
        for event, coord_map in tile.waits:
            event_edges = tuple(by_consumer_event[(tile.name, id(event))])
            steps.append(WaitStep(event, coord_map, edges=event_edges))
        steps.append(HookStep("prefetch", target=tile.impl))
        steps.append(RunStep())
        for event, coord_map in tile.notifies:
            event_edges = tuple(by_producer_event[(tile.name, id(event))])
            steps.append(NotifyStep(event, coord_map, edges=event_edges))
        programs.append(TileProgram(tile, tuple(steps), smem_scope="program"))

    region_name = "device"
    return ExecutionPlan(
        kernel=kernel,
        device_regions=(
            DeviceRegionPlan(
                region_name,
                prologue_steps=prologue,
                tile_programs=tuple(programs),
                epilogue_steps=epilogue,
                attrs={"schedule": "static"},
            ),
        ),
        semantic_plan=semantic,
    ).validate()


__all__ = [
    "BarrierStep",
    "DeviceRegionPlan",
    "EdgePlacement",
    "ExecutionPlan",
    "ExecutionPlanBackend",
    "FetchGuardStep",
    "HookStep",
    "HostCallStep",
    "HostRegionPlan",
    "HostSyncStep",
    "LogicalEdge",
    "MidBodyPortStep",
    "NotifyStep",
    "ProgramStep",
    "QueuePushStep",
    "RegionDependencyPlan",
    "RunStep",
    "RuntimeEventInitStep",
    "TileProgram",
    "WaitStep",
    "logical_edges",
    "make_static_execution_plan",
]
