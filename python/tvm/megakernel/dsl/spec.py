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
"""Core building blocks for the megakernel DSL.

The DSL records the logical specification of a megakernel: tensors, events,
tile instances, and wait/notify relationships.  It stays independent from the
concrete CUDA/TIRX/runtime implementation chosen by later lowering passes.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .impl import TileImpl


@dataclass(frozen=True)
class VarSpec:
    """Symbolic variable used in shapes, tile counts, and event counts."""

    name: str
    dtype: str = "int32"


ExprLike = int | VarSpec
ShapeType = ExprLike | tuple[ExprLike, ...] | list[ExprLike]
CoordMapType = Callable[[int, int, int], tuple[int, ...]] | tuple[int, ...] | list[int]
TileNumType = tuple[ExprLike, ExprLike, ExprLike] | list[ExprLike]


@dataclass(frozen=True)
class TensorSpec:
    """Logical tensor or buffer in the megakernel spec."""

    name: str
    shape: ShapeType
    dtype: str


@dataclass(frozen=True)
class EventSpec:
    """Logical readiness event independent of its physical implementation."""

    name: str
    shape: ShapeType
    init_count: int | Callable[[tuple[int, ...]], int]
    dtype: str = "int32"
    attrs: dict[str, Any] = field(default_factory=dict)


DependencyType = tuple[EventSpec, CoordMapType]


@dataclass
class TileSpec:
    """One logical tile stage in the megakernel spec.

    ``tile_num`` always uses the three axes ``(m, n, k)``.  Extent ``1`` is
    used for an axis that is not meaningful to a tile implementation.
    """

    name: str
    impl: TileImpl
    tile_num: TileNumType
    reads: list[TensorSpec] = field(default_factory=list)
    writes: list[TensorSpec] = field(default_factory=list)
    waits: list[DependencyType] = field(default_factory=list)
    notifies: list[DependencyType] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)

    def wait(self, event: EventSpec, coord_map: CoordMapType):
        """Declare that this tile waits on ``event`` at ``coord_map``."""

        self.waits.append((event, coord_map))
        return self

    def notify(self, event: EventSpec, coord_map: CoordMapType):
        """Declare that this tile notifies ``event`` at ``coord_map``."""

        self.notifies.append((event, coord_map))
        return self


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _shape_items(
    shape: ShapeType,
    *,
    label: str,
    var_ids: set[int] | None = None,
) -> tuple[ExprLike, ...]:
    if isinstance(shape, int | VarSpec) and not isinstance(shape, bool):
        values = (shape,)
    elif isinstance(shape, tuple | list):
        values = tuple(shape)
    else:
        raise TypeError(f"{label} must be an int, VarSpec, tuple, or list")

    for extent in values:
        if _is_int(extent):
            if extent <= 0:
                raise ValueError(f"{label} extents must be positive")
        elif isinstance(extent, VarSpec):
            if not extent.name:
                raise ValueError(f"{label} VarSpec names must be non-empty")
            if var_ids is not None and id(extent) not in var_ids:
                raise ValueError(f"{label} references a VarSpec outside this kernel")
        else:
            raise TypeError(f"{label} extents must be int or VarSpec")
    return values


def _bind_callable(func: Callable[..., Any], args: tuple[Any, ...], *, label: str) -> None:
    try:
        inspect.signature(func).bind(*args)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{label} has an invalid callable signature") from err


def _call_twice(func: Callable[..., Any], args: tuple[Any, ...], *, label: str) -> Any:
    try:
        first = func(*args)
        second = func(*args)
    except Exception as err:  # pylint: disable=broad-exception-caught
        raise ValueError(f"{label} failed during validation") from err
    if type(first) is not type(second) or first != second:  # pylint: disable=unidiomatic-typecheck
        raise ValueError(f"{label} must be a pure deterministic callable")
    return first


def _validate_coord_map(coord_map: CoordMapType, *, rank: int, label: str) -> None:
    if callable(coord_map):
        args = (0, 0, 0)
        _bind_callable(coord_map, args, label=label)
        coordinates = _call_twice(coord_map, args, label=label)
        other = _call_twice(coord_map, (1, 2, 3), label=label)
        if type(coordinates) is not type(other):  # pylint: disable=unidiomatic-typecheck
            raise ValueError(f"{label} must return a stable coordinate type")
    else:
        coordinates = coord_map

    if not isinstance(coordinates, tuple | list):
        raise ValueError(f"{label} must return a tuple or list")
    if len(coordinates) != rank:
        raise ValueError(f"{label} rank {len(coordinates)} does not match event rank {rank}")
    if not all(_is_int(value) for value in coordinates):
        raise TypeError(f"{label} coordinates must be integers")


def _validate_init_count(event: EventSpec, *, rank: int) -> None:
    label = f"event {event.name!r} init_count"
    if _is_int(event.init_count):
        if event.init_count <= 0:
            raise ValueError(f"{label} must be positive")
        return
    if not callable(event.init_count):
        raise TypeError(f"{label} must be a positive integer or callable")

    args = ((0,) * rank,)
    _bind_callable(event.init_count, args, label=label)
    values = (
        _call_twice(event.init_count, args, label=label),
        _call_twice(event.init_count, ((1,) * rank,), label=label),
    )
    if not all(_is_int(value) and value > 0 for value in values):
        raise ValueError(f"{label} must return a positive integer")


class KernelSpec:
    """Container and fluent builder for one logical megakernel specification."""

    def __init__(self, name: str, attrs: dict[str, Any] | None = None):
        self.name = name
        self.attrs = attrs or {}
        self.vars: dict[str, VarSpec] = {}
        self.tensors: dict[str, TensorSpec] = {}
        self.events: dict[str, EventSpec] = {}
        self.tiles: list[TileSpec] = []

    def var(self, name: str, dtype: str = "int32"):
        """Register a symbolic variable."""

        if name in self.vars:
            raise ValueError(f"Duplicate var: {name}")
        var = VarSpec(name=name, dtype=dtype)
        self.vars[name] = var
        return var

    def tensor(self, name: str, shape: ShapeType, dtype: str):
        """Register a logical tensor."""

        if name in self.tensors:
            raise ValueError(f"Duplicate tensor: {name}")
        tensor = TensorSpec(name=name, shape=shape, dtype=dtype)
        self.tensors[name] = tensor
        return tensor

    def event(
        self,
        name: str,
        shape: ShapeType,
        init_count: int | Callable[[tuple[int, ...]], int],
        dtype: str = "int32",
        attrs: dict[str, Any] | None = None,
    ):
        """Register a logical event."""

        if name in self.events:
            raise ValueError(f"Duplicate event: {name}")
        if init_count is None:
            raise ValueError("event init_count must be specified")
        event = EventSpec(
            name=name,
            shape=shape,
            init_count=init_count,
            dtype=dtype,
            attrs=attrs or {},
        )
        self.events[name] = event
        return event

    def tile(
        self,
        name: str,
        impl: TileImpl,
        tile_num: TileNumType,
        reads: list[TensorSpec] | None = None,
        writes: list[TensorSpec] | None = None,
        attrs: dict[str, Any] | None = None,
    ):
        """Register one tile stage."""

        if any(tile.name == name for tile in self.tiles):
            raise ValueError(f"Duplicate tile: {name}")
        tile = TileSpec(
            name=name,
            impl=impl,
            tile_num=tile_num,
            reads=list(reads or ()),
            writes=list(writes or ()),
            attrs=attrs or {},
        )
        self.tiles.append(tile)
        return tile

    def validate(self):
        """Validate the implementation-independent logical specification."""

        if not isinstance(self.attrs, dict):
            raise TypeError("kernel attrs must be a dict")

        var_ids = {id(var) for var in self.vars.values()}
        tensor_ids = {id(tensor) for tensor in self.tensors.values()}
        event_ids = {id(event) for event in self.events.values()}
        event_ranks: dict[int, int] = {}
        for name, var in self.vars.items():
            if var.name != name:
                raise ValueError(f"var registry name mismatch: {name}")
            if not var.name:
                raise ValueError("VarSpec names must be non-empty")
            if not isinstance(var.dtype, str) or not var.dtype:
                raise TypeError(f"var {name!r} dtype must be a non-empty string")

        for name, tensor in self.tensors.items():
            if tensor.name != name:
                raise ValueError(f"tensor registry name mismatch: {name}")
            _shape_items(tensor.shape, label=f"tensor {name!r} shape", var_ids=var_ids)

        for name, event in self.events.items():
            if event.name != name:
                raise ValueError(f"event registry name mismatch: {name}")
            rank = len(_shape_items(event.shape, label=f"event {name!r} shape", var_ids=var_ids))
            event_ranks[id(event)] = rank
            _validate_init_count(event, rank=rank)
            if not isinstance(event.attrs, dict):
                raise TypeError(f"event {name!r} attrs must be a dict")

        tile_names: set[str] = set()
        notifiers: dict[int, list[str]] = {event_id: [] for event_id in event_ids}
        waiters: list[tuple[str, EventSpec]] = []
        tensor_producers: dict[int, str] = {}
        tensor_consumers: dict[int, list[str]] = {tensor_id: [] for tensor_id in tensor_ids}
        for tile in self.tiles:
            if tile.name in tile_names:
                raise ValueError(f"Duplicate tile: {tile.name}")
            tile_names.add(tile.name)
            if not isinstance(tile.impl, TileImpl) or inspect.isabstract(type(tile.impl)):
                raise TypeError(f"tile {tile.name!r} impl must be a concrete TileImpl instance")
            if not isinstance(tile.tile_num, tuple | list) or len(tile.tile_num) != 3:
                raise ValueError(f"tile {tile.name!r} tile_num must contain exactly three axes")
            _shape_items(
                tile.tile_num,
                label=f"tile {tile.name!r} tile_num",
                var_ids=var_ids,
            )
            if not isinstance(tile.attrs, dict):
                raise TypeError(f"tile {tile.name!r} attrs must be a dict")

            for tensor in tile.reads:
                if not isinstance(tensor, TensorSpec) or id(tensor) not in tensor_ids:
                    raise ValueError(f"tile {tile.name!r} references a tensor outside this kernel")
                tensor_consumers[id(tensor)].append(tile.name)
            for tensor in tile.writes:
                if not isinstance(tensor, TensorSpec) or id(tensor) not in tensor_ids:
                    raise ValueError(f"tile {tile.name!r} references a tensor outside this kernel")
                previous = tensor_producers.setdefault(id(tensor), tile.name)
                if previous != tile.name:
                    raise ValueError(f"tensor {tensor.name!r} has multiple producers")
            for kind, dependencies in (("wait", tile.waits), ("notify", tile.notifies)):
                for dependency in dependencies:
                    if not isinstance(dependency, tuple) or len(dependency) != 2:
                        raise TypeError(f"tile {tile.name!r} has an invalid {kind} dependency")
                    event, coord_map = dependency
                    if not isinstance(event, EventSpec) or id(event) not in event_ids:
                        raise ValueError(
                            f"tile {tile.name!r} references an event outside this kernel"
                        )
                    _validate_coord_map(
                        coord_map,
                        rank=event_ranks[id(event)],
                        label=f"tile {tile.name!r} {kind} coord_map",
                    )
                    if kind == "notify":
                        notifiers[id(event)].append(tile.name)
                    else:
                        waiters.append((tile.name, event))

        edges = {name: set() for name in tile_names}
        for waiter, event in waiters:
            producers = notifiers[id(event)]
            if not producers:
                raise ValueError(f"event {event.name!r} is waited on but has no notifier")
            for producer in producers:
                edges[producer].add(waiter)
        for tensor_id, producer in tensor_producers.items():
            for consumer in tensor_consumers[tensor_id]:
                if consumer != producer:
                    edges[producer].add(consumer)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(tile_name: str) -> None:
            if tile_name in visiting:
                raise ValueError("logical dependencies must be acyclic")
            if tile_name in visited:
                return
            visiting.add(tile_name)
            for consumer in edges[tile_name]:
                visit(consumer)
            visiting.remove(tile_name)
            visited.add(tile_name)

        for tile_name in tile_names:
            visit(tile_name)
        return self

    def lower(self, options=None):
        """Lower this graph with the default single-device static policy."""

        from tvm.megakernel.transform import lower_to_tirx

        return lower_to_tirx(self, options)
