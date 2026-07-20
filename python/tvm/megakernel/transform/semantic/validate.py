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
"""Semantic validation for logical megakernel graphs."""

from __future__ import annotations

import inspect
import math
from collections import defaultdict
from collections.abc import Iterable
from itertools import product
from typing import Any

from ...dsl import (
    EventSpec,
    ExprSpec,
    RegionRange,
    RegionSpec,
    TensorSpec,
    TileImpl,
    VarSpec,
    eval_expr_like,
    expr_bounds,
)
from .build import event_init_count, semantic_edges
from .model import SemanticPlan

_EXACT_ENUMERATION_LIMIT = 262_144
_SYMBOLIC_ENVIRONMENT_LIMIT = 4_096
_REGION_ENUMERATION_LIMIT = 65_536
_EXPR_ARITY = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "floordiv": 2,
    "mod": 2,
    "neg": 1,
    "ceildiv": 2,
}


def validate_semantic_plan(plan: SemanticPlan) -> SemanticPlan:
    """Validate the backend-independent meaning of a megakernel graph."""

    if not isinstance(plan, SemanticPlan):
        raise TypeError("semantic validation requires a SemanticPlan")
    kernel = plan.kernel
    if not isinstance(kernel.attrs, dict):
        raise TypeError("kernel attrs must be a dict")
    _validate_plan_snapshot(plan)

    var_ids = {id(var) for var in plan.vars}
    tensor_ids = {id(tensor) for tensor in plan.tensors}
    event_ids = {id(event) for event in plan.events}
    event_ranks: dict[int, int] = {}

    for registry_name, var in kernel.vars.items():
        if var.name != registry_name:
            raise ValueError(f"var registry name mismatch: {registry_name}")
        if not var.name:
            raise ValueError("VarSpec names must be non-empty")
        if not isinstance(var.dtype, str) or not var.dtype:
            raise TypeError(f"var {registry_name!r} dtype must be a non-empty string")
        _validate_var_range(var)

    for registry_name, tensor in kernel.tensors.items():
        if tensor.name != registry_name:
            raise ValueError(f"tensor registry name mismatch: {registry_name}")
        if tensor.base is not None or tensor.has_region:
            raise ValueError(f"registered tensor {registry_name!r} cannot be a region view")
        _shape_items(tensor.shape, f"tensor {registry_name!r} shape", var_ids)
        if not isinstance(tensor.dtype, str) or not tensor.dtype:
            raise TypeError(f"tensor {registry_name!r} dtype must be a non-empty string")

    for registry_name, event in kernel.events.items():
        if event.name != registry_name:
            raise ValueError(f"event registry name mismatch: {registry_name}")
        rank = len(_shape_items(event.shape, f"event {registry_name!r} shape", var_ids))
        event_ranks[id(event)] = rank
        _validate_init_count(event, rank)
        if not isinstance(event.dtype, str) or not event.dtype:
            raise TypeError(f"event {registry_name!r} dtype must be a non-empty string")
        if not isinstance(event.attrs, dict):
            raise TypeError(f"event {registry_name!r} attrs must be a dict")

    producers: dict[int, list] = defaultdict(list)
    consumers: dict[int, list] = defaultdict(list)
    tensor_writers: dict[int, list[tuple[Any, TensorSpec]]] = defaultdict(list)
    tensor_readers: dict[int, list[tuple[Any, TensorSpec]]] = defaultdict(list)
    tile_names: set[str] = set()
    for tile in plan.tiles:
        if tile.name in tile_names:
            raise ValueError(f"Duplicate tile: {tile.name}")
        tile_names.add(tile.name)
        if not isinstance(tile.impl, TileImpl) or inspect.isabstract(type(tile.impl)):
            raise TypeError(f"tile {tile.name!r} impl must be a concrete TileImpl instance")
        if not isinstance(tile.tile_num, tuple | list) or len(tile.tile_num) != 3:
            raise ValueError(f"tile {tile.name!r} tile_num must contain exactly three axes")
        _shape_items(tile.tile_num, f"tile {tile.name!r} tile_num", var_ids)
        if not isinstance(tile.attrs, dict):
            raise TypeError(f"tile {tile.name!r} attrs must be a dict")

        for access in tile.reads:
            _validate_tensor_access(tile, access, tensor_ids, var_ids, "read")
            tensor_readers[id(access.base_tensor)].append((tile, access))
        for access in tile.writes:
            _validate_tensor_access(tile, access, tensor_ids, var_ids, "write")
            tensor_writers[id(access.base_tensor)].append((tile, access))

        for kind, dependencies in (("wait", tile.waits), ("notify", tile.notifies)):
            seen_events: set[int] = set()
            for dependency in dependencies:
                if not isinstance(dependency, tuple) or len(dependency) != 2:
                    raise TypeError(f"tile {tile.name!r} has an invalid {kind} dependency")
                event, coord_map = dependency
                if not isinstance(event, EventSpec) or id(event) not in event_ids:
                    raise ValueError(f"tile {tile.name!r} references an event outside this kernel")
                if id(event) in seen_events:
                    raise ValueError(
                        f"tile {tile.name!r} {kind}s event {event.name!r} more than once"
                    )
                seen_events.add(id(event))
                _validate_coord_map(
                    coord_map,
                    rank=event_ranks[id(event)],
                    label=f"tile {tile.name!r} {kind} coord_map",
                    var_ids=var_ids,
                )
                targets = producers if kind == "notify" else consumers
                targets[id(event)].append(tile)

    _validate_tensor_access_regions(plan)
    _validate_tensor_writers(tensor_writers, plan)
    for event in plan.events:
        if consumers[id(event)] and not producers[id(event)]:
            raise ValueError(f"event {event.name!r} is waited on but has no notifier")
        _validate_event_counts(event, producers[id(event)], consumers[id(event)], plan.vars)

    event_adjacency = _event_adjacency(plan)
    _validate_dependencies_acyclic(plan, tensor_writers, tensor_readers, event_adjacency)
    event_ordering = _transitive_closure(event_adjacency)
    _validate_tensor_dependencies(plan, tensor_writers, tensor_readers, event_ordering)
    return plan


def _validate_plan_snapshot(plan: SemanticPlan) -> None:
    kernel = plan.kernel
    snapshots = (
        (plan.vars, tuple(kernel.vars.values()), "variables"),
        (plan.tensors, tuple(kernel.tensors.values()), "tensors"),
        (plan.events, tuple(kernel.events.values()), "events"),
        (plan.tiles, tuple(kernel.tiles), "tiles"),
    )
    for actual, expected, label in snapshots:
        if len(actual) != len(expected) or any(
            lhs is not rhs for lhs, rhs in zip(actual, expected)
        ):
            raise ValueError(f"semantic plan {label} do not match its KernelSpec")
    if plan.logical_edges != semantic_edges(kernel):
        raise ValueError("semantic plan logical edges do not match its KernelSpec")


def _validate_var_range(var: VarSpec) -> None:
    if var.range is None:
        return
    if (
        not isinstance(var.range, tuple)
        or len(var.range) != 2
        or any(not _is_int(value) for value in var.range)
    ):
        raise TypeError(f"var {var.name!r} range must be a tuple of two integers")
    if var.range[0] <= 0 or var.range[0] > var.range[1]:
        raise ValueError(f"var {var.name!r} range must satisfy 0 < minimum <= maximum")


def _shape_items(shape: Any, label: str, var_ids: set[int]) -> tuple[Any, ...]:
    if _is_expr_like(shape):
        values = (shape,)
    elif isinstance(shape, tuple | list):
        values = tuple(shape)
    else:
        raise TypeError(f"{label} must be an int, VarSpec, ExprSpec, tuple, or list")
    if not values:
        raise ValueError(f"{label} must have at least one extent")
    for extent in values:
        _validate_expr(extent, label, var_ids)
        bounds = expr_bounds(extent, require_bounded=False)
        if bounds is not None and bounds[0] <= 0:
            raise ValueError(f"{label} extents must be positive")
    return values


def _validate_expr(value: Any, label: str, var_ids: set[int]) -> None:
    if _is_int(value):
        return
    if isinstance(value, VarSpec):
        if not value.name:
            raise ValueError(f"{label} VarSpec names must be non-empty")
        if id(value) not in var_ids:
            raise ValueError(f"{label} references a VarSpec outside this kernel")
        return
    if isinstance(value, ExprSpec):
        expected_arity = _EXPR_ARITY.get(value.op)
        if expected_arity is None:
            raise ValueError(f"{label} uses unsupported ExprSpec op {value.op!r}")
        if len(value.args) != expected_arity:
            raise ValueError(f"{label} ExprSpec op {value.op!r} expects {expected_arity} arguments")
        for arg in value.args:
            _validate_expr(arg, label, var_ids)
        expr_bounds(value, require_bounded=False)
        return
    raise TypeError(f"{label} extents must be int, VarSpec, or ExprSpec")


def _validate_init_count(event: EventSpec, rank: int) -> None:
    label = f"event {event.name!r} init_count"
    if _is_int(event.init_count):
        if event.init_count <= 0:
            raise ValueError(f"{label} must be positive")
        return
    if not callable(event.init_count):
        raise TypeError(f"{label} must be a positive integer or callable")
    args = ((0,) * rank,)
    _bind_callable(event.init_count, args, label)
    values = (
        _call_twice(event.init_count, args, label),
        _call_twice(event.init_count, ((1,) * rank,), label),
    )
    if not all(_is_int(value) and value > 0 for value in values):
        raise ValueError(f"{label} must return a positive integer")


def _validate_coord_map(coord_map, rank: int, label: str, var_ids: set[int]) -> None:
    coordinates = _call_coord_map(coord_map, (0, 0, 0), label)
    other = _call_coord_map(coord_map, (1, 2, 3), label)
    if type(coordinates) is not type(other):  # pylint: disable=unidiomatic-typecheck
        raise ValueError(f"{label} must return a stable coordinate type")
    if not isinstance(coordinates, tuple | list):
        raise ValueError(f"{label} must return a tuple or list")
    if len(coordinates) != rank:
        raise ValueError(f"{label} rank {len(coordinates)} does not match event rank {rank}")
    for coordinate in coordinates:
        if not _is_expr_like(coordinate):
            raise TypeError(f"{label} coordinates must be integers or logical expressions")
        _validate_expr(coordinate, label, var_ids)


def _call_coord_map(coord_map, indices: tuple[int, int, int], label: str):
    if not callable(coord_map):
        return coord_map
    _bind_callable(coord_map, indices, label)
    return _call_twice(coord_map, indices, label)


def _bind_callable(func, args: tuple[Any, ...], label: str) -> None:
    try:
        inspect.signature(func).bind(*args)
    except (TypeError, ValueError) as err:
        raise ValueError(f"{label} has an invalid callable signature") from err


def _call_twice(func, args: tuple[Any, ...], label: str):
    try:
        first = func(*args)
        second = func(*args)
    except Exception as err:  # pylint: disable=broad-exception-caught
        raise ValueError(f"{label} failed during validation") from err
    if type(first) is not type(second) or first != second:  # pylint: disable=unidiomatic-typecheck
        raise ValueError(f"{label} must be a pure deterministic callable")
    return first


def _validate_tensor_access(
    tile, access: TensorSpec, tensor_ids: set[int], var_ids: set[int], kind: str
) -> None:
    if not isinstance(access, TensorSpec) or id(access.base_tensor) not in tensor_ids:
        raise ValueError(f"tile {tile.name!r} references a tensor outside this kernel")
    if not access.has_region:
        return
    if access.region_dynamic:
        if access.region_map is not None:
            raise ValueError("dynamic tensor region cannot also provide a region map")
        if not isinstance(access.region_reason, str) or not access.region_reason.strip():
            raise ValueError("dynamic tensor region requires a non-empty reason")
        return
    label = f"tile {tile.name!r} {kind} region"
    region = _call_region_map(access.region_map, (0, 0, 0), label)
    other = _call_region_map(access.region_map, (1, 2, 3), label)
    if type(region) is not type(other):  # pylint: disable=unidiomatic-typecheck
        raise ValueError(f"{label} must return a stable region type")
    region = _normalize_region(region)
    if region.dynamic:
        raise ValueError(
            f"{label} must declare dynamic=True and a non-empty reason on TensorSpec.region"
        )
    tensor_rank = len(_shape_tuple(access.base_tensor.shape))
    if len(region.dims) != tensor_rank:
        raise ValueError(
            f"{label} rank {len(region.dims)} does not match tensor "
            f"{access.base_tensor.name!r} rank {tensor_rank}"
        )
    for dimension in region.dims:
        _validate_expr(dimension.start, label, var_ids)
        _validate_expr(dimension.extent, label, var_ids)


def _call_region_map(region_map, indices: tuple[int, int, int], label: str):
    if not callable(region_map):
        return region_map
    _bind_callable(region_map, indices, label)
    return _call_twice(region_map, indices, label)


def _normalize_region(value: Any) -> RegionSpec:
    if isinstance(value, RegionSpec):
        return value
    if isinstance(value, tuple | list):
        return RegionSpec(tuple(RegionRange(index, 1) for index in value))
    raise TypeError(f"region map must return RegionSpec, tuple, or list, got {value!r}")


def _validate_tensor_writers(writers, plan: SemanticPlan) -> None:
    for tensor in plan.tensors:
        entries = writers[id(tensor)]
        distinct_tiles = []
        for tile, access in entries:
            if all(candidate is not tile for candidate in distinct_tiles):
                distinct_tiles.append(tile)
            if len(distinct_tiles) <= 1:
                continue
            if not access.has_region or access.region_dynamic:
                raise ValueError(f"tensor {tensor.name!r} has multiple producers")
        if len(distinct_tiles) <= 1:
            continue
        if any(not access.has_region or access.region_dynamic for _, access in entries):
            raise ValueError(f"tensor {tensor.name!r} has multiple producers")
        _validate_writer_regions_disjoint(tensor, entries, plan.vars)


def _validate_tensor_access_regions(plan: SemanticPlan) -> None:
    static_accesses = [
        (tile, access, kind)
        for tile in plan.tiles
        for kind, accesses in (("read", tile.reads), ("write", tile.writes))
        for access in accesses
        if access.has_region and not access.region_dynamic
    ]
    if not static_accesses:
        return
    environments = _bounded_environments(
        plan.vars,
        "static tensor-region bounds validation",
        require_bounded=True,
    )
    for env in environments:
        for tile, access, kind in static_accesses:
            tensor = access.base_tensor
            shape = _resolve_tuple(tensor.shape, env)
            if shape is None:
                raise ValueError(f"tensor {tensor.name!r} shape cannot be resolved")
            label = f"tile {tile.name!r} {kind}-region validation"
            for indices in _tile_indices(tile.tile_num, env, label):
                region = _resolve_access_region(access, indices, env)
                _validate_region_bounds(tensor, region, shape, label)


def _validate_writer_regions_disjoint(tensor, writers, variables) -> None:
    environments = _bounded_environments(
        variables,
        f"tensor {tensor.name!r} writer-region validation",
        require_bounded=True,
    )
    for env in environments:
        shape = _resolve_tuple(tensor.shape, env)
        if shape is None:
            raise ValueError(f"tensor {tensor.name!r} shape cannot be resolved")
        for index, (lhs_tile, lhs_access) in enumerate(writers):
            lhs_indices = _tile_indices(
                lhs_tile.tile_num,
                env,
                f"tile {lhs_tile.name!r} writer-region validation",
            )
            for rhs_tile, rhs_access in writers[index + 1 :]:
                if lhs_tile is rhs_tile:
                    continue
                rhs_indices = _tile_indices(
                    rhs_tile.tile_num,
                    env,
                    f"tile {rhs_tile.name!r} writer-region validation",
                )
                label = (
                    f"tensor {tensor.name!r} writer regions for "
                    f"{lhs_tile.name!r} and {rhs_tile.name!r}"
                )
                for lhs_idx, rhs_idx in _region_pairs(lhs_indices, rhs_indices, label):
                    lhs = _resolve_access_region(lhs_access, lhs_idx, env)
                    rhs = _resolve_access_region(rhs_access, rhs_idx, env)
                    _validate_region_bounds(tensor, lhs, shape, f"{lhs_tile.name}.write_region")
                    _validate_region_bounds(tensor, rhs, shape, f"{rhs_tile.name}.write_region")
                    if _regions_overlap(lhs, rhs):
                        raise ValueError(
                            f"tensor {tensor.name!r} has multiple producers with overlapping "
                            f"regions: {lhs_tile.name!r} {_region_label(lhs)} and "
                            f"{rhs_tile.name!r} {_region_label(rhs)}"
                        )


def _validate_event_counts(event, producers, consumers, variables) -> None:
    if not producers:
        return
    variables = tuple(variables)
    environments = (
        _bounded_environments(
            variables,
            f"event {event.name!r} count validation",
            require_bounded=False,
        )
        if _event_requires_environment(event, producers, consumers)
        else (None,)
    )
    for env in environments:
        event_shape = _resolve_tuple(event.shape, env)
        if event_shape is None:
            raise ValueError(f"event {event.name!r} shape cannot be resolved")
        producer_extents = [_resolve_tuple(tile.tile_num, env) for tile in producers]
        consumer_extents = [_resolve_tuple(tile.tile_num, env) for tile in consumers]
        if any(extents is None for extents in (*producer_extents, *consumer_extents)):
            raise ValueError(f"event {event.name!r} tile extents cannot be resolved")
        enumeration_size = math.prod(event_shape) + sum(
            math.prod(extents) for extents in (*producer_extents, *consumer_extents)
        )
        if enumeration_size > _EXACT_ENUMERATION_LIMIT:
            raise ValueError(
                f"event {event.name!r} exact semantic validation needs "
                f"{enumeration_size} points, exceeding the limit "
                f"{_EXACT_ENUMERATION_LIMIT}"
            )
        counts: dict[tuple[int, ...], int] = defaultdict(int)
        for tile, extents in zip(producers, producer_extents):
            for indices in product(*(range(extent) for extent in extents)):
                for notify_event, coord_map in tile.notifies:
                    if notify_event is not event:
                        continue
                    coord = _resolve_coord(coord_map, indices, env)
                    _validate_static_coord(event, coord, event_shape, f"{tile.name}.notify")
                    counts[coord] += 1
        for coord in product(*(range(extent) for extent in event_shape)):
            expected = event_init_count(event, coord)
            actual = counts.get(coord, 0)
            if actual != expected:
                raise ValueError(
                    f"event {event.name!r} coord {coord} expects init_count {expected}, "
                    f"but has {actual} static notifies"
                )
        for tile, extents in zip(consumers, consumer_extents):
            for indices in product(*(range(extent) for extent in extents)):
                for wait_event, coord_map in tile.waits:
                    if wait_event is not event:
                        continue
                    coord = _resolve_coord(coord_map, indices, env)
                    _validate_static_coord(event, coord, event_shape, f"{tile.name}.wait")
                    if counts.get(coord, 0) == 0:
                        raise ValueError(
                            f"tile {tile.name!r} waits on event {event.name!r} coord "
                            f"{coord} without a producer notify"
                        )


def _event_requires_environment(event, producers, consumers) -> bool:
    event_shape = _resolve_tuple(event.shape, None)
    tile_extents = [_resolve_tuple(tile.tile_num, None) for tile in (*producers, *consumers)]
    if event_shape is None or any(extents is None for extents in tile_extents):
        return True
    point_count = sum(math.prod(extents) for extents in tile_extents)
    if point_count > _EXACT_ENUMERATION_LIMIT:
        return False
    for tile, extents in zip((*producers, *consumers), tile_extents):
        dependencies = (*tile.notifies, *tile.waits)
        for indices in product(*(range(extent) for extent in extents)):
            for dependency_event, coord_map in dependencies:
                if dependency_event is not event:
                    continue
                coord = _resolve_coord(coord_map, indices, None)
                if any(not _is_int(value) for value in coord):
                    return True
    return False


def _event_adjacency(plan: SemanticPlan) -> dict[int, set[int]]:
    adjacency = {id(tile): set() for tile in plan.tiles}
    for edge in plan.logical_edges:
        if edge.producer is not edge.consumer:
            adjacency[id(edge.producer)].add(id(edge.consumer))
    return adjacency


def _transitive_closure(adjacency: dict[int, set[int]]) -> dict[int, set[int]]:
    closure = {}
    for source, direct_consumers in adjacency.items():
        reachable = set()
        pending = list(direct_consumers)
        while pending:
            current = pending.pop()
            if current in reachable:
                continue
            reachable.add(current)
            pending.extend(adjacency[current] - reachable)
        closure[source] = reachable
    return closure


def _validate_dependencies_acyclic(plan, tensor_writers, tensor_readers, event_adjacency) -> None:
    edges = {tile_id: set(consumers) for tile_id, consumers in event_adjacency.items()}
    for tensor in plan.tensors:
        for producer, _ in tensor_writers[id(tensor)]:
            for consumer, _ in tensor_readers[id(tensor)]:
                if producer is not consumer:
                    edges[id(producer)].add(id(consumer))

    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(tile_id: int) -> None:
        if tile_id in visiting:
            raise ValueError("logical dependencies must be acyclic")
        if tile_id in visited:
            return
        visiting.add(tile_id)
        for consumer in edges[tile_id]:
            visit(consumer)
        visiting.remove(tile_id)
        visited.add(tile_id)

    for tile_id in edges:
        visit(tile_id)


def _validate_tensor_dependencies(plan, writers, readers, event_ordering) -> None:
    if not any(access.has_region for tile in plan.tiles for access in (*tile.reads, *tile.writes)):
        for tensor in plan.tensors:
            for producer, _ in writers[id(tensor)]:
                for consumer, _ in readers[id(tensor)]:
                    if producer is consumer:
                        continue
                    if not _tiles_ordered_by_events(event_ordering, producer, consumer):
                        raise ValueError(
                            f"tile {consumer.name!r} reads tensor {tensor.name!r} written "
                            f"by tile {producer.name!r} without an event dependency"
                        )
        return

    dynamic_pairs = set()
    has_static_pairs = False
    for tensor in plan.tensors:
        for producer, write_access in writers[id(tensor)]:
            for consumer, read_access in readers[id(tensor)]:
                if producer is consumer:
                    continue
                if write_access.region_dynamic or read_access.region_dynamic:
                    if not _tiles_ordered_by_events(event_ordering, producer, consumer):
                        raise ValueError(
                            f"tile {consumer.name!r} dynamically reads tensor "
                            f"{tensor.name!r} written by tile {producer.name!r} without "
                            "an event dependency"
                        )
                    dynamic_pairs.add((id(tensor), id(producer), id(consumer)))
                else:
                    has_static_pairs = True

    if not has_static_pairs:
        return

    environments = _bounded_environments(
        plan.vars,
        "static tensor-region dependency validation",
        require_bounded=True,
    )
    for env in environments:
        for tensor in plan.tensors:
            shape = _resolve_tuple(tensor.shape, env)
            if shape is None:
                raise ValueError(f"tensor {tensor.name!r} shape cannot be resolved")
            for producer, write_access in writers[id(tensor)]:
                for consumer, read_access in readers[id(tensor)]:
                    if producer is consumer:
                        continue
                    pair_key = (id(tensor), id(producer), id(consumer))
                    if pair_key in dynamic_pairs:
                        continue
                    producer_indices = _tile_indices(
                        producer.tile_num,
                        env,
                        f"tile {producer.name!r} write-region validation",
                    )
                    consumer_indices = _tile_indices(
                        consumer.tile_num,
                        env,
                        f"tile {consumer.name!r} read-region validation",
                    )
                    label = (
                        f"tensor {tensor.name!r} dependency regions for "
                        f"{producer.name!r} and {consumer.name!r}"
                    )
                    for producer_idx, consumer_idx in _region_pairs(
                        producer_indices, consumer_indices, label
                    ):
                        write_region = _resolve_access_region(write_access, producer_idx, env)
                        read_region = _resolve_access_region(read_access, consumer_idx, env)
                        _validate_region_bounds(
                            tensor, write_region, shape, f"{producer.name}.write_region"
                        )
                        _validate_region_bounds(
                            tensor, read_region, shape, f"{consumer.name}.read_region"
                        )
                        if not _regions_overlap(write_region, read_region):
                            continue
                        if _matching_event_coord(
                            producer, producer_idx, consumer, consumer_idx, env
                        ):
                            continue
                        raise ValueError(
                            f"tile {consumer.name!r} idx {consumer_idx} reads tensor "
                            f"{tensor.name!r} region {_region_label(read_region)} overlapping "
                            f"tile {producer.name!r} idx {producer_idx} write region "
                            f"{_region_label(write_region)} without a matching event coordinate"
                        )


def _tiles_ordered_by_events(event_ordering, producer, consumer) -> bool:
    if producer is consumer:
        return True
    return id(consumer) in event_ordering[id(producer)]


def _matching_event_coord(producer, producer_idx, consumer, consumer_idx, env) -> bool:
    for notify_event, notify_map in producer.notifies:
        notify_coord = _resolve_coord(notify_map, producer_idx, env)
        for wait_event, wait_map in consumer.waits:
            wait_coord = _resolve_coord(wait_map, consumer_idx, env)
            if wait_event is notify_event and wait_coord == notify_coord:
                return True
    return False


def _resolve_access_region(access: TensorSpec, indices, env) -> RegionSpec:
    tensor_shape = _resolve_tuple(access.base_tensor.shape, env)
    if access.region_dynamic:
        return RegionSpec(dynamic=True, reason=access.region_reason)
    if not access.has_region:
        if tensor_shape is None:
            return RegionSpec(dynamic=True, reason="unresolved full tensor")
        return RegionSpec(tuple(RegionRange(0, extent) for extent in tensor_shape))
    value = access.region_map(*indices) if callable(access.region_map) else access.region_map
    region = _normalize_region(value)
    if region.dynamic:
        return region
    return RegionSpec(
        tuple(
            RegionRange(
                _resolve_expr(dimension.start, env),
                _resolve_expr(dimension.extent, env),
            )
            for dimension in region.dims
        )
    )


def _validate_region_bounds(tensor, region, shape, label: str) -> None:
    if region.dynamic:
        return
    if len(region.dims) != len(shape):
        raise ValueError(f"{label} rank does not match tensor {tensor.name!r}")
    for dimension, tensor_extent in zip(region.dims, shape):
        if not _is_int(dimension.start) or not _is_int(dimension.extent):
            raise TypeError(f"{label} must resolve to integer starts and extents")
        if dimension.extent <= 0:
            raise ValueError(f"{label} region {_region_label(region)} has non-positive extent")
        if dimension.start < 0 or dimension.start + dimension.extent > tensor_extent:
            raise ValueError(
                f"{label} region {_region_label(region)} is out of bounds for tensor "
                f"{tensor.name!r} shape {shape}"
            )


def _regions_overlap(lhs: RegionSpec, rhs: RegionSpec) -> bool:
    if lhs.dynamic or rhs.dynamic:
        return True
    if len(lhs.dims) != len(rhs.dims):
        return False
    for lhs_dim, rhs_dim in zip(lhs.dims, rhs.dims):
        if lhs_dim.start >= rhs_dim.start + rhs_dim.extent:
            return False
        if rhs_dim.start >= lhs_dim.start + lhs_dim.extent:
            return False
    return True


def _region_label(region: RegionSpec) -> str:
    if region.dynamic:
        return f"dynamic({region.reason})" if region.reason else "dynamic"
    return (
        "["
        + ", ".join(
            str(dim.start) if dim.extent == 1 else f"{dim.start}:{dim.start + dim.extent}"
            for dim in region.dims
        )
        + "]"
    )


def _bounded_environments(
    variables: Iterable[VarSpec],
    label: str,
    *,
    require_bounded: bool,
):
    variables = tuple(variables)
    if not variables:
        return (None,)
    unbounded = [var.name for var in variables if var.range is None]
    if unbounded:
        if require_bounded:
            names = ", ".join(repr(name) for name in unbounded)
            raise ValueError(
                f"{label} requires explicit ranges for symbolic variables {names}; "
                "use a dynamic tensor region with a non-empty reason when the region "
                "cannot be proven statically"
            )
        return ()
    count = math.prod(var.range[1] - var.range[0] + 1 for var in variables)
    if count > _SYMBOLIC_ENVIRONMENT_LIMIT:
        raise ValueError(
            f"{label} needs {count} symbolic environments, exceeding the exact "
            f"validation limit {_SYMBOLIC_ENVIRONMENT_LIMIT}"
        )
    ranges = [range(var.range[0], var.range[1] + 1) for var in variables]
    return tuple(dict(zip(variables, values)) for values in product(*ranges))


def _tile_indices(tile_num, env, label: str) -> tuple[tuple[int, ...], ...]:
    extents = _resolve_tuple(tile_num, env)
    if extents is None:
        raise ValueError(f"{label} tile extents cannot be resolved")
    count = math.prod(extents)
    if count > _REGION_ENUMERATION_LIMIT:
        raise ValueError(
            f"{label} needs {count} tile points, exceeding the exact region-validation "
            f"limit {_REGION_ENUMERATION_LIMIT}; use a dynamic tensor region with a "
            "non-empty reason when the relationship cannot be proven statically"
        )
    return tuple(product(*(range(extent) for extent in extents)))


def _region_pairs(lhs, rhs, label: str):
    count = len(lhs) * len(rhs)
    if count > _REGION_ENUMERATION_LIMIT:
        raise ValueError(
            f"{label} needs {count} tile pairs, exceeding the exact region-validation "
            f"limit {_REGION_ENUMERATION_LIMIT}; use a dynamic tensor region with a "
            "non-empty reason when the relationship cannot be proven statically"
        )
    return product(lhs, rhs)


def _resolve_coord(coord_map, indices, env) -> tuple[int, ...]:
    coord = coord_map(*indices) if callable(coord_map) else coord_map
    if not isinstance(coord, tuple | list):
        raise TypeError("coordinate map must return a tuple or list")
    return tuple(_resolve_expr(value, env) for value in coord)


def _resolve_expr(value, env):
    resolved = eval_expr_like(value, env)
    return value if resolved is None else resolved


def _resolve_tuple(values, env) -> tuple[int, ...] | None:
    result = []
    for value in _shape_tuple(values):
        resolved = eval_expr_like(value, env)
        if not _is_int(resolved) or resolved <= 0:
            return None
        result.append(resolved)
    return tuple(result)


def _validate_static_coord(event, coord, shape, label: str) -> None:
    if len(coord) != len(shape):
        raise ValueError(f"{label} coordinate rank does not match event {event.name!r}")
    for value, extent in zip(coord, shape):
        if not _is_int(value):
            raise TypeError(f"{label} static coordinate contains non-integer value {value!r}")
        if value < 0 or value >= extent:
            raise ValueError(
                f"{label} coord {coord} is out of bounds for event {event.name!r} shape {shape}"
            )


def _shape_tuple(shape) -> tuple[Any, ...]:
    return tuple(shape) if isinstance(shape, tuple | list) else (shape,)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_expr_like(value: Any) -> bool:
    return _is_int(value) or isinstance(value, VarSpec | ExprSpec)


__all__ = ["validate_semantic_plan"]
