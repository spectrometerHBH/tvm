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
"""Validation for logical megakernel specifications.

``validate_kernel`` runs every backend-independent check directly on a
``KernelSpec``.  It derives the variables, tensors, events, tiles, and
logical event edges it needs from the spec itself; there is no separate
plan object.  Logical edges are derived per event as the cross product of
{tiles notifying the event} x {tiles waiting on the event}, in stable tile
order.
"""

from __future__ import annotations

import inspect
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import product
from typing import Any

from ..dsl import (
    EventSpec,
    ExprSpec,
    KernelSpec,
    RegionRange,
    RegionSpec,
    TensorSpec,
    TileImpl,
    TileSpec,
    VarSpec,
    eval_expr_like,
    expr_bounds,
    expr_vars,
)

_EXACT_ENUMERATION_LIMIT = 262_144
_SYMBOLIC_ENVIRONMENT_LIMIT = 4_096
_REGION_ENUMERATION_LIMIT = 65_536
_COMBINED_VALIDATION_LIMIT = 4_194_304
_EXPR_ARITY = {
    "add": 2,
    "sub": 2,
    "mul": 2,
    "floordiv": 2,
    "mod": 2,
    "neg": 1,
    "ceildiv": 2,
}


@dataclass(frozen=True, eq=False)
class _LogicalEdge:
    """One derived producer-to-consumer edge induced by a logical event."""

    event: EventSpec
    producer: TileSpec
    consumer: TileSpec


@dataclass(frozen=True)
class _SpecView:
    """Internal derived view of one ``KernelSpec`` consumed by the checkers."""

    kernel: KernelSpec
    vars: tuple[VarSpec, ...]
    tensors: tuple[TensorSpec, ...]
    events: tuple[EventSpec, ...]
    tiles: tuple[TileSpec, ...]
    logical_edges: tuple[_LogicalEdge, ...]


def _spec_view(kernel: KernelSpec) -> _SpecView:
    """Derive the internal validation view of one ``KernelSpec``."""

    return _SpecView(
        kernel=kernel,
        vars=tuple(kernel.vars.values()),
        tensors=tuple(kernel.tensors.values()),
        events=tuple(kernel.events.values()),
        tiles=tuple(kernel.tiles),
        logical_edges=_logical_edges(kernel),
    )


def _logical_edges(kernel: KernelSpec) -> tuple[_LogicalEdge, ...]:
    """Return logical event edges in stable event and tile order."""

    producers: dict[int, list] = {id(event): [] for event in kernel.events.values()}
    consumers: dict[int, list] = {id(event): [] for event in kernel.events.values()}
    for tile in kernel.tiles:
        for event, _ in tile.notifies:
            if all(candidate is not tile for candidate in producers.setdefault(id(event), [])):
                producers[id(event)].append(tile)
        for event, _ in tile.waits:
            if all(candidate is not tile for candidate in consumers.setdefault(id(event), [])):
                consumers[id(event)].append(tile)

    result = []
    for event in kernel.events.values():
        for producer in producers.get(id(event), ()):
            for consumer in consumers.get(id(event), ()):
                result.append(_LogicalEdge(event, producer, consumer))
    return tuple(result)


def _event_init_count(event: EventSpec, coord: tuple[int, ...]) -> int:
    """Evaluate one logical event count at a concrete coordinate."""

    return event.init_count(coord) if callable(event.init_count) else event.init_count


def validate_kernel(kernel: KernelSpec) -> KernelSpec:
    """Validate the backend-independent meaning of a megakernel spec."""

    if not isinstance(kernel, KernelSpec):
        raise TypeError("kernel validation requires a KernelSpec")
    if not isinstance(kernel.attrs, dict):
        raise TypeError("kernel attrs must be a dict")
    view = _spec_view(kernel)

    var_ids = {id(var) for var in view.vars}
    tensor_ids = {id(tensor) for tensor in view.tensors}
    event_ids = {id(event) for event in view.events}
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
    for tile in view.tiles:
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

    _validate_tensor_access_regions(view)
    _validate_tensor_writers(tensor_writers, view)
    for event in view.events:
        if consumers[id(event)] and not producers[id(event)]:
            raise ValueError(f"event {event.name!r} is waited on but has no notifier")
        if producers[id(event)] and not consumers[id(event)]:
            raise ValueError(f"event {event.name!r} is notified but has no consumer")
        _validate_event_counts(event, producers[id(event)], consumers[id(event)], view.vars)

    event_adjacency = _event_adjacency(view)
    event_ordering = _transitive_closure(event_adjacency)
    _validate_waited_region_sources(view, tensor_writers)
    _validate_tensor_dependencies(view, tensor_writers, tensor_readers, event_ordering)
    return kernel


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


def _related_variables(variables: Iterable[VarSpec], *values: Any) -> tuple[VarSpec, ...]:
    referenced_ids = {id(var) for value in values for var in _value_variables(value)}
    return tuple(var for var in variables if id(var) in referenced_ids)


def _value_variables(value: Any) -> tuple[VarSpec, ...]:
    """Return expression variables, including values captured by a callable."""

    result: list[VarSpec] = []
    seen_vars: set[int] = set()
    seen_values: set[int] = set()

    def record(var: VarSpec) -> None:
        if id(var) not in seen_vars:
            seen_vars.add(id(var))
            result.append(var)

    def visit(item: Any, *, inspect_attrs: bool = False) -> None:
        if isinstance(item, VarSpec):
            record(item)
            return
        if isinstance(item, ExprSpec):
            for var in expr_vars(item):
                record(var)
            return
        if item is None or isinstance(item, str | bytes | int | float | bool):
            return
        item_id = id(item)
        if item_id in seen_values:
            return
        seen_values.add(item_id)
        if isinstance(item, dict):
            for key, child in item.items():
                visit(key, inspect_attrs=inspect_attrs)
                visit(child, inspect_attrs=inspect_attrs)
            return
        if isinstance(item, tuple | list | set | frozenset):
            for child in item:
                visit(child, inspect_attrs=inspect_attrs)
            return
        if callable(item):
            visit_callable(item)
            return
        if inspect_attrs:
            try:
                attributes = vars(item)
            except TypeError:
                return
            for child in attributes.values():
                visit(child)

    def visit_callable(func: Any) -> None:
        bound_self = getattr(func, "__self__", None)
        if bound_self is not None:
            visit(bound_self, inspect_attrs=True)

        target = getattr(func, "__func__", func)
        visit(getattr(target, "__defaults__", None), inspect_attrs=True)
        visit(getattr(target, "__kwdefaults__", None), inspect_attrs=True)
        for cell in getattr(target, "__closure__", None) or ():
            try:
                captured = cell.cell_contents
            except ValueError:
                continue
            visit(captured, inspect_attrs=True)
        try:
            closure = inspect.getclosurevars(target)
        except (TypeError, ValueError):
            closure = None
        if closure is not None:
            visit(closure.nonlocals, inspect_attrs=True)
            # Global namespaces commonly contain modules and classes.  Scan
            # their explicit values without recursively walking those global
            # namespaces.  Ordinary holder objects may still carry a VarSpec
            # in an attribute used by the callable.
            for child in closure.globals.values():
                visit(
                    child,
                    inspect_attrs=not (inspect.ismodule(child) or inspect.isclass(child)),
                )

        # Callable instances keep their bound values on the object rather than
        # in ``__self__`` or a Python function closure.
        if target is func and not inspect.isfunction(func):
            try:
                attributes = vars(func)
            except TypeError:
                attributes = {}
            for child in attributes.values():
                visit(child)

    visit(value)
    return tuple(result)


def _region_validation_values(tile, access: TensorSpec) -> tuple[Any, ...]:
    values = [tile.tile_num, access.base_tensor.shape, access.region_map]
    if access.has_region and not access.region_dynamic:
        for indices in ((0, 0, 0), (1, 2, 3)):
            region = _normalize_region(
                _call_region_map(access.region_map, indices, "tensor-region variable discovery")
            )
            values.extend((dimension.start, dimension.extent) for dimension in region.dims)
    return tuple(values)


def _event_validation_values(event, producers, consumers) -> tuple[Any, ...]:
    values = [event.shape]
    for tile in (*producers, *consumers):
        values.append(tile.tile_num)
        for dependency_event, coord_map in (*tile.notifies, *tile.waits):
            if dependency_event is not event:
                continue
            values.append(coord_map)
            values.extend(
                _call_coord_map(coord_map, indices, "event-coordinate variable discovery")
                for indices in ((0, 0, 0), (1, 2, 3))
            )
    return tuple(values)


def _validate_tensor_writers(writers, view: _SpecView) -> None:
    for tensor in view.tensors:
        entries = writers[id(tensor)]
        distinct_tiles = []
        for tile, _ in entries:
            if all(candidate is not tile for candidate in distinct_tiles):
                distinct_tiles.append(tile)
        all_static = all(access.has_region and not access.region_dynamic for _, access in entries)
        if len(distinct_tiles) > 1 and not all_static:
            raise ValueError(f"tensor {tensor.name!r} has multiple producers")
        if entries and all_static:
            _validate_writer_regions_disjoint(tensor, entries, view.vars)


def _validate_tensor_access_regions(view: _SpecView) -> None:
    static_accesses = [
        (tile, access, kind)
        for tile in view.tiles
        for kind, accesses in (("read", tile.reads), ("write", tile.writes))
        for access in accesses
        if access.has_region and not access.region_dynamic
    ]
    if not static_accesses:
        return
    for tile, access, kind in static_accesses:
        variables = _related_variables(view.vars, *_region_validation_values(tile, access))
        environments = _bounded_environments(
            variables,
            "static tensor-region bounds validation",
            require_bounded=True,
            work_per_environment=_validation_work_upper_bound(tile.tile_num),
        )
        for env in environments:
            tensor = access.base_tensor
            shape = _resolve_tuple(tensor.shape, env)
            if shape is None:
                raise ValueError(f"tensor {tensor.name!r} shape cannot be resolved")
            label = f"tile {tile.name!r} {kind}-region validation"
            for indices in _tile_indices(tile.tile_num, env, label):
                region = _resolve_access_region(access, indices, env)
                _validate_region_bounds(tensor, region, shape, label)


def _validate_writer_regions_disjoint(tensor, writers, variables) -> None:
    related = _related_variables(
        variables,
        *(value for tile, access in writers for value in _region_validation_values(tile, access)),
    )
    environments = _bounded_environments(
        related,
        f"tensor {tensor.name!r} writer-region validation",
        require_bounded=True,
        work_per_environment=_writer_comparison_work_upper_bound(writers),
    )
    for env in environments:
        shape = _resolve_tuple(tensor.shape, env)
        if shape is None:
            raise ValueError(f"tensor {tensor.name!r} shape cannot be resolved")
        resolved_writers = [
            (
                tile,
                access,
                _tile_indices(
                    tile.tile_num,
                    env,
                    f"tile {tile.name!r} writer-region validation",
                ),
            )
            for tile, access in writers
        ]
        for lhs_tile, lhs_access, lhs_indices in resolved_writers:
            label = f"tensor {tensor.name!r} writer instances for {lhs_tile.name!r}"
            for lhs_idx, rhs_idx in _region_pairs(lhs_indices, lhs_indices, label):
                if lhs_idx >= rhs_idx:
                    continue
                _validate_writer_region_pair(
                    tensor,
                    shape,
                    env,
                    lhs_tile,
                    lhs_access,
                    lhs_idx,
                    lhs_tile,
                    lhs_access,
                    rhs_idx,
                )
        for index, (lhs_tile, lhs_access, lhs_indices) in enumerate(resolved_writers):
            for rhs_tile, rhs_access, rhs_indices in resolved_writers[index + 1 :]:
                label = (
                    f"tensor {tensor.name!r} writer regions for "
                    f"{lhs_tile.name!r} and {rhs_tile.name!r}"
                )
                for lhs_idx, rhs_idx in _region_pairs(lhs_indices, rhs_indices, label):
                    if lhs_tile is rhs_tile and lhs_idx == rhs_idx:
                        continue
                    _validate_writer_region_pair(
                        tensor,
                        shape,
                        env,
                        lhs_tile,
                        lhs_access,
                        lhs_idx,
                        rhs_tile,
                        rhs_access,
                        rhs_idx,
                    )


def _writer_comparison_work_upper_bound(writers) -> int:
    point_counts = [_shape_point_upper_bound(tile.tile_num) or 1 for tile, _ in writers]
    return sum(count * count for count in point_counts) + sum(
        lhs_count * rhs_count
        for index, lhs_count in enumerate(point_counts)
        for rhs_count in point_counts[index + 1 :]
    )


def _validate_writer_region_pair(
    tensor,
    shape,
    env,
    lhs_tile,
    lhs_access,
    lhs_idx,
    rhs_tile,
    rhs_access,
    rhs_idx,
) -> None:
    lhs = _resolve_access_region(lhs_access, lhs_idx, env)
    rhs = _resolve_access_region(rhs_access, rhs_idx, env)
    _validate_region_bounds(tensor, lhs, shape, f"{lhs_tile.name}.write_region")
    _validate_region_bounds(tensor, rhs, shape, f"{rhs_tile.name}.write_region")
    if _regions_overlap(lhs, rhs):
        raise ValueError(
            f"tensor {tensor.name!r} has multiple producers with overlapping regions: "
            f"{lhs_tile.name!r} idx {lhs_idx} {_region_label(lhs)} and "
            f"{rhs_tile.name!r} idx {rhs_idx} {_region_label(rhs)}"
        )


def _validate_event_counts(event, producers, consumers, variables) -> None:
    if not producers:
        return
    plan_variables = tuple(variables)
    variables = _related_variables(
        plan_variables, *_event_validation_values(event, producers, consumers)
    )
    shape_variables = _related_variables(plan_variables, event.shape)
    active_tile_variables = _related_variables(
        plan_variables, *(tile.tile_num for tile in (*producers, *consumers))
    )
    # A symbolic event shape denotes its full runtime domain.  A constant
    # event shape paired with a symbolic tile count is a capacity: only
    # coordinates reached by active producers or consumers are required.  A
    # captured variable alone does not make an otherwise fixed tile domain a
    # capacity; it may be unused by the callable.  The capacity form is used
    # by persistent kernels such as MoE, where the event buffer is reserved at
    # a maximum size but ``routed_rows`` is runtime-active.
    full_domain = not active_tile_variables or bool(shape_variables)
    work_per_environment = _validation_work_upper_bound(
        event.shape, *(tile.tile_num for tile in (*producers, *consumers))
    )
    environments = (
        _bounded_environments(
            variables,
            f"event {event.name!r} count validation",
            require_bounded=False,
            work_per_environment=work_per_environment,
        )
        if variables
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
        producer_points = sum(math.prod(extents) for extents in producer_extents)
        producer_exact = producer_points <= _EXACT_ENUMERATION_LIMIT
        consumer_points = sum(math.prod(extents) for extents in consumer_extents)
        consumer_exact = consumer_points <= _EXACT_ENUMERATION_LIMIT
        if full_domain and _is_int(event.init_count):
            expected_total = math.prod(event_shape) * event.init_count
            if producer_points != expected_total:
                raise ValueError(
                    f"event {event.name!r} expects init_count {event.init_count} across its "
                    f"full domain ({expected_total} total static notifies), but producer "
                    f"grids contain {producer_points}"
                )
        counts: dict[tuple[int, ...], int] = defaultdict(int)
        for tile, extents in zip(producers, producer_extents):
            indices_values = (
                tuple(product(*(range(extent) for extent in extents)))
                if producer_exact
                else _grid_indices(extents, 0)[0]
            )
            for indices in indices_values:
                for notify_event, coord_map in tile.notifies:
                    if notify_event is not event:
                        continue
                    coord = _resolve_coord(coord_map, indices, env)
                    _validate_static_coord(event, coord, event_shape, f"{tile.name}.notify")
                    counts[coord] += 1
        waited_coords: set[tuple[int, ...]] = set()
        for tile, extents in zip(consumers, consumer_extents):
            indices_values = (
                tuple(product(*(range(extent) for extent in extents)))
                if consumer_exact
                else _grid_indices(extents, 0)[0]
            )
            for indices in indices_values:
                for wait_event, coord_map in tile.waits:
                    if wait_event is not event:
                        continue
                    coord = _resolve_coord(coord_map, indices, env)
                    _validate_static_coord(event, coord, event_shape, f"{tile.name}.wait")
                    waited_coords.add(coord)
                    if producer_exact and counts.get(coord, 0) == 0:
                        raise ValueError(
                            f"tile {tile.name!r} waits on event {event.name!r} coord "
                            f"{coord} without a producer notify"
                        )

        event_coords = ()
        if full_domain:
            event_coords, _ = _grid_indices(event_shape, _EXACT_ENUMERATION_LIMIT)
        count_coords = tuple(dict.fromkeys((*event_coords, *counts, *waited_coords)))
        for coord in count_coords:
            expected = _event_init_count(event, coord)
            if not _is_int(expected) or expected <= 0:
                raise ValueError(
                    f"event {event.name!r} init_count at coord {coord} must be a positive integer"
                )
            actual = counts.get(coord, 0)
            invalid = actual != expected if producer_exact else actual > expected
            if invalid:
                qualifier = "" if producer_exact else "sampled "
                raise ValueError(
                    f"event {event.name!r} coord {coord} expects init_count {expected}, "
                    f"but has {actual} {qualifier}static notifies"
                )


def _event_adjacency(view: _SpecView) -> dict[int, set[int]]:
    adjacency = {id(tile): set() for tile in view.tiles}
    for edge in view.logical_edges:
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


class _InstanceEventGraph:
    """Concrete event-coordinate graph for one symbolic environment."""

    def __init__(self, adjacency, tile_projection, *, exact: bool):
        self.adjacency = adjacency
        self.tile_projection = tile_projection
        self.exact = exact
        self._reachable: dict[Any, set[Any]] = {}
        self._nodes_by_tile: dict[int, list[Any]] = defaultdict(list)
        for node in adjacency:
            if node[0] == "tile":
                self._nodes_by_tile[node[1]].append(node)

    def _reachable_from(self, source):
        reachable = self._reachable.get(source)
        if reachable is None:
            reachable = set()
            pending = list(self.adjacency.get(source, ()))
            while pending:
                current = pending.pop()
                if current in reachable:
                    continue
                reachable.add(current)
                pending.extend(self.adjacency.get(current, ()) - reachable)
            self._reachable[source] = reachable
        return reachable

    def ordered(self, producer, producer_idx, consumer, consumer_idx) -> bool | None:
        source = ("tile", id(producer), tuple(producer_idx))
        target = ("tile", id(consumer), tuple(consumer_idx))
        if target in self._reachable_from(source):
            return True
        return False if self.exact else None

    def tiles_ordered(self, producer, consumer) -> bool | None:
        consumer_id = id(consumer)
        for source in self._nodes_by_tile.get(id(producer), ()):
            if any(
                node[0] == "tile" and node[1] == consumer_id
                for node in self._reachable_from(source)
            ):
                return True
        return False if self.exact else None

    def has_cycle(self) -> bool:
        visiting = set()
        visited = set()

        def visit(node) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            if any(visit(target) for target in self.adjacency.get(node, ())):
                return True
            visiting.remove(node)
            visited.add(node)
            return False

        return any(visit(node) for node in self.adjacency)


def _event_graph_validation_values(view: _SpecView, tiles=None) -> tuple[Any, ...]:
    tiles = tuple(view.tiles if tiles is None else tiles)
    referenced_events = {
        id(event): event for tile in tiles for event, _ in (*tile.notifies, *tile.waits)
    }
    values: list[Any] = [event.shape for event in referenced_events.values()]
    for tile in tiles:
        values.append(tile.tile_num)
        for _, coord_map in (*tile.notifies, *tile.waits):
            values.append(coord_map)
            values.extend(
                _call_coord_map(coord_map, indices, "event-graph variable discovery")
                for indices in ((0, 0, 0), (1, 2, 3))
            )
    return tuple(values)


def _build_instance_event_graph(view: _SpecView, env, tiles=None) -> _InstanceEventGraph:
    tiles = tuple(view.tiles if tiles is None else tiles)
    referenced_events = {
        id(event): event for tile in tiles for event, _ in (*tile.notifies, *tile.waits)
    }
    event_shapes = {}
    for event in referenced_events.values():
        shape = _resolve_tuple(event.shape, env)
        if shape is None:
            raise ValueError(f"event {event.name!r} shape cannot be resolved")
        event_shapes[id(event)] = shape

    adjacency = {}
    producers: dict[tuple[int, tuple[int, ...]], list[Any]] = defaultdict(list)
    consumers: dict[tuple[int, tuple[int, ...]], list[Any]] = defaultdict(list)
    exact = True
    for tile in tiles:
        extents = _resolve_tuple(tile.tile_num, env)
        if extents is None:
            raise ValueError(f"tile {tile.name!r} event-graph extents cannot be resolved")
        indices_values, tile_exact = _grid_indices(extents, _REGION_ENUMERATION_LIMIT)
        exact &= tile_exact
        for indices in indices_values:
            node = ("tile", id(tile), tuple(indices))
            adjacency.setdefault(node, set())
            for event, coord_map in tile.notifies:
                coord = _resolve_coord(coord_map, indices, env)
                _validate_static_coord(event, coord, event_shapes[id(event)], f"{tile.name}.notify")
                producers[(id(event), coord)].append(node)
            for event, coord_map in tile.waits:
                coord = _resolve_coord(coord_map, indices, env)
                _validate_static_coord(event, coord, event_shapes[id(event)], f"{tile.name}.wait")
                consumers[(id(event), coord)].append(node)

    tile_projection = {id(tile): set() for tile in tiles}
    for key, producer_nodes in producers.items():
        consumer_nodes = tuple(consumers.get(key, ()))
        if not consumer_nodes:
            continue
        event_node = ("event", key[0], key[1])
        adjacency.setdefault(event_node, set()).update(consumer_nodes)
        consumer_tile_ids = {node[1] for node in consumer_nodes}
        producer_tile_ids = {node[1] for node in producer_nodes}
        for producer_tile_id in producer_tile_ids:
            tile_projection[producer_tile_id].update(consumer_tile_ids)
        for producer_node in producer_nodes:
            adjacency[producer_node].add(event_node)
    return _InstanceEventGraph(adjacency, tile_projection, exact=exact)


def _adjacency_has_cycle(adjacency) -> bool:
    visiting = set()
    visited = set()

    def visit(node) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        if any(visit(target) for target in adjacency.get(node, ())):
            return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in adjacency)


def _tiles_on_dependency_paths(view: _SpecView, producer, consumer):
    adjacency = _event_adjacency(view)
    reverse = {tile_id: set() for tile_id in adjacency}
    for source, targets in adjacency.items():
        for target in targets:
            reverse[target].add(source)

    def reachable(graph, start):
        result = {start}
        pending = [start]
        while pending:
            current = pending.pop()
            for target in graph[current] - result:
                result.add(target)
                pending.append(target)
        return result

    candidate_ids = reachable(adjacency, id(producer)) & reachable(reverse, id(consumer))
    candidate_ids.update((id(producer), id(consumer)))
    return tuple(tile for tile in view.tiles if id(tile) in candidate_ids)


def _tiles_in_dependency_cycles(view: _SpecView):
    adjacency = {id(tile): set() for tile in view.tiles}
    for edge in view.logical_edges:
        adjacency[id(edge.producer)].add(id(edge.consumer))
    closure = _transitive_closure(adjacency)
    cyclic_ids = {tile_id for tile_id, reachable in closure.items() if tile_id in reachable}
    return adjacency, tuple(tile for tile in view.tiles if id(tile) in cyclic_ids)


def _tile_pair_ordered_by_event_coords(
    view: _SpecView,
    event_ordering,
    producer,
    consumer,
    graph_cache,
) -> bool:
    """Check existential tile ordering using concrete event coordinates."""

    if not _tiles_ordered_by_events(event_ordering, producer, consumer):
        return False
    candidate_tiles = _tiles_on_dependency_paths(view, producer, consumer)
    values = _event_graph_validation_values(view, candidate_tiles)
    variables = _related_variables(view.vars, *values)
    environments = _bounded_environments(
        variables,
        "tensor event-coordinate dependency validation",
        require_bounded=False,
        work_per_environment=_validation_work_upper_bound(
            *(tile.tile_num for tile in candidate_tiles)
        ),
    )
    for env in environments:
        key = (
            tuple(id(tile) for tile in candidate_tiles),
            tuple(env[var] for var in variables) if env is not None else (),
        )
        graph = graph_cache.get(key)
        if graph is None:
            graph = _build_instance_event_graph(view, env, candidate_tiles)
            graph_cache[key] = graph
        ordering = graph.tiles_ordered(producer, consumer)
        if ordering is False:
            return False
        if ordering is None:
            # The selected tile grid exceeded the deterministic point budget.
            # The coarse logical path remains a conservative fallback only in
            # this explicitly inexact case.
            continue
    return True


def _static_event_tile_adjacency(view: _SpecView):
    """Return the exact tile-phase DAG required by the default static queue."""

    coarse, cyclic_tiles = _tiles_in_dependency_cycles(view)
    if not _adjacency_has_cycle(coarse):
        return coarse

    values = _event_graph_validation_values(view, cyclic_tiles)
    variables = _related_variables(view.vars, *values)
    environments, environments_exact = _validation_environments(
        variables,
        "static event dependency cycle validation",
        require_bounded=False,
        work_per_environment=_validation_work_upper_bound(
            *(tile.tile_num for tile in cyclic_tiles)
        ),
    )
    all_exact = environments_exact
    matched_projection = {id(tile): set() for tile in cyclic_tiles}
    for env in environments:
        graph = _build_instance_event_graph(view, env, cyclic_tiles)
        if graph.has_cycle():
            raise ValueError(
                "the default static schedule requires event dependencies to be acyclic"
            )
        for producer_id, consumer_ids in graph.tile_projection.items():
            matched_projection[producer_id].update(consumer_ids)
        all_exact &= graph.exact
    if not all_exact:
        raise ValueError(
            "the default static schedule could not prove symbolic event dependencies acyclic"
        )

    # A tile phase cannot interleave two TileSpec programs.  Even when the
    # concrete instance graph is acyclic, a cycle in its tile projection cannot
    # be represented by this static phase scheduler.
    if _adjacency_has_cycle(matched_projection):
        raise ValueError(
            "the default static schedule requires event-coordinate dependencies "
            "to project to an acyclic tile-phase order"
        )

    cyclic_ids = set(matched_projection)
    result = {tile_id: set(consumers) for tile_id, consumers in coarse.items()}
    for producer_id in cyclic_ids:
        result[producer_id].difference_update(cyclic_ids)
        result[producer_id].update(matched_projection[producer_id])
    if _adjacency_has_cycle(result):
        raise ValueError(
            "the default static schedule requires event-coordinate dependencies "
            "to project to an acyclic tile-phase order"
        )
    return result


def _validate_tensor_dependencies(view, writers, readers, event_ordering) -> None:
    tile_pair_graph_cache = {}
    if not any(access.has_region for tile in view.tiles for access in (*tile.reads, *tile.writes)):
        for tensor in view.tensors:
            for producer, _ in writers[id(tensor)]:
                for consumer, _ in readers[id(tensor)]:
                    if producer is consumer:
                        continue
                    if not _tile_pair_ordered_by_event_coords(
                        view,
                        event_ordering,
                        producer,
                        consumer,
                        tile_pair_graph_cache,
                    ):
                        raise ValueError(
                            f"tile {consumer.name!r} reads tensor {tensor.name!r} written "
                            f"by tile {producer.name!r} without an event dependency"
                        )
        return

    static_pairs = []
    for tensor in view.tensors:
        for producer, write_access in writers[id(tensor)]:
            for consumer, read_access in readers[id(tensor)]:
                if producer is consumer:
                    continue
                if write_access.region_dynamic or read_access.region_dynamic:
                    if not _tile_pair_ordered_by_event_coords(
                        view,
                        event_ordering,
                        producer,
                        consumer,
                        tile_pair_graph_cache,
                    ):
                        raise ValueError(
                            f"tile {consumer.name!r} dynamically reads tensor "
                            f"{tensor.name!r} written by tile {producer.name!r} without "
                            "an event dependency"
                        )
                else:
                    static_pairs.append((tensor, producer, write_access, consumer, read_access))

    if not static_pairs:
        return

    for tensor, producer, write_access, consumer, read_access in static_pairs:
        candidate_tiles = _tiles_on_dependency_paths(view, producer, consumer)
        event_graph_values = _event_graph_validation_values(view, candidate_tiles)
        event_graph_variables = _related_variables(view.vars, *event_graph_values)
        event_graph_work = _validation_work_upper_bound(
            *(tile.tile_num for tile in candidate_tiles)
        )
        graph_cache: dict[tuple[int, ...], _InstanceEventGraph] = {}
        region_values = (
            *_region_validation_values(producer, write_access),
            *_region_validation_values(consumer, read_access),
        )
        region_variables = _related_variables(view.vars, *region_values)
        _require_bounded_variables(
            region_variables,
            "static tensor-region dependency validation",
        )
        variables = _related_variables(
            view.vars,
            *region_values,
            *event_graph_values,
        )
        pair_work = _validation_pair_work_upper_bound(producer.tile_num, consumer.tile_num)
        environments = _bounded_environments(
            variables,
            "static tensor-region dependency validation",
            require_bounded=False,
            work_per_environment=(pair_work or 1) + (event_graph_work or 1),
        )
        for env in environments:
            graph_key = tuple(env[var] for var in event_graph_variables) if env is not None else ()
            instance_graph = graph_cache.get(graph_key)
            if instance_graph is None:
                instance_graph = _build_instance_event_graph(view, env, candidate_tiles)
                graph_cache[graph_key] = instance_graph
            shape = _resolve_tuple(tensor.shape, env)
            if shape is None:
                raise ValueError(f"tensor {tensor.name!r} shape cannot be resolved")
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
                _validate_region_bounds(tensor, read_region, shape, f"{consumer.name}.read_region")
                if not _regions_overlap(write_region, read_region):
                    continue
                instance_ordering = instance_graph.ordered(
                    producer,
                    producer_idx,
                    consumer,
                    consumer_idx,
                )
                if instance_ordering is True:
                    continue
                if instance_ordering is None and _tiles_ordered_by_events(
                    event_ordering, producer, consumer
                ):
                    continue
                raise ValueError(
                    f"tile {consumer.name!r} idx {consumer_idx} reads tensor "
                    f"{tensor.name!r} region {_region_label(read_region)} overlapping "
                    f"tile {producer.name!r} idx {producer_idx} write region "
                    f"{_region_label(write_region)} without an event dependency"
                )


def _tiles_ordered_by_events(event_ordering, producer, consumer) -> bool:
    if producer is consumer:
        return True
    return id(consumer) in event_ordering[id(producer)]


def _validate_waited_region_sources(view: _SpecView, writers) -> None:
    """Ensure a waited event coordinate protects each related static read region."""

    relevant = [
        (consumer, read_access, wait_event, wait_map)
        for consumer in view.tiles
        for read_access in consumer.reads
        if read_access.has_region and not read_access.region_dynamic
        for wait_event, wait_map in consumer.waits
        if any(
            producer is not consumer and any(event is wait_event for event, _ in producer.notifies)
            for producer, _ in writers[id(read_access.base_tensor)]
        )
    ]
    if not relevant:
        return
    for consumer, read_access, wait_event, wait_map in relevant:
        tensor = read_access.base_tensor
        candidates = [
            (producer, write_access)
            for producer, write_access in writers[id(tensor)]
            if producer is not consumer
            and any(event is wait_event for event, _ in producer.notifies)
        ]
        candidate_producers = []
        for producer, _ in candidates:
            if all(existing is not producer for existing in candidate_producers):
                candidate_producers.append(producer)
        variables = _related_variables(
            view.vars,
            *_region_validation_values(consumer, read_access),
            *(
                value
                for producer, write_access in candidates
                for value in _region_validation_values(producer, write_access)
            ),
            *_event_validation_values(wait_event, candidate_producers, (consumer,)),
        )
        environments = _bounded_environments(
            variables,
            "waited tensor-region source validation",
            require_bounded=True,
            work_per_environment=(
                (_shape_point_upper_bound(consumer.tile_num) or 1)
                * sum(
                    _shape_point_upper_bound(producer.tile_num) or 1 for producer, _ in candidates
                )
            ),
        )
        for env in environments:
            shape = _resolve_tuple(tensor.shape, env)
            if shape is None:
                raise ValueError(f"tensor {tensor.name!r} shape cannot be resolved")
            for consumer_idx in _tile_indices(
                consumer.tile_num,
                env,
                f"tile {consumer.name!r} waited-region validation",
            ):
                read_region = _resolve_access_region(read_access, consumer_idx, env)
                _validate_region_bounds(tensor, read_region, shape, f"{consumer.name}.read_region")
                wait_coord = _resolve_coord(wait_map, consumer_idx, env)
                has_source = _wait_has_region_source(
                    candidates,
                    tensor,
                    shape,
                    read_region,
                    wait_event,
                    wait_coord,
                    env,
                )
                if has_source is not False:
                    continue
                raise ValueError(
                    f"tile {consumer.name!r} idx {consumer_idx} waits on event "
                    f"{wait_event.name!r} coord {wait_coord} and reads tensor "
                    f"{tensor.name!r} region {_region_label(read_region)}, but no producer "
                    "notifying that coord writes an overlapping region"
                )


def _wait_has_region_source(
    candidates,
    tensor,
    shape,
    read_region,
    wait_event,
    wait_coord,
    env,
) -> bool | None:
    all_exact = True
    for producer, write_access in candidates:
        producer_extents = _resolve_tuple(producer.tile_num, env)
        if producer_extents is None:
            raise ValueError(
                f"tile {producer.name!r} waited-region source extents cannot be resolved"
            )
        producer_indices, exact = _grid_indices(producer_extents, _REGION_ENUMERATION_LIMIT)
        all_exact &= exact
        for producer_idx in producer_indices:
            if not _producer_notifies_coord(producer, producer_idx, wait_event, wait_coord, env):
                continue
            write_region = _resolve_access_region(write_access, producer_idx, env)
            _validate_region_bounds(tensor, write_region, shape, f"{producer.name}.write_region")
            if _regions_overlap(write_region, read_region):
                return True
    return False if all_exact else None


def _producer_notifies_coord(producer, producer_idx, wait_event, wait_coord, env) -> bool:
    for notify_event, notify_map in producer.notifies:
        if (
            notify_event is wait_event
            and _resolve_coord(notify_map, producer_idx, env) == wait_coord
        ):
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
    work_per_environment: int | None = None,
):
    return _validation_environments(
        variables,
        label,
        require_bounded=require_bounded,
        work_per_environment=work_per_environment,
    )[0]


def _require_bounded_variables(variables: Iterable[VarSpec], label: str) -> None:
    unbounded = [var.name for var in variables if var.range is None]
    if not unbounded:
        return
    names = ", ".join(repr(name) for name in unbounded)
    raise ValueError(
        f"{label} requires explicit ranges for symbolic variables {names}; "
        "use a dynamic tensor region with a non-empty reason when the region "
        "cannot be proven statically"
    )


def _validation_environments(
    variables: Iterable[VarSpec],
    label: str,
    *,
    require_bounded: bool,
    work_per_environment: int | None = None,
):
    variables = tuple(variables)
    if not variables:
        return (None,), True
    unbounded = [var.name for var in variables if var.range is None]
    if unbounded and require_bounded:
        _require_bounded_variables(variables, label)
    if not unbounded:
        count = math.prod(var.range[1] - var.range[0] + 1 for var in variables)
        combined_work = count * max(1, work_per_environment or 1)
        if count <= _SYMBOLIC_ENVIRONMENT_LIMIT and combined_work <= _COMBINED_VALIDATION_LIMIT:
            ranges = [range(var.range[0], var.range[1] + 1) for var in variables]
            return tuple(dict(zip(variables, values)) for values in product(*ranges)), True
    return _sample_environments(variables), False


def _validation_work_upper_bound(*shapes: Any) -> int | None:
    total = 0
    for shape in shapes:
        point_count = _shape_point_upper_bound(shape)
        if point_count is None:
            return None
        total += point_count
    return total


def _shape_point_upper_bound(shape: Any) -> int | None:
    point_count = 1
    for extent in _shape_tuple(shape):
        bounds = expr_bounds(extent, require_bounded=False)
        if bounds is None:
            return None
        point_count *= max(1, bounds[1])
    return point_count


def _validation_pair_work_upper_bound(lhs: Any, rhs: Any) -> int | None:
    lhs_work = _shape_point_upper_bound(lhs)
    rhs_work = _shape_point_upper_bound(rhs)
    if lhs_work is None or rhs_work is None:
        return None
    return lhs_work * rhs_work


def _sample_environments(variables: tuple[VarSpec, ...]):
    rng = random.Random(0)
    candidates = []
    for base in (1, 2, 4, 8, 16):
        candidates.append(
            {
                var: base if var.range is None else min(max(base, var.range[0]), var.range[1])
                for var in variables
            }
        )
    if all(var.range is not None for var in variables):
        boundary_values = []
        for var in variables:
            lower, upper = var.range
            boundary_values.append((lower, (lower + upper) // 2, upper))
        for values in zip(*boundary_values):
            candidates.append(dict(zip(variables, values)))
    for _ in range(5):
        candidates.append(
            {
                var: (
                    rng.randint(1, 16)
                    if var.range is None
                    else rng.randint(var.range[0], var.range[1])
                )
                for var in variables
            }
        )
    unique = []
    seen = set()
    for env in candidates:
        key = tuple(env[var] for var in variables)
        if key not in seen:
            seen.add(key)
            unique.append(env)
    return tuple(unique)


def _tile_indices(tile_num, env, label: str) -> tuple[tuple[int, ...], ...]:
    extents = _resolve_tuple(tile_num, env)
    if extents is None:
        raise ValueError(f"{label} tile extents cannot be resolved")
    return _grid_indices(extents, _REGION_ENUMERATION_LIMIT)[0]


def _region_pairs(lhs, rhs, label: str):
    count = len(lhs) * len(rhs)
    if count <= _REGION_ENUMERATION_LIMIT:
        return product(lhs, rhs)
    rng = random.Random(0)
    candidates = [
        (lhs[0], rhs[0]),
        (lhs[-1], rhs[-1]),
        (lhs[len(lhs) // 2], rhs[len(rhs) // 2]),
        (lhs[0], rhs[-1]),
        (lhs[-1], rhs[0]),
    ]
    candidates.extend(
        (lhs[rng.randrange(len(lhs))], rhs[rng.randrange(len(rhs))]) for _ in range(11)
    )
    return iter(tuple(dict.fromkeys(candidates)))


def _grid_indices(extents: tuple[int, ...], limit: int):
    count = math.prod(extents)
    if count <= limit:
        return tuple(product(*(range(extent) for extent in extents))), True
    lower = tuple(0 for _ in extents)
    upper = tuple(extent - 1 for extent in extents)
    middle = tuple((extent - 1) // 2 for extent in extents)
    candidates = [lower, middle, upper]
    for axis, extent in enumerate(extents):
        for value in (0, (extent - 1) // 2, extent - 1):
            point = list(middle)
            point[axis] = value
            candidates.append(tuple(point))
    rng = random.Random(0)
    candidates.extend(tuple(rng.randrange(extent) for extent in extents) for _ in range(10))
    return tuple(dict.fromkeys(candidates)), False


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


__all__ = ["validate_kernel"]
