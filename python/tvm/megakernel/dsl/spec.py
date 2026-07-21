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

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .impl import TileImpl


@dataclass(frozen=True)
class VarSpec:
    """Symbolic variable used in shapes, tile counts, and event counts.

    ``range`` is an optional inclusive ``(minimum, maximum)`` bound.  A
    backend may use the upper bound to reserve static storage while lowering
    the runtime value as an ordinary kernel argument.
    """

    name: str
    dtype: str = "int32"
    range: tuple[int, int] | None = None

    def __add__(self, other):
        return _binary_expr("add", self, other)

    def __radd__(self, other):
        return _binary_expr("add", other, self)

    def __sub__(self, other):
        return _binary_expr("sub", self, other)

    def __rsub__(self, other):
        return _binary_expr("sub", other, self)

    def __mul__(self, other):
        return _binary_expr("mul", self, other)

    def __rmul__(self, other):
        return _binary_expr("mul", other, self)

    def __floordiv__(self, other):
        return _binary_expr("floordiv", self, other)

    def __rfloordiv__(self, other):
        return _binary_expr("floordiv", other, self)

    def __mod__(self, other):
        return _binary_expr("mod", self, other)

    def __rmod__(self, other):
        return _binary_expr("mod", other, self)

    def __neg__(self):
        return ExprSpec("neg", (self,))

    def ceildiv(self, other):
        """Return ``ceildiv(self, other)`` as a logical integer expression."""

        return _binary_expr("ceildiv", self, other)


@dataclass(frozen=True)
class ExprSpec:
    """Small integer expression over ``VarSpec`` and integer constants."""

    op: str
    args: tuple[ExprLike, ...]

    def __add__(self, other):
        return _binary_expr("add", self, other)

    def __radd__(self, other):
        return _binary_expr("add", other, self)

    def __sub__(self, other):
        return _binary_expr("sub", self, other)

    def __rsub__(self, other):
        return _binary_expr("sub", other, self)

    def __mul__(self, other):
        return _binary_expr("mul", self, other)

    def __rmul__(self, other):
        return _binary_expr("mul", other, self)

    def __floordiv__(self, other):
        return _binary_expr("floordiv", self, other)

    def __rfloordiv__(self, other):
        return _binary_expr("floordiv", other, self)

    def __mod__(self, other):
        return _binary_expr("mod", self, other)

    def __rmod__(self, other):
        return _binary_expr("mod", other, self)

    def __neg__(self):
        return ExprSpec("neg", (self,))

    def ceildiv(self, other):
        """Return ``ceildiv(self, other)`` as a logical integer expression."""

        return _binary_expr("ceildiv", self, other)


@dataclass(frozen=True, eq=False)
class ScalarSpec:
    """Symbolic integer read from a device buffer at kernel runtime.

    A runtime scalar is neither a host constant nor a kernel parameter: its
    value is loaded at kernel runtime from ``source``, a registered base
    tensor plus a static index into it.  Tiles whose ``tile_num`` depends on
    a runtime scalar cannot be pre-enumerated on the host and require dynamic
    dispatch from the backend.

    ``range`` is an optional inclusive ``(minimum, maximum)`` bound used only
    for validation bound proofs, mirroring ``VarSpec.range`` semantics.
    """

    name: str
    source: tuple[TensorSpec, tuple[ExprLike, ...]]
    dtype: str = "int32"
    range: tuple[int, int] | None = None

    def __add__(self, other):
        return _binary_expr("add", self, other)

    def __radd__(self, other):
        return _binary_expr("add", other, self)

    def __sub__(self, other):
        return _binary_expr("sub", self, other)

    def __rsub__(self, other):
        return _binary_expr("sub", other, self)

    def __mul__(self, other):
        return _binary_expr("mul", self, other)

    def __rmul__(self, other):
        return _binary_expr("mul", other, self)

    def __floordiv__(self, other):
        return _binary_expr("floordiv", self, other)

    def __rfloordiv__(self, other):
        return _binary_expr("floordiv", other, self)

    def __mod__(self, other):
        return _binary_expr("mod", self, other)

    def __rmod__(self, other):
        return _binary_expr("mod", other, self)

    def __neg__(self):
        return ExprSpec("neg", (self,))

    def ceildiv(self, other):
        """Return ``ceildiv(self, other)`` as a logical integer expression."""

        return _binary_expr("ceildiv", self, other)


ExprLike = int | VarSpec | ExprSpec | ScalarSpec
ShapeType = ExprLike | tuple[ExprLike, ...] | list[ExprLike]
CoordMapType = (
    Callable[[int, int, int], tuple[ExprLike, ...]] | tuple[ExprLike, ...] | list[ExprLike]
)
TileNumType = tuple[ExprLike, ExprLike, ExprLike] | list[ExprLike]


def _as_expr_like(value: Any) -> ExprLike:
    if isinstance(value, bool):
        raise TypeError("boolean values are not valid megakernel expressions")
    if isinstance(value, int | VarSpec | ExprSpec | ScalarSpec):
        return value
    raise TypeError(
        "megakernel expression operands must be int, VarSpec, ScalarSpec, or ExprSpec, "
        f"got {value!r}"
    )


def _binary_expr(op: str, lhs: Any, rhs: Any) -> ExprSpec:
    return ExprSpec(op, (_as_expr_like(lhs), _as_expr_like(rhs)))


def expr_vars(value: Any) -> tuple[VarSpec, ...]:
    """Return the ``VarSpec`` leaves referenced by an expression or shape."""

    result: list[VarSpec] = []
    seen: set[VarSpec] = set()

    def visit(item: Any) -> None:
        if isinstance(item, VarSpec):
            if item not in seen:
                seen.add(item)
                result.append(item)
        elif isinstance(item, ExprSpec):
            for arg in item.args:
                visit(arg)
        elif isinstance(item, tuple | list):
            for child in item:
                visit(child)

    visit(value)
    return tuple(result)


def expr_scalars(value: Any) -> tuple[ScalarSpec, ...]:
    """Return the ``ScalarSpec`` leaves referenced by an expression or shape."""

    result: list[ScalarSpec] = []
    seen: set[ScalarSpec] = set()

    def visit(item: Any) -> None:
        if isinstance(item, ScalarSpec):
            if item not in seen:
                seen.add(item)
                result.append(item)
        elif isinstance(item, ExprSpec):
            for arg in item.args:
                visit(arg)
        elif isinstance(item, tuple | list):
            for child in item:
                visit(child)

    visit(value)
    return tuple(result)


def eval_expr_like(value: Any, env: dict[VarSpec, int] | None = None) -> int | None:
    """Evaluate a logical integer expression in a concrete variable environment."""

    if _is_int(value):
        return value
    if isinstance(value, VarSpec):
        return None if env is None else env.get(value)
    if isinstance(value, ScalarSpec):
        # Runtime scalars are read from device memory at kernel runtime; they
        # never resolve in a static variable environment.
        return None
    if isinstance(value, ExprSpec):
        args = [eval_expr_like(arg, env) for arg in value.args]
        if any(arg is None for arg in args):
            return None
        return _eval_expr_op(value.op, args)
    return None


def _eval_expr_op(op: str, args: list[int]) -> int:
    if op == "add":
        return args[0] + args[1]
    if op == "sub":
        return args[0] - args[1]
    if op == "mul":
        return args[0] * args[1]
    if op == "floordiv":
        return args[0] // args[1]
    if op == "mod":
        return args[0] % args[1]
    if op == "neg":
        return -args[0]
    if op == "ceildiv":
        return -((-args[0]) // args[1])
    raise ValueError(f"unsupported ExprSpec op {op!r}")


def expr_bounds(value: Any, *, require_bounded: bool = True) -> tuple[int, int] | None:
    """Return conservative inclusive bounds for a logical integer expression."""

    if _is_int(value):
        return value, value
    if isinstance(value, VarSpec):
        if value.range is None:
            if require_bounded:
                raise ValueError(f"symbolic VarSpec({value.name!r}) has no range")
            return None
        return value.range
    if isinstance(value, ScalarSpec):
        if value.range is None:
            if require_bounded:
                raise ValueError(f"runtime ScalarSpec({value.name!r}) has no range")
            return None
        return value.range
    if not isinstance(value, ExprSpec):
        if require_bounded:
            raise TypeError(f"expected int, VarSpec, ScalarSpec, or ExprSpec, got {value!r}")
        return None
    arg_bounds = [expr_bounds(arg, require_bounded=require_bounded) for arg in value.args]
    if any(bound is None for bound in arg_bounds):
        return None
    return _expr_bounds_op(value.op, arg_bounds)


def _expr_bounds_op(op: str, bounds: list[tuple[int, int]]) -> tuple[int, int]:
    if op == "add":
        return bounds[0][0] + bounds[1][0], bounds[0][1] + bounds[1][1]
    if op == "sub":
        return bounds[0][0] - bounds[1][1], bounds[0][1] - bounds[1][0]
    if op == "mul":
        values = [lhs * rhs for lhs in bounds[0] for rhs in bounds[1]]
        return min(values), max(values)
    if op in ("floordiv", "ceildiv", "mod"):
        divisor_min, divisor_max = bounds[1]
        if divisor_min <= 0 <= divisor_max:
            raise ValueError(f"{op} expression divisor range must not include zero")
        if op == "mod":
            if divisor_min > 0:
                return 0, divisor_max - 1
            return divisor_min + 1, 0
        if op == "floordiv":
            values = [lhs // rhs for lhs in bounds[0] for rhs in bounds[1]]
        else:
            values = [-((-lhs) // rhs) for lhs in bounds[0] for rhs in bounds[1]]
        return min(values), max(values)
    if op == "neg":
        return -bounds[0][1], -bounds[0][0]
    raise ValueError(f"unsupported ExprSpec op {op!r}")


@dataclass(frozen=True)
class RegionRange:
    """One half-open tensor-region dimension ``[start, start + extent)``."""

    start: Any
    extent: Any


@dataclass(frozen=True)
class RegionSpec:
    """Logical tensor region attached to one tile read or write."""

    dims: tuple[RegionRange, ...] = ()
    dynamic: bool = False
    reason: str = ""


class RegionBuilder:
    """Build a region with syntax such as ``R[m, n:n + 8]``."""

    def __getitem__(self, indices):
        if not isinstance(indices, tuple):
            indices = (indices,)
        dims = []
        for index in indices:
            if isinstance(index, slice):
                if index.step is not None:
                    raise ValueError("region slices do not support a step")
                if index.stop is None:
                    raise ValueError("region slices require a stop")
                start = 0 if index.start is None else index.start
                dims.append(RegionRange(start=start, extent=index.stop - start))
            else:
                dims.append(RegionRange(start=index, extent=1))
        return RegionSpec(dims=tuple(dims))


R = RegionBuilder()
RegionValueType = RegionSpec | tuple[ExprLike, ...] | list[ExprLike]
RegionMapType = Callable[[int, int, int], RegionValueType] | RegionValueType


@dataclass(frozen=True)
class TensorSpec:
    """Logical tensor or buffer in the megakernel spec."""

    name: str
    shape: ShapeType
    dtype: str
    region_map: RegionMapType | None = field(default=None, compare=False)
    region_dynamic: bool = field(default=False, compare=False)
    region_reason: str = field(default="", compare=False)
    base: TensorSpec | None = field(default=None, compare=False)

    @property
    def base_tensor(self) -> TensorSpec:
        """Return the registered tensor behind this access view."""

        return self.base if self.base is not None else self

    @property
    def has_region(self) -> bool:
        """Whether this value carries tile-specific access-region metadata."""

        return self.region_map is not None or self.region_dynamic

    def region(
        self,
        region_map: RegionMapType | None = None,
        *,
        dynamic: bool = False,
        reason: str = "",
    ) -> TensorSpec:
        """Return an access view that identifies the tile's tensor region."""

        base = self.base_tensor
        if dynamic:
            if region_map is not None:
                raise ValueError("a dynamic tensor region cannot also provide a region_map")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("a dynamic tensor region requires a non-empty reason")
            return TensorSpec(
                base.name,
                base.shape,
                base.dtype,
                region_dynamic=True,
                region_reason=reason,
                base=base,
            )
        if region_map is None:
            raise ValueError("tensor.region requires a region_map unless dynamic=True")
        return TensorSpec(
            base.name,
            base.shape,
            base.dtype,
            region_map=region_map,
            base=base,
        )


@dataclass(frozen=True, eq=False)
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


class KernelSpec:
    """Container and fluent builder for one logical megakernel specification."""

    def __init__(self, name: str, attrs: dict[str, Any] | None = None):
        self.name = name
        self.attrs = attrs or {}
        self.vars: dict[str, VarSpec] = {}
        self.scalars: dict[str, ScalarSpec] = {}
        self.tensors: dict[str, TensorSpec] = {}
        self.events: dict[str, EventSpec] = {}
        self.tiles: list[TileSpec] = []

    def var(
        self,
        name: str,
        dtype: str = "int32",
        range: tuple[int, int] | None = None,
    ):
        """Register a symbolic variable."""

        if name in self.vars:
            raise ValueError(f"Duplicate var: {name}")
        if range is not None:
            if (
                not isinstance(range, tuple)
                or len(range) != 2
                or any(not _is_int(value) for value in range)
            ):
                raise TypeError("var range must be a tuple of two integers")
            if range[0] <= 0 or range[0] > range[1]:
                raise ValueError("var range must satisfy 0 < minimum <= maximum")
        var = VarSpec(name=name, dtype=dtype, range=range)
        self.vars[name] = var
        return var

    def scalar(
        self,
        name: str,
        source: tuple[TensorSpec, tuple[ExprLike, ...] | list[ExprLike]],
        dtype: str = "int32",
        range: tuple[int, int] | None = None,
    ):
        """Register a runtime scalar loaded from a device buffer at kernel runtime."""

        if name in self.scalars:
            raise ValueError(f"Duplicate scalar: {name}")
        tensor, index = _normalize_scalar_source(name, source)
        if range is not None:
            if (
                not isinstance(range, tuple)
                or len(range) != 2
                or any(not _is_int(value) for value in range)
            ):
                raise TypeError("scalar range must be a tuple of two integers")
            if range[0] <= 0 or range[0] > range[1]:
                raise ValueError("scalar range must satisfy 0 < minimum <= maximum")
        scalar = ScalarSpec(name=name, source=(tensor, index), dtype=dtype, range=range)
        self.scalars[name] = scalar
        return scalar

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
        read_tensors = _normalize_accesses(reads or (), "reads")
        write_tensors = _normalize_accesses(writes or (), "writes")
        tile = TileSpec(
            name=name,
            impl=impl,
            tile_num=tile_num,
            reads=read_tensors,
            writes=write_tensors,
            attrs=attrs or {},
        )
        self.tiles.append(tile)
        return tile

    def validate(self):
        """Validate the implementation-independent logical specification."""

        from tvm.megakernel.transform import validate_kernel

        return validate_kernel(self)

    def lower(self, options=None):
        """Lower this graph with the runtime-library builder (static by default)."""

        from tvm.megakernel.transform import lower_to_tirx

        return lower_to_tirx(self, options)


def _normalize_scalar_source(name: str, source) -> tuple[TensorSpec, tuple[ExprLike, ...]]:
    label = f"scalar {name!r} source"
    if not isinstance(source, tuple | list) or len(source) != 2:
        raise TypeError(f"{label} must be a (TensorSpec, index) pair")
    tensor, index = source
    if not isinstance(tensor, TensorSpec):
        raise TypeError(f"{label} tensor must be a TensorSpec, got {tensor!r}")
    if tensor.base is not None or tensor.has_region:
        raise ValueError(f"{label} tensor must be a registered base tensor, not a region view")
    if not isinstance(index, tuple | list):
        raise TypeError(f"{label} index must be a tuple or list, got {index!r}")
    rank = _shape_rank(tensor.shape)
    if len(index) != rank:
        raise ValueError(
            f"{label} index rank {len(index)} does not match tensor {tensor.name!r} rank {rank}"
        )
    normalized = tuple(_as_expr_like(entry) for entry in index)
    if expr_scalars(normalized):
        raise ValueError(f"{label} index must not reference runtime scalars")
    return tensor, normalized


def _shape_rank(shape: ShapeType) -> int:
    return len(shape) if isinstance(shape, tuple | list) else 1


def _normalize_accesses(accesses, label: str) -> list[TensorSpec]:
    result = []
    for access in accesses:
        if not isinstance(access, TensorSpec):
            raise TypeError(f"tile {label} entries must be TensorSpec, got {access!r}")
        result.append(access)
    return result
