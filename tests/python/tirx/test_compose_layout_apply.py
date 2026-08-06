# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import itertools

import pytest

import tvm
from tvm.ir import assert_structural_equal
from tvm.tirx import Var
from tvm.tirx.layout import ComposeLayout, S, TileLayout


def _swizzle(value, per_element, swizzle_len, atom_len, swizzle_inner=True):
    chunk = 1 << per_element
    index = value // chunk
    mask = (1 << swizzle_len) - 1
    if swizzle_inner:
        index ^= (index & (mask << atom_len)) >> atom_len
    else:
        index ^= (index & mask) << atom_len
    return index * chunk + value % chunk


def _evaluate(expr, values):
    node_type = type(expr).__name__
    if node_type == "IntImm":
        return int(expr.value)
    if node_type == "Var":
        return values[expr]
    if node_type == "Let":
        bindings = dict(values)
        bindings[expr.var] = _evaluate(expr.value, bindings)
        return _evaluate(expr.body, bindings)
    if node_type in ("Add", "Sub", "Mul", "FloorDiv", "FloorMod"):
        lhs = _evaluate(expr.a, values)
        rhs = _evaluate(expr.b, values)
        if node_type == "Add":
            return lhs + rhs
        if node_type == "Sub":
            return lhs - rhs
        if node_type == "Mul":
            return lhs * rhs
        if node_type == "FloorDiv":
            return lhs // rhs
        return lhs % rhs
    if node_type == "Cast":
        return _evaluate(expr.value, values)
    if node_type == "Call":
        args = [_evaluate(arg, values) for arg in expr.args]
        op_name = str(expr.op.name)
        if op_name == "tirx.bitwise_xor":
            return args[0] ^ args[1]
        if op_name == "tirx.bitwise_and":
            return args[0] & args[1]
        if op_name == "tirx.shift_left":
            return args[0] << args[1]
        if op_name == "tirx.shift_right":
            return args[0] >> args[1]
        raise AssertionError(f"Cannot evaluate call {op_name}")
    raise AssertionError(f"Cannot evaluate node type {node_type}")


def _naive_expr(tile, coords, per_element=3, swizzle_len=3, atom_len=3, inner=True):
    period = 1 << (per_element + swizzle_len + atom_len)
    swizzle = ComposeLayout(
        per_element,
        swizzle_len,
        atom_len,
        TileLayout(S[(period,)]),
        inner,
    )
    return swizzle.apply(tile.apply(*coords)["m"])["m"]


@pytest.mark.parametrize("swizzle_len", [3, 2, 1, 0])
def test_structured_apply_uses_bounded_normal_form(swizzle_len):
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    tile = TileLayout(S[(8, 8) : (64, 1)] + 72)
    layout = ComposeLayout(3, swizzle_len, 3, tile)

    actual = layout.apply(x, y)["m"]
    shaped = layout.apply(x, y, shape=(8, 8))["m"]
    if swizzle_len == 0:
        expected = x * 64 + y + 72
    else:
        mask = (1 << swizzle_len) - 1
        base = x * 64 + (y ^ (((x + 1) & mask) << 3)) + 64
        expected = base ^ 8
    assert_structural_equal(actual, expected)
    assert_structural_equal(shaped, expected)

    for xv, yv in itertools.product(range(8), repeat=2):
        logical = xv * 64 + yv + 72
        assert _evaluate(actual, {x: xv, y: yv}) == _swizzle(logical, 3, swizzle_len, 3)


def test_scalar_apply_keeps_full_swizzle_semantics():
    coord = Var("coord", ty="int32")
    tile = TileLayout(S[(8, 8) : (64, 1)] + 72)
    layout = ComposeLayout(3, 3, 3, tile)
    actual = layout.apply(coord)["m"]

    assert "//" in str(actual)
    assert "%" in str(actual)
    for value in range(256):
        logical = value // 8 * 64 + value % 8 + 72
        assert _evaluate(actual, {coord: value}) == _swizzle(logical, 3, 3, 3)


def test_shape_aware_apply_decomposes_each_logical_coordinate():
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    tile = TileLayout(S[(2, 4, 8, 8) : (256, 64, 8, 1)])
    layout = ComposeLayout(3, 3, 3, tile)

    actual = layout.apply(x, y, shape=(8, 64))["m"]
    expected = x * 64 + (y ^ ((x & 7) << 3))
    assert_structural_equal(actual, expected)
    for xv, yv in itertools.product(range(8), range(64)):
        assert _evaluate(actual, {x: xv, y: yv}) == _swizzle(xv * 64 + yv, 3, 3, 3)


def test_structured_apply_handles_uint32_casts_per_term():
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    x_u32 = tvm.tirx.Cast("uint32", x)
    y_u32 = tvm.tirx.Cast("uint32", y)
    layout = ComposeLayout(3, 3, 3, TileLayout(S[(8, 8) : (64, 1)] + 72))

    actual = layout.apply(x_u32, y_u32, shape=(8, 8))["m"]
    high = x_u32 * 64 + 64
    expected = (high + (y_u32 ^ (((x_u32 + 1) & 7) << 3))) ^ 8
    assert_structural_equal(actual, expected)
    assert "// T.uint32(64)" not in str(actual)

    for xv, yv in itertools.product(range(8), repeat=2):
        logical = xv * 64 + yv + 72
        assert _evaluate(actual, {x: xv, y: yv}) == _swizzle(logical, 3, 3, 3)


def test_structured_apply_classifies_dynamic_high_and_bounded_low_offsets():
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    high = Var("high", ty="int32")
    low = Var("low", ty="int32")
    tile = TileLayout(S[(8, 8) : (64, 1)] + high * 64 + low % 8 + 16)
    layout = ComposeLayout(3, 3, 3, tile)

    actual = layout.apply(x, y)["m"]
    expected = (x * 64 + high * 64 + ((y + low % 8) ^ (((x + high) & 7) << 3))) ^ 16
    assert_structural_equal(actual, expected)
    for xv, yv, hv, lv in itertools.product(range(2), range(8), range(2), range(8)):
        logical = xv * 64 + yv + hv * 64 + lv % 8 + 16
        assert _evaluate(actual, {x: xv, y: yv, high: hv, low: lv}) == _swizzle(logical, 3, 3, 3)


def test_extent_one_coordinate_is_zero_before_classification():
    unused = Var("unused", ty="int32")
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    tile = TileLayout(S[(1, 8, 8) : (-1, 64, 1)] + 8)
    layout = ComposeLayout(3, 3, 3, tile)

    actual = layout.apply(unused, x, y)["m"]
    expected = (x * 64 + (y ^ ((x & 7) << 3))) ^ 8
    assert_structural_equal(actual, expected)
    assert "unused" not in str(actual)


@pytest.mark.parametrize(
    "case",
    ["outer_swizzle", "negative_stride", "unbounded_low", "carry", "symbolic_stride"],
)
def test_structured_apply_falls_back_to_full_swizzle(case):
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    value = Var("value", ty="int32")
    inner = True
    if case == "outer_swizzle":
        tile = TileLayout(S[(8, 8) : (64, 1)] + 8)
        inner = False
    elif case == "negative_stride":
        tile = TileLayout(S[(8, 8) : (-64, 1)] + 8)
    elif case == "unbounded_low":
        tile = TileLayout(S[(8, 8) : (64, 1)] + value)
    elif case == "carry":
        tile = TileLayout(S[(8, 8) : (64, 1)] + 4)
    else:
        tile = TileLayout(S[(8, 8) : (value, 1)] + 8)

    layout = ComposeLayout(3, 3, 3, tile, inner)
    actual = layout.apply(x, y)["m"]
    expected = _naive_expr(tile, (x, y), inner=inner)

    assert type(actual).__name__ == "Let"
    assert_structural_equal(actual.value, tile.apply(x, y)["m"])
    quotient = actual.body
    assert type(quotient).__name__ == "Let"
    assert_structural_equal(quotient.value, actual.var // 8)
    body_vars = tvm.tirx.analysis.undefined_vars(quotient.body)
    assert all(var.same_as(actual.var) or var.same_as(quotient.var) for var in body_vars)

    for xv, yv, vv in itertools.product(range(3), range(8), [-3, 0, 5, 64]):
        values = {x: xv, y: yv, value: vv}
        assert _evaluate(actual, values) == _evaluate(expected, values)


def test_structured_apply_emits_shared_base_address_chain():
    x = Var("x", ty="int32")
    y = Var("y", ty="int32")
    constants = [0, 8, 16, 24, 512, 520, 528, 536]
    addresses = [
        ComposeLayout(3, 3, 3, TileLayout(S[(8, 8) : (64, 1)] + constant)).apply(x, y)["m"]
        for constant in constants
    ]

    base = addresses[0]
    high_base = base + 512
    expected = [
        base,
        base ^ 8,
        base ^ 16,
        base ^ 24,
        high_base,
        high_base ^ 8,
        high_base ^ 16,
        high_base ^ 24,
    ]
    for actual, expected_expr in zip(addresses, expected):
        assert_structural_equal(actual, expected_expr)

    for xv, yv in itertools.product(range(8), repeat=2):
        for constant, address in zip(constants, addresses):
            logical = xv * 64 + yv + constant
            assert _evaluate(address, {x: xv, y: yv}) == _swizzle(logical, 3, 3, 3)
