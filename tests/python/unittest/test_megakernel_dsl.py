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
"""Tests for the implementation-independent megakernel DSL."""

import inspect

import pytest

from tvm.megakernel.dsl import (
    DependencySpec,
    EventSpec,
    KernelSpec,
    TensorSpec,
    TileImpl,
    TileSpec,
    VarSpec,
)


class RecordingTile(TileImpl):
    def __init__(self):
        super().__init__()
        self.indices = []

    def run(self, m_idx, n_idx, k_idx):
        self.indices.append((m_idx, n_idx, k_idx))


def _two_stage_kernel():
    kernel = KernelSpec("two_stage", attrs={"source": "B = f(A); C = g(B)"})
    rows = VarSpec("rows")
    tensor_a = kernel.input("A", (rows, 16), "float16")
    tensor_b = kernel.intermediate("B", (rows, 4), "float32")
    tensor_c = kernel.output("C", (rows,), "float32")
    ready = kernel.event(
        "ready",
        (rows,),
        lambda coord: coord[0] + 1,
        attrs={"meaning": "all partial values are ready"},
    )
    producer_impl = RecordingTile()
    consumer_impl = RecordingTile()
    producer = (
        kernel.tile(
            "producer",
            producer_impl,
            (rows, 4, 1),
            attrs={"purpose": "produce partial values"},
        )
        .read(tensor_a)
        .write(tensor_b)
        .notify(ready, lambda m, n, k: (m,))
    )
    consumer = (
        kernel.tile("consumer", consumer_impl, (rows, 1, 1))
        .read(tensor_b)
        .write(tensor_c)
        .wait(ready, lambda m, n, k: (m,))
    )
    return kernel, producer, consumer, producer_impl


def test_api_exports_fluent_builder_and_direct_tile_impl():
    kernel, producer, consumer, impl = _two_stage_kernel()

    assert kernel.validate() is kernel
    assert isinstance(kernel.tensors["A"], TensorSpec)
    assert not hasattr(kernel.tensors["A"], "role")
    assert isinstance(kernel.events["ready"], EventSpec)
    assert isinstance(producer, TileSpec)
    assert isinstance(producer.notifies[0], DependencySpec)
    assert producer.impl is impl
    assert consumer.waits[0].event is kernel.events["ready"]
    assert producer.attrs == {"purpose": "produce partial values"}
    assert kernel.attrs == {"source": "B = f(A); C = g(B)"}
    assert inspect.signature(KernelSpec.lower).parameters.keys() == {"self"}

    impl.run(1, 2, 3)
    assert impl.indices == [(1, 2, 3)]
    assert RecordingTile.class_name() == "RecordingTile"


@pytest.mark.parametrize(
    "shape,error",
    [
        ((4, object()), "extents must be int or VarSpec"),
        ((4, 0), "extents must be positive"),
        (VarSpec(""), "names must be non-empty"),
    ],
)
def test_validate_rejects_invalid_shapes(shape, error):
    kernel = KernelSpec("invalid_shape")
    kernel.input("A", shape, "float32")
    with pytest.raises((TypeError, ValueError), match=error):
        kernel.validate()


@pytest.mark.parametrize("tile_num", [(1, 1), (1, 1, 1, 1), (1, 0, 1)])
def test_validate_rejects_invalid_three_axis_tile_num(tile_num):
    kernel = KernelSpec("invalid_tile_num")
    kernel.tile("tile", RecordingTile(), tile_num)
    with pytest.raises(ValueError, match="tile_num"):
        kernel.validate()


def test_validate_rejects_non_tile_impl_and_foreign_tensor():
    kernel = KernelSpec("invalid_impl")
    kernel.tile("tile", object(), (1, 1, 1))
    with pytest.raises(TypeError, match="concrete TileImpl instance"):
        kernel.validate()

    owner = KernelSpec("owner")
    foreign = KernelSpec("foreign").input("A", (1,), "float32")
    owner.tile("tile", RecordingTile(), (1, 1, 1)).read(foreign)
    with pytest.raises(ValueError, match="tensor outside this kernel"):
        owner.validate()


def test_validate_rejects_foreign_event_and_missing_notifier():
    kernel = KernelSpec("foreign_event")
    foreign = KernelSpec("foreign").event("ready", (1,), 1)
    kernel.tile("tile", RecordingTile(), (1, 1, 1)).wait(foreign, (0,))
    with pytest.raises(ValueError, match="event outside this kernel"):
        kernel.validate()

    kernel = KernelSpec("missing_notifier")
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("tile", RecordingTile(), (1, 1, 1)).wait(ready, (0,))
    with pytest.raises(ValueError, match="has no notifier"):
        kernel.validate()


@pytest.mark.parametrize(
    "coord_map,error",
    [
        (lambda m, n: (m,), "signature"),
        (lambda m, n, k: m, "tuple or list"),
        (lambda m, n, k: (m, n), "does not match event rank"),
        (lambda m, n, k: (str(m),), "coordinates must be integers"),
    ],
)
def test_validate_rejects_invalid_coord_maps(coord_map, error):
    kernel = KernelSpec("invalid_coord")
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("producer", RecordingTile(), (1, 1, 1)).notify(ready, coord_map)
    kernel.tile("consumer", RecordingTile(), (1, 1, 1)).wait(ready, (0,))
    with pytest.raises((TypeError, ValueError), match=error):
        kernel.validate()


def test_validate_rejects_stateful_coord_map():
    state = {"value": 0}

    def stateful(m, n, k):
        state["value"] += 1
        return (state["value"],)

    kernel = KernelSpec("stateful_coord")
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("producer", RecordingTile(), (1, 1, 1)).notify(ready, stateful)
    kernel.tile("consumer", RecordingTile(), (1, 1, 1)).wait(ready, (0,))
    with pytest.raises(ValueError, match="pure deterministic"):
        kernel.validate()


@pytest.mark.parametrize(
    "init_count,error",
    [
        (0, "must be positive"),
        (lambda: 1, "signature"),
        (lambda coord: 0, "positive integer"),
        (lambda coord: "one", "positive integer"),
    ],
)
def test_validate_rejects_invalid_event_init_count(init_count, error):
    kernel = KernelSpec("invalid_init")
    kernel.event("ready", (1,), init_count)
    with pytest.raises((TypeError, ValueError), match=error):
        kernel.validate()


def test_validate_rejects_stateful_init_count_and_event_cycle():
    state = {"value": 0}

    def stateful(coord):
        state["value"] += 1
        return state["value"]

    kernel = KernelSpec("stateful_init")
    kernel.event("ready", (1,), stateful)
    with pytest.raises(ValueError, match="pure deterministic"):
        kernel.validate()

    kernel = KernelSpec("cycle")
    first_ready = kernel.event("first_ready", (1,), 1)
    second_ready = kernel.event("second_ready", (1,), 1)
    (
        kernel.tile("first", RecordingTile(), (1, 1, 1))
        .wait(second_ready, (0,))
        .notify(first_ready, (0,))
    )
    (
        kernel.tile("second", RecordingTile(), (1, 1, 1))
        .wait(first_ready, (0,))
        .notify(second_ready, (0,))
    )
    with pytest.raises(ValueError, match="acyclic"):
        kernel.validate()


if __name__ == "__main__":
    pytest.main([__file__])
