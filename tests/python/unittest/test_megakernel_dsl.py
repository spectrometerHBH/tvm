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
"""Tests for logical megakernel specs, validation, and static TIRX lowering."""

import inspect

import pytest

import tvm
from tvm.megakernel.dsl import (
    EventSpec,
    ExprSpec,
    KernelSpec,
    R,
    RegionSpec,
    TensorSpec,
    TileImpl,
    TileSpec,
    VarSpec,
    eval_expr_like,
    expr_bounds,
)
from tvm.megakernel.transform import (
    LoweringOptions,
    lower_static_queue_init_to_tirx,
    lower_to_tirx,
    lower_to_tirx_module,
    validate_kernel,
)
from tvm.megakernel.transform.prepare import prepare_tirx_lowering_plan
from tvm.megakernel.transform.validate import _build_instance_event_graph, _spec_view
from tvm.script import tirx as T


class RecordingTile(TileImpl):
    def __init__(self):
        super().__init__()
        self.indices = []

    def run(self, m_idx, n_idx, k_idx):
        self.indices.append((m_idx, n_idx, k_idx))


class CopyTile(TileImpl):
    def __init__(self, source, destination):
        super().__init__()
        self.source = source
        self.destination = destination

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.destination[m_idx] = self.source[m_idx]


class HostInitTile(RecordingTile):
    def __init__(self, hook_name):
        super().__init__()
        self.hook_name = hook_name

    def host_init(self):
        T.evaluate(T.call_packed(self.hook_name))


class PrefetchMarkerTile(RecordingTile):
    @T.inline
    def prefetch(self, m_idx, n_idx, k_idx):
        T.evaluate(T.call_packed("test.prefetch.before_wait"))


class BarrierTile(RecordingTile):
    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        T.evaluate(T.cuda.cta_sync())
        T.evaluate(T.cuda.warp_sync())


class CapturedVarHolder:
    def __init__(self, target=None):
        self.target = target


_GLOBAL_COORD_HOLDER = CapturedVarHolder()


def _global_captured_coord(m, n, k):
    return (_GLOBAL_COORD_HOLDER.target + 1,) if m == 2 else (m,)


class DeviceInitManagedSmemTile(RecordingTile):
    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        smem_manager.alloc((16,), "uint8", name="device_init_managed")


class RunManagedSmemTile(RecordingTile):
    @classmethod
    def init_shared_resources(cls, smem_manager):
        cls.smem_manager = smem_manager

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.alloc((16,), "uint8", name="run_managed")
        self.smem_manager.acquire_all()
        self.smem_manager.release_all()
        self.smem_manager.advance()


class ClassManagedSmemTile(RecordingTile):
    @classmethod
    def init_shared_resources(cls, smem_manager):
        cls.smem_manager = smem_manager
        cls.buffer = smem_manager.alloc((16,), "uint8", name="class_managed")


class SynchronizedClassManagedSmemTile(ClassManagedSmemTile):
    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.acquire_all()
        self.smem_manager.release_all()
        self.smem_manager.advance()


class MissingAdvanceManagedSmemTile(ClassManagedSmemTile):
    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.acquire_all()
        self.smem_manager.release_all()


class WrongOrderManagedSmemTile(ClassManagedSmemTile):
    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.advance()
        self.smem_manager.acquire_all()
        self.smem_manager.release_all()


class ClassAndDeviceManagedSmemTile(RecordingTile):
    @classmethod
    def init_shared_resources(cls, smem_manager):
        cls.smem_manager = smem_manager
        cls.class_buffer = smem_manager.alloc((64,), "uint8", name="class_layout")

    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        smem_manager.alloc((64,), "uint8", name="tile_layout")

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.acquire_all()
        self.smem_manager.release_all()
        self.smem_manager.advance()


class ClassSharedDeviceExclusiveSmemTile(ClassAndDeviceManagedSmemTile):
    @T.inline
    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        smem_manager.alloc((64,), "uint8", name="exclusive_tile_layout", policy="exclusive")


class OversizedClassSmemTile(RecordingTile):
    @classmethod
    def init_shared_resources(cls, smem_manager):
        cls.smem_manager = smem_manager
        cls.buffer = smem_manager.alloc((5120,), "uint8", name="oversized_class")

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.acquire_all()
        self.smem_manager.release_all()
        self.smem_manager.advance()


class SmallClassSmemTile(RecordingTile):
    @classmethod
    def init_shared_resources(cls, smem_manager):
        cls.smem_manager = smem_manager
        cls.buffer = smem_manager.alloc((16,), "uint8", name="small_class")

    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.acquire_all()
        self.smem_manager.release_all()
        self.smem_manager.advance()


class FinalizeOwnerTile(RecordingTile):
    finalized_key = None

    @classmethod
    def finalize_shared_resources(cls, smem_manager):
        cls.finalized_key = smem_manager.cur_tile_name


def _two_stage_kernel():
    kernel = KernelSpec("two_stage", attrs={"source": "B = f(A); C = g(B)"})
    rows = kernel.var("rows", "int32")
    tensor_a = kernel.tensor("A", (rows, 16), "float16")
    tensor_b = kernel.tensor("B", (rows, 4), "float32")
    tensor_c = kernel.tensor("C", (rows,), "float32")
    ready = kernel.event(
        "ready",
        (rows,),
        4,
        attrs={"meaning": "all partial values are ready"},
    )
    producer_impl = RecordingTile()
    consumer_impl = RecordingTile()
    producer = kernel.tile(
        "producer",
        producer_impl,
        (rows, 4, 1),
        reads=[tensor_a],
        writes=[tensor_b],
        attrs={"purpose": "produce partial values"},
    ).notify(ready, lambda m, n, k: (m,))
    consumer = kernel.tile(
        "consumer",
        consumer_impl,
        (rows, 1, 1),
        reads=[tensor_b],
        writes=[tensor_c],
    ).wait(ready, lambda m, n, k: (m,))
    return kernel, producer, consumer, producer_impl


def test_api_matches_parser_style_contract():
    kernel, producer, consumer, impl = _two_stage_kernel()

    assert kernel.validate() is kernel
    assert kernel.vars["rows"] == VarSpec("rows", "int32")
    assert isinstance(kernel.tensors["A"], TensorSpec)
    assert isinstance(kernel.events["ready"], EventSpec)
    assert isinstance(producer, TileSpec)
    assert producer.notifies[0][0] is kernel.events["ready"]
    assert consumer.waits[0][0] is kernel.events["ready"]
    assert producer.impl is impl
    assert producer.attrs == {"purpose": "produce partial values"}
    assert inspect.signature(KernelSpec.lower).parameters.keys() == {"self", "options"}
    assert not hasattr(kernel, "input")
    assert not hasattr(producer, "read")

    impl.run(1, 2, 3)
    assert impl.indices == [(1, 2, 3)]
    assert RecordingTile.class_name() == "RecordingTile"


def test_event_specs_use_identity_equality_and_hashing():
    first = EventSpec("ready", (1,), 1)
    second = EventSpec("ready", (1,), 1)

    assert first is not second
    assert first != second
    assert len({first, second}) == 2


def test_validate_kernel_returns_the_spec_and_rejects_non_specs():
    kernel, _, _, _ = _two_stage_kernel()

    assert validate_kernel(kernel) is kernel
    with pytest.raises(TypeError, match="KernelSpec"):
        validate_kernel(object())


def test_bounded_var_expressions_evaluate_and_bound_shapes():
    kernel = KernelSpec("bounded")
    rows = kernel.var("rows", range=(1, 17))
    tiles = (rows + 3).ceildiv(4)
    kernel.tensor("A", (tiles, rows * 2), "float16")

    assert isinstance(tiles, ExprSpec)
    assert eval_expr_like(tiles, {rows: 5}) == 2
    assert expr_bounds(tiles) == (1, 5)
    assert kernel.validate() is kernel


@pytest.mark.parametrize(
    "bounds,error",
    [((0, 4), "0 < minimum"), ((5, 4), "0 < minimum"), ([1, 4], "tuple")],
)
def test_var_range_rejects_invalid_bounds(bounds, error):
    kernel = KernelSpec("bad_bounds")
    with pytest.raises((TypeError, ValueError), match=error):
        kernel.var("rows", range=bounds)


def test_semantic_validation_checks_exact_event_counts():
    kernel = KernelSpec("event_count")
    ready = kernel.event("ready", (2,), lambda coord: 2 - coord[0])
    kernel.tile("producer", RecordingTile(), (3, 1, 1)).notify(ready, lambda m, n, k: (m // 2,))
    kernel.tile("consumer", RecordingTile(), (2, 1, 1)).wait(ready, lambda m, n, k: (m,))
    kernel.validate()

    invalid = KernelSpec("invalid_event_count")
    invalid_ready = invalid.event("ready", (2,), 1)
    invalid.tile("producer", RecordingTile(), (3, 1, 1)).notify(
        invalid_ready, lambda m, n, k: (m // 2,)
    )
    invalid.tile("consumer", RecordingTile(), (2, 1, 1)).wait(invalid_ready, lambda m, n, k: (m,))
    with pytest.raises(ValueError, match="expects init_count"):
        invalid.validate()


def test_semantic_validation_exhausts_bounded_symbolic_event_counts():
    kernel = KernelSpec("bounded_event_count")
    rows = kernel.var("rows", range=(1, 3))
    ready = kernel.event("ready", (rows,), 1)
    kernel.tile("producer", RecordingTile(), (rows, 1, 1)).notify(ready, lambda m, n, k: (m,))
    kernel.tile("consumer", RecordingTile(), (rows, 1, 1)).wait(ready, lambda m, n, k: (m,))

    assert kernel.validate() is kernel

    invalid = KernelSpec("bounded_event_count_failure")
    invalid_rows = invalid.var("rows", range=(1, 3))
    invalid_ready = invalid.event(
        "ready",
        (invalid_rows,),
        lambda coord: 2 if coord[0] == 2 else 1,
    )
    invalid.tile("producer", RecordingTile(), (invalid_rows, 1, 1)).notify(
        invalid_ready, lambda m, n, k: (m,)
    )
    invalid.tile("consumer", RecordingTile(), (invalid_rows, 1, 1)).wait(
        invalid_ready, lambda m, n, k: (m,)
    )
    with pytest.raises(ValueError, match=r"coord \(2,\) expects init_count 2"):
        invalid.validate()


def test_semantic_samples_unbounded_symbolic_event_counts():
    kernel = KernelSpec("unbounded_event_count")
    rows = kernel.var("rows")
    ready = kernel.event("ready", (rows,), 2)
    kernel.tile("producer", RecordingTile(), (rows, 1, 1)).notify(ready, lambda m, n, k: (m,))
    kernel.tile("consumer", RecordingTile(), (rows, 1, 1)).wait(ready, lambda m, n, k: (m,))

    with pytest.raises(ValueError, match="expects init_count 2"):
        kernel.validate()


def test_symbolic_event_shape_checks_zero_notify_coordinates_in_full_domain():
    kernel = KernelSpec("symbolic_full_event_domain")
    rows = kernel.var("rows", range=(1, 3))
    ready = kernel.event("ready", (rows,), 1)
    kernel.tile("producer", RecordingTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("consumer", RecordingTile(), (1, 1, 1)).wait(ready, (0,))

    with pytest.raises(ValueError, match="expects init_count 1.*full domain"):
        kernel.validate()


def test_constant_capacity_event_checks_only_runtime_active_coordinates():
    kernel = KernelSpec("runtime_active_event_capacity")
    active_rows = kernel.var("active_rows")
    ready = kernel.event("ready", (64,), 12)
    kernel.tile("producer", RecordingTile(), (active_rows, 12, 1)).notify(
        ready, lambda m, n, k: (m,)
    )
    kernel.tile("consumer", RecordingTile(), (active_rows, 1, 1)).wait(ready, lambda m, n, k: (m,))

    assert kernel.validate() is kernel


def test_constant_event_shape_ignores_unused_callable_captures_for_coverage():
    kernel = KernelSpec("constant_event_full_domain")
    unused = kernel.var("unused", range=(1, 1))
    ready = kernel.event("ready", (2,), 1)

    def coord(m, n, k, captured=unused):
        del captured
        return (m,)

    kernel.tile("producer", RecordingTile(), (1, 1, 1)).notify(ready, coord)
    kernel.tile("consumer", RecordingTile(), (1, 1, 1)).wait(ready, coord)

    with pytest.raises(ValueError, match="expects init_count 1.*full domain"):
        kernel.validate()


def test_instance_event_graph_uses_one_synthetic_node_per_event_coordinate():
    kernel = KernelSpec("linear_event_coordinate_graph")
    event = kernel.event("ready", (1,), 512)
    tensor = kernel.tensor("buffer", (512,), "float32")
    kernel.tile("producer", RecordingTile(), (512, 1, 1), writes=[tensor]).notify(event, (0,))
    kernel.tile("consumer", RecordingTile(), (512, 1, 1), reads=[tensor]).wait(event, (0,))

    assert validate_kernel(kernel) is kernel
    graph = _build_instance_event_graph(_spec_view(kernel), {})

    assert sum(map(len, graph.adjacency.values())) == 1024


def test_semantic_samples_large_bounded_regions_and_ignores_unrelated_vars():
    kernel = KernelSpec("large_bounded_region")
    rows = kernel.var("rows", range=(1, 8192))
    kernel.var("unrelated", range=(1, 1_000_000))
    tensor = kernel.tensor("buffer", (rows,), "float32")
    kernel.tile(
        "writer",
        RecordingTile(),
        (1, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[0])],
    )

    assert kernel.validate() is kernel


def test_semantic_samples_large_static_event_space_without_rejecting_it():
    kernel = KernelSpec("large_event_space")
    extent = 262_145
    ready = kernel.event("ready", (extent,), 1)
    kernel.tile("producer", RecordingTile(), (extent, 1, 1)).notify(ready, lambda m, n, k: (m,))
    kernel.tile("consumer", RecordingTile(), (extent, 1, 1)).wait(ready, lambda m, n, k: (m,))

    assert kernel.validate() is kernel


def test_large_static_event_uses_total_cardinality_check():
    kernel = KernelSpec("large_event_count_mismatch")
    extent = 262_145
    ready = kernel.event("ready", (extent,), 2)
    kernel.tile("producer", RecordingTile(), (extent, 1, 1)).notify(ready, lambda m, n, k: (m,))
    kernel.tile("consumer", RecordingTile(), (extent, 1, 1)).wait(ready, lambda m, n, k: (m,))

    with pytest.raises(ValueError, match="expects init_count 2.*full domain"):
        kernel.validate()


def test_event_variable_discovery_covers_closure_default_and_bound_self():
    class BoundMapper:
        def __init__(self, target):
            self.target = target

        def coord(self, m, n, k):
            return (self.target + 1,) if m == 2 else (m,)

    for capture_kind in ("closure", "default", "bound_self", "global"):
        kernel = KernelSpec(f"captured_event_var_{capture_kind}")
        target = kernel.var("target", range=(1, 1))
        ready = kernel.event("ready", (3,), 1)
        if capture_kind == "closure":

            def coord(m, n, k):
                return (target + 1,) if m == 2 else (m,)

        elif capture_kind == "default":

            def coord(m, n, k, captured=target):
                return (captured + 1,) if m == 2 else (m,)

        else:
            if capture_kind == "bound_self":
                coord = BoundMapper(target).coord
            else:
                _GLOBAL_COORD_HOLDER.target = target
                coord = _global_captured_coord
        kernel.tile("producer", RecordingTile(), (3, 1, 1)).notify(ready, coord)
        kernel.tile("consumer", RecordingTile(), (3, 1, 1)).wait(ready, lambda m, n, k: (m,))

        assert kernel.validate() is kernel


def test_symbolic_environment_enumeration_has_a_combined_work_budget():
    kernel = KernelSpec("combined_validation_budget")
    rows = kernel.var("rows", range=(1, 4096))
    ready = kernel.event("ready", (1024,), 1)
    calls = {"count": 0}

    def coord(m, n, k):
        calls["count"] += 1
        return (m + rows - rows,)

    kernel.tile("producer", RecordingTile(), (1024, 1, 1)).notify(ready, coord)
    kernel.tile("consumer", RecordingTile(), (1024, 1, 1)).wait(ready, coord)

    assert kernel.validate() is kernel
    assert calls["count"] < 100_000


def test_tensor_regions_validate_matching_event_coordinates():
    kernel = KernelSpec("regions")
    tensor = kernel.tensor("buffer", (8,), "float32")
    ready = kernel.event("ready", (2,), 1)
    kernel.tile(
        "producer",
        RecordingTile(),
        (2, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[m * 4 : (m + 1) * 4])],
    ).notify(ready, lambda m, n, k: (m,))
    consumer = kernel.tile(
        "consumer",
        RecordingTile(),
        (2, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[m * 4 : (m + 1) * 4])],
    ).wait(ready, lambda m, n, k: (m,))

    assert kernel.validate() is kernel
    consumer.waits[0] = (ready, lambda m, n, k: (1 - m,))
    with pytest.raises(ValueError, match="no producer notifying that coord"):
        kernel.validate()


def test_semantic_rejects_waited_coord_without_overlapping_writer_region():
    kernel = KernelSpec("waited_coord_wrong_region")
    tensor = kernel.tensor("buffer", (2,), "float32")
    ready = kernel.event("ready", (1,), 1)
    kernel.tile(
        "producer",
        RecordingTile(),
        (1, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[0])],
    ).notify(ready, (0,))
    kernel.tile(
        "consumer",
        RecordingTile(),
        (1, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[1])],
    ).wait(ready, (0,))

    with pytest.raises(ValueError, match="no producer notifying that coord"):
        kernel.validate()


def test_tensor_dependency_requires_ordering_from_every_overlapping_writer():
    kernel = KernelSpec("partial_waited_region_coverage")
    tensor = kernel.tensor("buffer", (2,), "float32")
    ready = kernel.event("ready", (2,), 1)
    kernel.tile(
        "left_producer",
        RecordingTile(),
        (1, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[0])],
    ).notify(ready, (0,))
    kernel.tile(
        "right_producer",
        RecordingTile(),
        (1, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[1])],
    ).notify(ready, (1,))
    kernel.tile(
        "consumer",
        RecordingTile(),
        (1, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[0:2])],
    ).wait(ready, (0,))

    with pytest.raises(ValueError, match="right_producer.*without an event dependency"):
        kernel.validate()


def test_static_region_dependency_accepts_transitive_event_ordering():
    kernel = KernelSpec("transitive_region_dependency")
    tensor = kernel.tensor("buffer", (1,), "float32")
    first = kernel.event("first", (1,), 1)
    second = kernel.event("second", (1,), 1)
    kernel.tile(
        "producer",
        RecordingTile(),
        (1, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[0])],
    ).notify(first, (0,))
    kernel.tile("middle", RecordingTile(), (1, 1, 1)).wait(first, (0,)).notify(second, (0,))
    kernel.tile(
        "consumer",
        RecordingTile(),
        (1, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[0])],
    ).wait(second, (0,))

    assert kernel.validate() is kernel


def test_static_region_transitive_ordering_requires_matching_event_coordinates():
    kernel = KernelSpec("transitive_region_coord_mismatch")
    tensor = kernel.tensor("buffer", (1,), "float32")
    first = kernel.event("first", (2,), 1)
    second = kernel.event("second", (1,), 1)
    kernel.tile(
        "producer",
        RecordingTile(),
        (1, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[0])],
    ).notify(first, (0,))
    kernel.tile("other_producer", RecordingTile(), (1, 1, 1)).notify(first, (1,))
    kernel.tile("middle", RecordingTile(), (1, 1, 1)).wait(first, (1,)).notify(second, (0,))
    kernel.tile(
        "consumer",
        RecordingTile(),
        (1, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[0])],
    ).wait(second, (0,))
    # This tile is structurally unrelated to the producer-to-consumer path and
    # must not make the coordinate graph inexact.
    kernel.tile("unrelated_large", RecordingTile(), (65_537, 1, 1))

    with pytest.raises(ValueError, match="without an event dependency"):
        kernel.validate()


@pytest.mark.parametrize("access_kind", ["bare", "dynamic"])
def test_tensor_transitive_ordering_requires_matching_event_coordinates(access_kind):
    kernel = KernelSpec(f"transitive_{access_kind}_coord_mismatch")
    tensor = kernel.tensor("buffer", (1,), "float32")
    first = kernel.event("first", (2,), 1)
    second = kernel.event("second", (1,), 1)
    access = (
        tensor
        if access_kind == "bare"
        else tensor.region(dynamic=True, reason="runtime-selected row")
    )
    kernel.tile(
        "producer",
        RecordingTile(),
        (1, 1, 1),
        writes=[access],
    ).notify(first, (0,))
    kernel.tile("other_producer", RecordingTile(), (1, 1, 1)).notify(first, (1,))
    kernel.tile("middle", RecordingTile(), (1, 1, 1)).wait(first, (1,)).notify(second, (0,))
    kernel.tile(
        "consumer",
        RecordingTile(),
        (1, 1, 1),
        reads=[access],
    ).wait(second, (0,))

    with pytest.raises(ValueError, match="without an event dependency"):
        kernel.validate()


def test_static_region_dependency_rejects_reverse_event_ordering():
    kernel = KernelSpec("reverse_region_dependency")
    tensor = kernel.tensor("buffer", (1,), "float32")
    reverse = kernel.event("reverse", (1,), 1)
    kernel.tile(
        "producer",
        RecordingTile(),
        (1, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[0])],
    ).wait(reverse, (0,))
    kernel.tile(
        "consumer",
        RecordingTile(),
        (1, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[0])],
    ).notify(reverse, (0,))

    with pytest.raises(ValueError, match="without an event dependency"):
        kernel.validate()


def test_bare_tensor_dependency_accepts_transitive_event_ordering():
    kernel = KernelSpec("transitive_tensor_dependency")
    tensor = kernel.tensor("buffer", (1,), "float32")
    first = kernel.event("first", (1,), 1)
    second = kernel.event("second", (1,), 1)
    kernel.tile(
        "producer",
        RecordingTile(),
        (1, 1, 1),
        writes=[tensor],
    ).notify(first, (0,))
    kernel.tile("middle", RecordingTile(), (1, 1, 1)).wait(first, (0,)).notify(second, (0,))
    kernel.tile(
        "consumer",
        RecordingTile(),
        (1, 1, 1),
        reads=[tensor],
    ).wait(second, (0,))

    assert kernel.validate() is kernel


def test_tensor_regions_allow_disjoint_writers_and_reject_overlap():
    kernel = KernelSpec("region_writers")
    tensor = kernel.tensor("buffer", (8,), "float32")
    kernel.tile(
        "left",
        RecordingTile(),
        (1, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[0:4])],
    )
    right = kernel.tile(
        "right",
        RecordingTile(),
        (1, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[4:8])],
    )
    kernel.validate()

    right.writes[0] = tensor.region(lambda m, n, k: R[3:8])
    with pytest.raises(ValueError, match="multiple producers with overlapping regions"):
        kernel.validate()


def test_tensor_regions_reject_overlapping_instances_of_one_writer_tile():
    kernel = KernelSpec("same_tile_region_writers")
    tensor = kernel.tensor("buffer", (1,), "float32")
    ready = kernel.event("ready", (2,), 1)
    kernel.tile(
        "producer",
        RecordingTile(),
        (2, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[0])],
    ).notify(ready, lambda m, n, k: (m,))
    kernel.tile(
        "consumer",
        RecordingTile(),
        (1, 1, 1),
        reads=[tensor.region(lambda m, n, k: R[0])],
    ).wait(ready, (0,))

    with pytest.raises(ValueError, match="multiple producers with overlapping regions"):
        kernel.validate()


def test_tensor_regions_reject_out_of_bounds_single_access():
    kernel = KernelSpec("region_bounds")
    tensor = kernel.tensor("buffer", (8,), "float32")
    kernel.tile(
        "producer",
        RecordingTile(),
        (2, 1, 1),
        writes=[tensor.region(lambda m, n, k: R[m * 8 : m * 8 + 8])],
    )

    with pytest.raises(ValueError, match="out of bounds"):
        kernel.validate()


def test_dynamic_tensor_region_requires_an_explicit_reason_only():
    tensor = TensorSpec("buffer", (8,), "float32")
    dynamic = tensor.region(dynamic=True, reason="data-dependent routing")
    assert dynamic.base_tensor is tensor
    assert dynamic.has_region
    assert RegionSpec(dynamic=True).dynamic
    with pytest.raises(ValueError, match="non-empty reason"):
        tensor.region(dynamic=True)
    with pytest.raises(ValueError, match="cannot also provide"):
        tensor.region(R[0:1], dynamic=True)


def test_dynamic_tensor_regions_allow_unbounded_runtime_dataflow():
    kernel = KernelSpec("dynamic_regions")
    rows = kernel.var("rows")
    tensor = kernel.tensor("buffer", (rows,), "float32")
    ready = kernel.event("ready", (rows,), 1)
    kernel.tile(
        "producer",
        RecordingTile(),
        (rows, 1, 1),
        writes=[tensor.region(dynamic=True, reason="runtime scatter destinations")],
    ).notify(ready, lambda m, n, k: (m,))
    kernel.tile(
        "consumer",
        RecordingTile(),
        (rows, 1, 1),
        reads=[tensor.region(dynamic=True, reason="runtime gather sources")],
    ).wait(ready, lambda m, n, k: (m,))

    assert kernel.validate() is kernel


def test_transform_exports_only_the_verify_and_build_contract():
    from tvm.megakernel import transform

    for old_name in (
        "Action",
        "EdgeBindingPlan",
        "EmissionContext",
        "MegakernelBackend",
        "MegakernelLowerer",
        "SchedulerFetchProgram",
        "TileActionProgram",
        "TileEmitter",
        "HookAction",
        "RunAction",
        "SmemEnterAction",
        "SmemExitAction",
    ):
        assert not hasattr(transform, old_name)
    for removed_name in (
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
        "SemanticEdge",
        "SemanticPlan",
        "TIRXStaticBackend",
        "TileProgram",
        "WaitStep",
        "build_semantic_plan",
        "logical_edges",
        "lower_execution_plan",
        "make_static_execution_plan",
        "validate_semantic_plan",
    ):
        assert not hasattr(transform, removed_name)
    assert set(transform.__all__) == {
        "LowerMegakernelDSL",
        "LoweringOptions",
        "lower_static_queue_init_to_tirx",
        "lower_to_tirx",
        "lower_to_tirx_module",
        "validate_kernel",
    }


def test_demo_uses_current_verify_and_build_api_and_lowers():
    from tvm.megakernel.demo import dsl as demo

    assert validate_kernel(demo.kernel) is demo.kernel
    assert set(demo.kernel.tensors) == {"A", "B", "C"}
    assert [tensor.name for tensor in demo.stage1.reads] == ["A"]
    assert [tensor.name for tensor in demo.stage1.writes] == ["B"]
    assert [tensor.name for tensor in demo.stage2.reads] == ["B"]
    assert [tensor.name for tensor in demo.stage2.writes] == ["C"]
    assert not hasattr(demo.kernel, "input")
    assert not hasattr(demo.stage1, "read")

    lowered = demo.kernel.lower()
    assert lowered.attrs["global_symbol"] == "two_stage_reduce"
    lowered_script = lowered.script()
    assert "tirx.megakernel.smem" not in lowered_script
    assert "T.ptx.mbarrier.init" in lowered_script
    assert "T.ptx.mbarrier.try_wait" in lowered_script
    assert "T.ptx.mbarrier.arrive" in lowered_script


@pytest.mark.parametrize("shape,error", [((4, object()), "extents"), ((4, 0), "positive")])
def test_validate_rejects_invalid_shapes(shape, error):
    kernel = KernelSpec("invalid_shape")
    kernel.tensor("A", shape, "float32")
    with pytest.raises((TypeError, ValueError), match=error):
        kernel.validate()


def test_validate_rejects_foreign_and_invalid_vars():
    kernel = KernelSpec("invalid_var")
    kernel.tensor("A", (VarSpec("rows"),), "float32")
    with pytest.raises(ValueError, match="VarSpec outside this kernel"):
        kernel.validate()

    kernel = KernelSpec("invalid_var_dtype")
    kernel.var("rows", "")
    with pytest.raises(TypeError, match="dtype"):
        kernel.validate()


@pytest.mark.parametrize("tile_num", [(1, 1), (1, 1, 1, 1), (1, 0, 1)])
def test_validate_rejects_invalid_three_axis_tile_num(tile_num):
    kernel = KernelSpec("invalid_tile_num")
    kernel.tile("tile", RecordingTile(), tile_num)
    with pytest.raises(ValueError, match="tile_num"):
        kernel.validate()


def test_validate_rejects_non_impl_foreign_tensor_and_multiple_producers():
    kernel = KernelSpec("invalid_impl")
    kernel.tile("tile", object(), (1, 1, 1))
    with pytest.raises(TypeError, match="concrete TileImpl"):
        kernel.validate()

    owner = KernelSpec("owner")
    foreign = KernelSpec("foreign").tensor("A", (1,), "float32")
    owner.tile("tile", RecordingTile(), (1, 1, 1), reads=[foreign])
    with pytest.raises(ValueError, match="tensor outside this kernel"):
        owner.validate()

    kernel = KernelSpec("multiple_producers")
    tensor = kernel.tensor("A", (1,), "float32")
    kernel.tile("first", RecordingTile(), (1, 1, 1), writes=[tensor])
    kernel.tile("second", RecordingTile(), (1, 1, 1), writes=[tensor])
    with pytest.raises(ValueError, match="multiple producers"):
        kernel.validate()


def test_validate_rejects_foreign_event_missing_notifier_and_bad_tuple():
    kernel = KernelSpec("foreign_event")
    foreign = KernelSpec("foreign").event("ready", (1,), 1)
    kernel.tile("tile", RecordingTile(), (1, 1, 1)).wait(foreign, (0,))
    with pytest.raises(ValueError, match="event outside this kernel"):
        kernel.validate()

    kernel = KernelSpec("missing_notifier")
    ready = kernel.event("ready", (1,), 1)
    tile = kernel.tile("tile", RecordingTile(), (1, 1, 1)).wait(ready, (0,))
    with pytest.raises(ValueError, match="has no notifier"):
        kernel.validate()
    tile.waits[0] = [ready, (0,)]
    with pytest.raises(TypeError, match="invalid wait dependency"):
        kernel.validate()


def test_validate_rejects_notify_without_consumer():
    kernel = KernelSpec("missing_consumer")
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("producer", RecordingTile(), (1, 1, 1)).notify(ready, (0,))

    with pytest.raises(ValueError, match="notified but has no consumer"):
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


def test_validate_rejects_stateful_callables():
    state = {"value": 0}

    def stateful_coord(m, n, k):
        state["value"] += 1
        return (state["value"],)

    kernel = KernelSpec("stateful_coord")
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("producer", RecordingTile(), (1, 1, 1)).notify(ready, stateful_coord)
    kernel.tile("consumer", RecordingTile(), (1, 1, 1)).wait(ready, (0,))
    with pytest.raises(ValueError, match="pure deterministic"):
        kernel.validate()


def test_static_lowering_rejects_event_cycles_accepted_by_the_spec():
    kernel = KernelSpec("cycle")
    first_ready = kernel.event("first_ready", (1,), 1)
    second_ready = kernel.event("second_ready", (1,), 1)
    kernel.tile("first", RecordingTile(), (1, 1, 1)).wait(second_ready, (0,)).notify(
        first_ready, (0,)
    )
    kernel.tile("second", RecordingTile(), (1, 1, 1)).wait(first_ready, (0,)).notify(
        second_ready, (0,)
    )
    assert kernel.validate() is kernel
    with pytest.raises(ValueError, match="static schedule.*acyclic"):
        lower_to_tirx(kernel)


def test_static_lowering_accepts_coord_disjoint_coarse_tile_cycle():
    kernel = KernelSpec("coord_disjoint_tile_cycle")
    first_ready = kernel.event("first_ready", (2,), 1)
    second_ready = kernel.event("second_ready", (2,), 1)
    kernel.tile("first", RecordingTile(), (1, 1, 1)).wait(second_ready, (1,)).notify(
        first_ready, (0,)
    )
    kernel.tile("second", RecordingTile(), (1, 1, 1)).wait(first_ready, (1,)).notify(
        second_ready, (0,)
    )
    kernel.tile("first_entry", RecordingTile(), (1, 1, 1)).notify(first_ready, (1,))
    kernel.tile("second_entry", RecordingTile(), (1, 1, 1)).notify(second_ready, (1,))

    assert isinstance(lower_to_tirx(kernel), tvm.tirx.PrimFunc)


def test_static_schedule_topologically_orders_waiting_tile_phases():
    kernel = KernelSpec("waiting_phase_order")
    entry_ready = kernel.event("entry_ready", (1,), 1)
    middle_ready = kernel.event("middle_ready", (1,), 1)
    kernel.tile("entry", RecordingTile(), (1, 1, 1)).notify(entry_ready, (0,))
    # Register the consumer first to ensure spec source order is unsafe.
    kernel.tile("consumer", RecordingTile(), (1, 1, 1)).wait(middle_ready, (0,))
    kernel.tile("middle", RecordingTile(), (1, 1, 1)).wait(entry_ready, (0,)).notify(
        middle_ready, (0,)
    )

    prepared = prepare_tirx_lowering_plan(kernel, LoweringOptions())
    labels = [
        phase.label
        for phase in prepared.static_schedule.phases
        if phase.label in {"entry", "middle", "consumer"}
    ]
    assert labels == ["entry", "middle", "consumer"]


def test_static_queue_initializer_partitions_rows_by_cta():
    kernel = KernelSpec("partitioned_queue")
    kernel.tile("tile", RecordingTile(), (4, 1, 1))

    script = lower_static_queue_init_to_tirx(
        kernel,
        LoweringOptions(attrs={"sm_count": 4}),
    ).script()

    assert "v = T.cta_id([4])" in script
    assert "if buffer % 4 == v:" in script
    assert "queue[v, buffer // 4]" in script


def test_static_schedule_uses_coordinate_projection_to_break_coarse_cycle():
    kernel = KernelSpec("coordinate_projected_phase_order")
    first_ready = kernel.event("first_ready", (2,), 1)
    second_ready = kernel.event("second_ready", (1,), 1)
    kernel.tile("first", RecordingTile(), (1, 1, 1)).wait(second_ready, (0,)).notify(
        first_ready, (0,)
    )
    kernel.tile("second", RecordingTile(), (1, 1, 1)).wait(first_ready, (1,)).notify(
        second_ready, (0,)
    )
    kernel.tile("entry", RecordingTile(), (1, 1, 1)).notify(first_ready, (1,))

    prepared = prepare_tirx_lowering_plan(kernel, LoweringOptions())
    labels = [
        phase.label
        for phase in prepared.static_schedule.phases
        if phase.label in {"entry", "first", "second"}
    ]
    assert labels == ["entry", "second", "first"]


def test_static_lowering_rejects_instance_dag_with_cyclic_tile_projection():
    kernel = KernelSpec("cyclic_tile_projection")
    first_ready = kernel.event("first_ready", (2,), 1)
    second_ready = kernel.event("second_ready", (3,), 1)
    kernel.tile("first", RecordingTile(), (2, 1, 1)).wait(
        second_ready, lambda m, n, k: (m,)
    ).notify(first_ready, lambda m, n, k: (m,))
    kernel.tile("second", RecordingTile(), (2, 1, 1)).wait(
        first_ready, lambda m, n, k: (m,)
    ).notify(second_ready, lambda m, n, k: (m + 1,))
    kernel.tile("entry", RecordingTile(), (1, 1, 1)).notify(second_ready, (0,))

    with pytest.raises(ValueError, match="project to an acyclic tile-phase order"):
        lower_to_tirx(kernel)


@pytest.mark.parametrize(
    "init_count,error",
    [
        (0, "positive"),
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


def test_prefetch_is_emitted_before_event_wait():
    kernel = KernelSpec("prefetch_before_wait")
    tensor = kernel.tensor("buffer", (1,), "float32")
    ready = kernel.event("ready", (1,), 1)
    kernel.tile("producer", RecordingTile(), (1, 1, 1), writes=[tensor]).notify(ready, (0,))
    kernel.tile("consumer", PrefetchMarkerTile(), (1, 1, 1), reads=[tensor]).wait(ready, (0,))

    script = lower_to_tirx(kernel).script()
    marker = script.index('T.call_packed("test.prefetch.before_wait")')
    assert marker < script.index("T.ptx.ld_global_acquire", marker)


def test_default_backend_lowers_release_event_publication():
    kernel = KernelSpec("release_event_publication")
    event = kernel.event("ready", (1,), 1)
    kernel.tile("producer", RecordingTile(), (1, 1, 1)).notify(event, (0,))
    kernel.tile("consumer", RecordingTile(), (1, 1, 1)).wait(event, (0,))
    script = lower_to_tirx(kernel).script()
    fence = script.index("T.cuda.thread_fence()")
    atomic = script.index("T.cuda.atomic_add", fence)
    assert fence < atomic


def test_managed_smem_requires_acquire_and_release_for_the_same_tile_key():
    kernel = KernelSpec("missing_smem_phase")
    kernel.tile("managed", DeviceInitManagedSmemTile(), (1, 1, 1))

    with pytest.raises(
        tvm.error.DiagnosticError,
        match=r"tile 'managed'.*acquire_all\(\).*release_all\(\)",
    ):
        lower_to_tirx(kernel)


def test_class_managed_smem_requires_each_tile_to_acquire_and_release():
    invalid = KernelSpec("missing_class_smem_phase")
    invalid.tile("managed", ClassManagedSmemTile(), (1, 1, 1))
    with pytest.raises(
        tvm.error.DiagnosticError,
        match=r"tile 'managed'.*acquire_all\(\).*release_all\(\)",
    ):
        lower_to_tirx(invalid)

    valid = KernelSpec("synchronized_class_smem_phase")
    valid.tile("managed", SynchronizedClassManagedSmemTile(), (1, 1, 1))
    script = lower_to_tirx(valid).script()
    assert "T.ptx.mbarrier.try_wait" in script
    assert "T.ptx.mbarrier.arrive" in script


def test_managed_smem_requires_phase_advance_after_release():
    kernel = KernelSpec("missing_smem_advance")
    kernel.tile("managed", MissingAdvanceManagedSmemTile(), (1, 1, 1))

    with pytest.raises(tvm.error.DiagnosticError, match=r"tile 'managed'.*advance\(\)"):
        lower_to_tirx(kernel)


def test_managed_smem_requires_acquire_release_advance_order():
    kernel = KernelSpec("wrong_smem_phase_order")
    kernel.tile("managed", WrongOrderManagedSmemTile(), (1, 1, 1))

    with pytest.raises(
        tvm.error.DiagnosticError,
        match=r"tile 'managed'.*acquire_all\(\).*release_all\(\).*advance\(\).*in order",
    ):
        lower_to_tirx(kernel)


def test_run_alloc_without_device_init_uses_the_tile_owner():
    kernel = KernelSpec("run_smem_without_device_init")
    kernel.tile("managed", RunManagedSmemTile(), (1, 1, 1))

    lowered = lower_to_tirx(kernel)
    assert RunManagedSmemTile.smem_manager.tiles["managed"]["shared"]
    assert "T.ptx.mbarrier.try_wait" in lowered.script()
    assert "T.ptx.mbarrier.arrive" in lowered.script()


def test_class_and_tile_transient_smem_layouts_do_not_alias():
    kernel = KernelSpec("class_and_tile_smem_layout")
    kernel.tile("managed", ClassAndDeviceManagedSmemTile(), (1, 1, 1))

    lower_to_tirx(kernel)
    manager = ClassAndDeviceManagedSmemTile.smem_manager
    class_info = manager.tiles[str(ClassAndDeviceManagedSmemTile)]["shared"][0]
    tile_info = manager.tiles["managed"]["shared"][0]
    _, class_begin, class_size, _ = class_info
    _, tile_begin, tile_size, _ = tile_info
    assert (class_begin, class_begin + class_size) == (0, 64)
    assert (tile_begin, tile_begin + tile_size) == (64, 128)


def test_class_and_tile_transient_smem_reject_mixed_policies():
    kernel = KernelSpec("class_and_tile_smem_policy")
    kernel.tile("managed", ClassSharedDeviceExclusiveSmemTile(), (1, 1, 1))

    with pytest.raises(
        tvm.error.DiagnosticError,
        match="cannot mix shared and exclusive smem allocation policies",
    ):
        lower_to_tirx(kernel)


def test_class_smem_records_are_not_overwritten_by_later_classes():
    kernel = KernelSpec("class_smem_records")
    kernel.tile("oversized", OversizedClassSmemTile(), (1, 1, 1))
    kernel.tile("small", SmallClassSmemTile(), (1, 1, 1))

    with pytest.raises(tvm.error.DiagnosticError, match="configured capacity"):
        lower_to_tirx(
            kernel,
            options=LoweringOptions(smem_max_bytes=4096, smem_chunk_size=1024),
        )


def test_finalize_shared_resources_uses_its_class_allocation_key():
    kernel = KernelSpec("finalize_class_smem_key")
    kernel.tile("tile", FinalizeOwnerTile(), (1, 1, 1))

    lower_to_tirx(kernel)
    assert FinalizeOwnerTile.finalized_key == str(FinalizeOwnerTile)


def test_host_init_is_emitted_before_device_entry():
    kernel = KernelSpec("host_init_order")
    kernel.tile("first", HostInitTile("test.host_init.first"), (1, 1, 1))
    kernel.tile("second", HostInitTile("test.host_init.second"), (1, 1, 1))

    script = lower_to_tirx(kernel).script()
    device_entry = script.index('"tirx.device_entry"')
    assert script.index('T.call_packed("test.host_init.first")') < device_entry
    assert script.index('T.call_packed("test.host_init.second")') < device_entry


def test_impl_emitted_barriers_appear_in_the_kernel():
    kernel = KernelSpec("explicit_barriers")
    kernel.tile("tile", BarrierTile(), (1, 1, 1))

    script = lower_to_tirx(kernel).script()
    assert "T.cuda.cta_sync()" in script
    assert "T.cuda.warp_sync()" in script


def test_backend_prepare_binds_bounded_workspace_and_static_phases():
    kernel = KernelSpec("prepared")
    rows = kernel.var("rows", range=(1, 8))
    source = kernel.tensor("source", (rows, 4), "float16")
    destination = kernel.tensor("destination", (rows, 4), "float16")
    tile_rows = rows.ceildiv(2)
    ready = kernel.event("ready", (tile_rows,), 1)
    kernel.tile(
        "producer",
        RecordingTile(),
        (tile_rows, 1, 1),
        reads=[source],
        writes=[destination],
    ).notify(ready, lambda m, n, k: (m,))
    kernel.tile(
        "consumer",
        RecordingTile(),
        (tile_rows, 1, 1),
        reads=[destination],
    ).wait(ready, lambda m, n, k: (m,))

    validate_kernel(kernel)
    prepared = prepare_tirx_lowering_plan(
        kernel,
        LoweringOptions(attrs={"sm_count": 2}),
    )

    assert prepared.kernel is kernel
    assert [binding.param_name for binding in prepared.var_bindings] == ["rows"]
    assert [binding.param_name for binding in prepared.tensor_bindings] == [
        "source",
        "destination",
    ]
    assert prepared.event_layouts[0].reserved_size == 4
    assert prepared.event_workspace_size == 5
    assert [tile.job_id for tile in prepared.tile_plans] == [0, 1]
    assert [phase.job_id for phase in prepared.static_schedule.phases] == [29, 0, 30, 1, 31]


def test_backend_prepare_rejects_unbounded_event_workspace():
    kernel = KernelSpec("unbounded_workspace")
    rows = kernel.var("rows")
    ready = kernel.event("ready", (rows,), 1)
    kernel.tile("producer", RecordingTile(), (rows, 1, 1)).notify(ready, lambda m, n, k: (m,))
    kernel.tile("consumer", RecordingTile(), (rows, 1, 1)).wait(ready, lambda m, n, k: (m,))

    with pytest.raises(ValueError, match="has no range"):
        prepare_tirx_lowering_plan(kernel, LoweringOptions())


def test_backend_prepare_rejects_non_int32_event_storage():
    kernel = KernelSpec("event_dtype")
    ready = kernel.event("ready", (1,), 1, dtype="int64")
    kernel.tile("producer", RecordingTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("consumer", RecordingTile(), (1, 1, 1)).wait(ready, (0,))

    with pytest.raises(ValueError, match="requires int32"):
        prepare_tirx_lowering_plan(kernel, LoweringOptions())


@pytest.mark.parametrize(
    "init_count",
    [32768, lambda coord: 32768],
    ids=["constant", "callable"],
)
def test_backend_prepare_rejects_event_counts_that_overflow_int32(init_count):
    kernel = KernelSpec("event_counter_overflow")
    ready = kernel.event("ready", (1,), init_count)
    kernel.tile("producer", RecordingTile(), (8192, 4, 1)).notify(ready, (0,))
    kernel.tile("consumer", RecordingTile(), (1, 1, 1)).wait(ready, (0,))

    with pytest.raises(ValueError, match="int32 semaphore encoding limit 32767"):
        prepare_tirx_lowering_plan(
            kernel,
            LoweringOptions(attrs={"sm_count": 148}),
        )


def test_static_schedule_auto_sizes_max_tasks():
    kernel = KernelSpec("max_tasks_default")
    kernel.tile("tile", RecordingTile(), (4, 1, 1))
    prepared = prepare_tirx_lowering_plan(kernel, LoweringOptions(attrs={"sm_count": 2}))
    assert prepared.static_schedule.max_tasks == 128

    large = KernelSpec("max_tasks_grown")
    large.tile("tile", RecordingTile(), (1024, 1, 1))
    prepared = prepare_tirx_lowering_plan(large, LoweringOptions(attrs={"sm_count": 1}))
    assert prepared.static_schedule.max_tasks == 1025

    explicit = KernelSpec("max_tasks_explicit")
    explicit.tile("tile", RecordingTile(), (4, 1, 1))
    prepared = prepare_tirx_lowering_plan(
        explicit,
        LoweringOptions(attrs={"sm_count": 2, "max_tasks": 64}),
    )
    assert prepared.static_schedule.max_tasks == 64


def test_lower_to_tirx_module_contains_kernel_and_queue_init():
    kernel = KernelSpec("module_two_stage")
    rows = kernel.var("rows", "int32", range=(1, 16))
    tensor_a = kernel.tensor("A", (rows, 4), "float16")
    tensor_b = kernel.tensor("B", (rows,), "float32")
    ready = kernel.event("ready", (rows,), 1)
    kernel.tile("producer", RecordingTile(), (rows, 1, 1), writes=[tensor_b]).notify(
        ready, lambda m, n, k: (m,)
    )
    kernel.tile(
        "consumer", RecordingTile(), (rows, 1, 1), reads=[tensor_b], writes=[tensor_a]
    ).wait(ready, lambda m, n, k: (m,))

    mod = lower_to_tirx_module(kernel, LoweringOptions(attrs={"sm_count": 2}))

    assert {gv.name_hint for gv in mod.functions} == {
        "module_two_stage",
        "module_two_stage_init_queue",
    }


def test_kernel_lower_uses_the_default_static_backend():
    kernel = KernelSpec("empty")
    lowered = kernel.lower()
    assert isinstance(lowered, tvm.tirx.PrimFunc)
    assert lowered.attrs["global_symbol"] == "empty"


def test_kernel_lower_binds_tensor_specs_and_tuple_events():
    kernel = KernelSpec("copy_chain")
    source = kernel.tensor("source", (4,), "float32")
    intermediate = kernel.tensor("intermediate", (4,), "float32")
    destination = kernel.tensor("destination", (4,), "float32")
    ready = kernel.event("ready", (1,), 4)
    kernel.tile(
        "producer",
        CopyTile(source, intermediate),
        (4, 1, 1),
        reads=[source],
        writes=[intermediate],
    ).notify(ready, (0,))
    kernel.tile(
        "consumer",
        CopyTile(intermediate, destination),
        (4, 1, 1),
        reads=[intermediate],
        writes=[destination],
    ).wait(ready, (0,))

    lowered = kernel.lower()
    assert isinstance(lowered, tvm.tirx.PrimFunc)
    assert [param.name for param in lowered.params[:3]] == [
        "source_handle",
        "intermediate_handle",
        "destination_handle",
    ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
