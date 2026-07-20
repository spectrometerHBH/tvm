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
"""Tests for logical megakernel specs and physical-program lowering."""

import inspect
from dataclasses import replace

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
    DeviceRegionPlan,
    EdgePlacement,
    ExecutionPlan,
    ExecutionPlanBackend,
    FetchGuardStep,
    HookStep,
    HostCallStep,
    HostRegionPlan,
    HostSyncStep,
    LoweringOptions,
    MidBodyPortStep,
    NotifyStep,
    RegionDependencyPlan,
    SemanticPlan,
    TileProgram,
    WaitStep,
    build_semantic_plan,
    logical_edges,
    lower_execution_plan,
    make_static_execution_plan,
    validate_semantic_plan,
)
from tvm.megakernel.transform.prepare import prepare_tirx_lowering_plan
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


def test_semantic_plan_is_distinct_from_the_physical_execution_plan():
    kernel, _, _, _ = _two_stage_kernel()
    semantic = build_semantic_plan(kernel)

    assert isinstance(semantic, SemanticPlan)
    assert semantic.kernel is kernel
    assert [edge.producer.name for edge in semantic.logical_edges] == ["producer"]
    assert [edge.consumer.name for edge in semantic.logical_edges] == ["consumer"]

    execution = make_static_execution_plan(kernel)
    assert execution.semantic_plan == semantic
    assert execution.resolved_semantic_plan().kernel is kernel

    stale = replace(semantic, logical_edges=())
    with pytest.raises(ValueError, match="logical edges do not match"):
        validate_semantic_plan(stale)


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
    with pytest.raises(ValueError, match="matching event coordinate"):
        kernel.validate()


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


def test_transform_exports_only_physical_step_contract():
    from tvm.megakernel import transform

    for old_name in (
        "Action",
        "EdgeBindingPlan",
        "EmissionContext",
        "MegakernelBackend",
        "SchedulerFetchProgram",
        "TileActionProgram",
        "TileEmitter",
    ):
        assert not hasattr(transform, old_name)
    for old_name in (
        "HookAction",
        "RunAction",
        "SmemEnterAction",
        "SmemExitAction",
    ):
        assert not hasattr(transform, old_name)


def test_demo_uses_current_parser_style_api_and_lowers():
    from tvm.megakernel.demo import dsl as demo

    assert demo.kernel.validate() is demo.kernel
    assert set(demo.kernel.tensors) == {"A", "B", "C"}
    assert [tensor.name for tensor in demo.stage1.reads] == ["A"]
    assert [tensor.name for tensor in demo.stage1.writes] == ["B"]
    assert [tensor.name for tensor in demo.stage2.reads] == ["B"]
    assert [tensor.name for tensor in demo.stage2.writes] == ["C"]
    assert not hasattr(demo.kernel, "input")
    assert not hasattr(demo.stage1, "read")
    assert [program.tile.name for program in demo.execution.device_regions[0].tile_programs] == [
        demo.stage1.name,
        demo.stage2.name,
    ]
    assert demo.execution.edge_placements()[0].location == "tile"

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


def test_validate_rejects_stateful_callables_and_cycles():
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

    kernel = KernelSpec("cycle")
    first_ready = kernel.event("first_ready", (1,), 1)
    second_ready = kernel.event("second_ready", (1,), 1)
    kernel.tile("first", RecordingTile(), (1, 1, 1)).wait(second_ready, (0,)).notify(
        first_ready, (0,)
    )
    kernel.tile("second", RecordingTile(), (1, 1, 1)).wait(first_ready, (0,)).notify(
        second_ready, (0,)
    )
    with pytest.raises(ValueError, match="acyclic"):
        kernel.validate()


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


def test_default_policy_is_an_ordered_physical_program():
    kernel, producer, consumer, _ = _two_stage_kernel()
    plan = make_static_execution_plan(kernel)
    assert plan.validate() is plan
    assert [program.tile.name for program in plan.device_regions[0].tile_programs] == [
        "producer",
        "consumer",
    ]
    producer_program = plan.device_regions[0].tile_programs[0]
    assert producer_program.smem_scope == "program"
    assert [type(step).__name__ for step in producer_program.steps] == [
        "HookStep",
        "HookStep",
        "RunStep",
        "NotifyStep",
    ]
    consumer_program = plan.device_regions[0].tile_programs[1]
    assert consumer_program.smem_scope == "program"
    assert [type(step).__name__ for step in consumer_program.steps] == [
        "HookStep",
        "WaitStep",
        "HookStep",
        "RunStep",
    ]
    placements = plan.edge_placements()
    assert len(placements) == 1
    assert isinstance(placements[0], EdgePlacement)
    assert placements[0].location == "tile"
    assert placements[0].region == "device"
    assert placements[0].port is None
    assert placements[0].edge.producer == producer.name
    assert placements[0].edge.consumer == consumer.name


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

    execution = make_static_execution_plan(kernel)
    prepared = prepare_tirx_lowering_plan(
        execution,
        LoweringOptions(attrs={"sm_count": 2}),
    )

    assert prepared.semantic.kernel is kernel
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
    execution = make_static_execution_plan(kernel)

    with pytest.raises(ValueError, match="has no range"):
        prepare_tirx_lowering_plan(execution, LoweringOptions())


def test_backend_prepare_rejects_non_int32_event_storage():
    kernel = KernelSpec("event_dtype")
    ready = kernel.event("ready", (1,), 1, dtype="int64")
    kernel.tile("producer", RecordingTile(), (1, 1, 1)).notify(ready, (0,))
    kernel.tile("consumer", RecordingTile(), (1, 1, 1)).wait(ready, (0,))

    with pytest.raises(ValueError, match="requires int32"):
        prepare_tirx_lowering_plan(make_static_execution_plan(kernel), LoweringOptions())


def test_edge_placement_rejects_missing_cross_location_and_cross_region():
    kernel, producer, consumer, _ = _two_stage_kernel()
    plan = make_static_execution_plan(kernel)
    region = plan.device_regions[0]
    stripped_programs = tuple(
        replace(
            program,
            steps=tuple(replace(step, edges=()) if step.edges else step for step in program.steps),
        )
        for program in region.tile_programs
    )
    with pytest.raises(ValueError, match="missing physical steps"):
        replace(plan, device_regions=(replace(region, tile_programs=stripped_programs),)).validate()

    edge = logical_edges(kernel)[0]
    fetch_and_tile = DeviceRegionPlan(
        "device",
        fetch_steps=(FetchGuardStep(edges=(edge,)),),
        tile_programs=(TileProgram(producer, (NotifyStep(edge.event, (0,), edges=(edge,)),)),),
    )
    with pytest.raises(ValueError, match="spans physical placements"):
        ExecutionPlan(kernel, device_regions=(fetch_and_tile,)).validate()

    producer_region = DeviceRegionPlan(
        "producer_region",
        tile_programs=(TileProgram(producer, (NotifyStep(edge.event, (0,), edges=(edge,)),)),),
    )
    consumer_region = DeviceRegionPlan(
        "consumer_region",
        tile_programs=(TileProgram(consumer, (WaitStep(edge.event, (0,), edges=(edge,)),)),),
    )
    with pytest.raises(ValueError, match="spans physical placements"):
        ExecutionPlan(kernel, device_regions=(producer_region, consumer_region)).validate()


def test_edge_placement_rejects_unknown_wrong_endpoint_and_wrong_port():
    kernel, producer, consumer, _ = _two_stage_kernel()
    edge = logical_edges(kernel)[0]
    unknown = type(edge)(edge.event, "unknown", edge.consumer)
    with pytest.raises(ValueError, match="unknown logical edge"):
        ExecutionPlan(
            kernel,
            host_regions=(HostRegionPlan("host", (HostSyncStep("completion", edges=(unknown,)),)),),
        ).validate()

    wrong_producer = DeviceRegionPlan(
        "device",
        tile_programs=(TileProgram(consumer, (NotifyStep(edge.event, (0,), edges=(edge,)),)),),
    )
    with pytest.raises(ValueError, match="logical producer tile"):
        ExecutionPlan(kernel, device_regions=(wrong_producer,)).validate()

    wrong_consumer = DeviceRegionPlan(
        "device",
        tile_programs=(TileProgram(producer, (WaitStep(edge.event, (0,), edges=(edge,)),)),),
    )
    with pytest.raises(ValueError, match="logical consumer tile"):
        ExecutionPlan(kernel, device_regions=(wrong_consumer,)).validate()

    wrong_port = DeviceRegionPlan(
        "device",
        tile_programs=(
            TileProgram(
                producer,
                (
                    MidBodyPortStep("after_store", edges=(edge,)),
                    MidBodyPortStep("after_advance", edges=(edge,)),
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="spans physical placements"):
        ExecutionPlan(kernel, device_regions=(wrong_port,)).validate()


def test_fetch_mid_body_and_host_edge_locations_are_derived():
    kernel, producer, _, _ = _two_stage_kernel()
    edge = logical_edges(kernel)[0]
    fetch_region = DeviceRegionPlan(
        "fetch_device",
        fetch_steps=(FetchGuardStep(edges=(edge,)),),
        tile_programs=(TileProgram(producer, ()),),
    )
    fetch = ExecutionPlan(kernel, device_regions=(fetch_region,)).edge_placements()[0]
    assert (fetch.location, fetch.region, fetch.port) == ("fetch", "fetch_device", None)

    port_region = DeviceRegionPlan(
        "port_device",
        tile_programs=(TileProgram(producer, (MidBodyPortStep("after_store", edges=(edge,)),)),),
    )
    port = ExecutionPlan(kernel, device_regions=(port_region,)).edge_placements()[0]
    assert (port.location, port.region, port.port) == ("tile", "port_device", "after_store")

    host_region = HostRegionPlan(
        "runtime", (HostSyncStep("completion", edges=(edge,)), HostCallStep("collective"))
    )
    host = ExecutionPlan(kernel, host_regions=(host_region,)).edge_placements()[0]
    assert (host.location, host.region, host.port) == ("host", "runtime", None)


def test_region_dag_distinguishes_launch_order_and_completion():
    kernel = KernelSpec("regions")
    producer = HostRegionPlan("producer", (HostCallStep("launch"),))
    overlap = HostRegionPlan("overlap", (HostCallStep("launch"),))
    consumer = HostRegionPlan("consumer", (HostCallStep("launch"),))
    plan = ExecutionPlan(
        kernel,
        host_regions=(consumer, overlap, producer),
        region_dependencies=(
            RegionDependencyPlan("producer", "overlap", "launch_order"),
            RegionDependencyPlan("overlap", "consumer", "completion"),
        ),
    )
    plan.validate()
    assert [region.name for region in plan.regions_in_dependency_order()] == [
        "producer",
        "overlap",
        "consumer",
    ]
    assert [dependency.kind for dependency in plan.region_dependencies] == [
        "launch_order",
        "completion",
    ]


class _RecordingBackend(ExecutionPlanBackend):
    def __init__(self):
        self.trace = []
        self.calls = 0

    def lower(self, plan):
        self.calls += 1
        for dependency in plan.region_dependencies:
            self.trace.append((dependency.kind, dependency.source, dependency.target))
        for region in plan.regions_in_dependency_order():
            if isinstance(region, DeviceRegionPlan):
                for step in region.prologue_steps:
                    self.trace.append(("device", region.name, type(step).__name__))
                for step in region.fetch_steps:
                    self.trace.append(("device", region.name, type(step).__name__))
                for program in region.tile_programs:
                    for step in program.steps:
                        self.trace.append(("device", region.name, type(step).__name__))
                for step in region.epilogue_steps:
                    self.trace.append(("device", region.name, type(step).__name__))
            else:
                for step in region.steps:
                    self.trace.append(("host", region.name, type(step).__name__))
        return tuple(self.trace)


def test_execution_plan_backend_owns_traversal_and_backend_selection():
    kernel = KernelSpec("region_protocol")
    partial = DeviceRegionPlan("partial", prologue_steps=(HookStep("launch"),))
    collective = HostRegionPlan("collective", (HostCallStep("reduce_scatter"),))
    reduce = DeviceRegionPlan("reduce", epilogue_steps=(HookStep("finish"),))
    plan = ExecutionPlan(
        kernel,
        device_regions=(partial, reduce),
        host_regions=(collective,),
        region_dependencies=(
            RegionDependencyPlan("partial", "collective", "launch_order"),
            RegionDependencyPlan("collective", "reduce", "completion"),
        ),
    )

    selected = _RecordingBackend()
    options_backend = _RecordingBackend()
    trace = lower_execution_plan(
        plan, backend=selected, options=LoweringOptions(backend=options_backend)
    )
    assert selected.calls == 1
    assert options_backend.calls == 0
    assert trace == (
        ("launch_order", "partial", "collective"),
        ("completion", "collective", "reduce"),
        ("device", "partial", "HookStep"),
        ("host", "collective", "HostCallStep"),
        ("device", "reduce", "HookStep"),
    )

    from_options = _RecordingBackend()
    assert lower_execution_plan(plan, options=LoweringOptions(backend=from_options))
    assert from_options.calls == 1


def test_region_dag_rejects_cycle():
    kernel = KernelSpec("cycle")
    plan = ExecutionPlan(
        kernel,
        host_regions=(HostRegionPlan("a", ()), HostRegionPlan("b", ())),
        region_dependencies=(
            RegionDependencyPlan("a", "b", "launch_order"),
            RegionDependencyPlan("b", "a", "completion"),
        ),
    )
    with pytest.raises(ValueError, match="acyclic"):
        plan.validate()


def test_kernel_lower_uses_default_static_execution_plan_backend():
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
    pytest.main([__file__])
