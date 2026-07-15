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
"""Tests for logical megakernel specs and action-program lowering."""

import inspect
from dataclasses import replace

import pytest

import tvm
from tvm.megakernel.dsl import EventSpec, KernelSpec, TensorSpec, TileImpl, TileSpec, VarSpec
from tvm.megakernel.transform import (
    DeviceRegionPlan,
    EdgeBindingPlan,
    ExecutionPlan,
    FetchGuardAction,
    HookAction,
    HostCallAction,
    HostEdgeAction,
    HostRegionPlan,
    MegakernelBackend,
    MidBodyPortAction,
    RegionDependencyPlan,
    SchedulerFetchProgram,
    TileActionProgram,
    TileEmitter,
    logical_edges,
    make_static_execution_plan,
)
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
        lambda coord: coord[0] + 1,
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


def test_default_policy_is_an_ordered_action_program():
    kernel, producer, consumer, _ = _two_stage_kernel()
    plan = make_static_execution_plan(kernel)
    assert plan.validate() is plan
    assert [program.tile.name for program in plan.device_regions[0].tile_programs] == [
        "producer",
        "consumer",
    ]
    producer_actions = plan.device_regions[0].tile_programs[0].actions
    assert [type(action).__name__ for action in producer_actions] == [
        "SmemEnterAction",
        "HookAction",
        "HookAction",
        "RunAction",
        "NotifyAction",
        "SmemExitAction",
    ]
    consumer_actions = plan.device_regions[0].tile_programs[1].actions
    assert [type(action).__name__ for action in consumer_actions] == [
        "SmemEnterAction",
        "HookAction",
        "WaitAction",
        "HookAction",
        "RunAction",
        "SmemExitAction",
    ]
    assert len(plan.edge_bindings) == 1
    assert plan.edge_bindings[0].location == "tile_action"
    assert plan.edge_bindings[0].edge.producer == producer.name
    assert plan.edge_bindings[0].edge.consumer == consumer.name


def test_edge_coverage_rejects_missing_duplicate_and_wrong_location():
    kernel, _, _, _ = _two_stage_kernel()
    plan = make_static_execution_plan(kernel)
    with pytest.raises(ValueError, match="unbound"):
        replace(plan, edge_bindings=()).validate()
    with pytest.raises(ValueError, match="duplicate"):
        replace(plan, edge_bindings=plan.edge_bindings * 2).validate()
    wrong = replace(plan.edge_bindings[0], location="fetch_guard")
    with pytest.raises(ValueError, match="placed at tile_action"):
        replace(plan, edge_bindings=(wrong,)).validate()


def test_fetch_mid_body_and_host_edge_locations_are_explicit():
    kernel, producer, _, _ = _two_stage_kernel()
    edge = logical_edges(kernel)[0]
    fetch_region = DeviceRegionPlan(
        "fetch_device",
        fetch_program=SchedulerFetchProgram((FetchGuardAction((edge,)),)),
        tile_programs=(TileActionProgram(producer, ()),),
    )
    ExecutionPlan(
        kernel,
        device_regions=(fetch_region,),
        edge_bindings=(EdgeBindingPlan(edge, "fetch_guard", fetch_region.name),),
    ).validate()

    port_region = DeviceRegionPlan(
        "port_device",
        tile_programs=(TileActionProgram(producer, (MidBodyPortAction((edge,), "after_store"),)),),
    )
    ExecutionPlan(
        kernel,
        device_regions=(port_region,),
        edge_bindings=(EdgeBindingPlan(edge, "mid_body_port", port_region.name, "after_store"),),
    ).validate()

    host_region = HostRegionPlan(
        "runtime", (HostEdgeAction((edge,), "completion"), HostCallAction("collective"))
    )
    ExecutionPlan(
        kernel,
        host_regions=(host_region,),
        edge_bindings=(EdgeBindingPlan(edge, "host_runtime", host_region.name),),
    ).validate()


class _RecordingBackend(MegakernelBackend):
    def __init__(self):
        self.trace = []

    def bind_region_dependency(self, plan, dependency):
        self.trace.append(("dependency", dependency.kind))

    def begin_region(self, plan, region):
        self.trace.append(("begin", region.name))

    def emit_action(self, action, context):
        self.trace.append((context.program, type(action).__name__))

    def end_region(self, plan, region):
        self.trace.append(("end", region.name))

    def end_execution(self, plan):
        return tuple(self.trace)


def test_region_dag_distinguishes_launch_order_and_completion():
    kernel = KernelSpec("regions")
    producer = HostRegionPlan("producer", (HostCallAction("launch"),))
    overlap = HostRegionPlan("overlap", (HostCallAction("launch"),))
    consumer = HostRegionPlan("consumer", (HostCallAction("launch"),))
    plan = ExecutionPlan(
        kernel,
        host_regions=(consumer, overlap, producer),
        region_dependencies=(
            RegionDependencyPlan("producer", "overlap", "launch_order"),
            RegionDependencyPlan("overlap", "consumer", "completion"),
        ),
    )
    trace = TileEmitter(_RecordingBackend()).emit(plan)
    assert [item for item in trace if item[0] == "begin"] == [
        ("begin", "producer"),
        ("begin", "overlap"),
        ("begin", "consumer"),
    ]
    assert trace[:2] == (("dependency", "launch_order"), ("dependency", "completion"))


class _RegionProtocolBackend(MegakernelBackend):
    def __init__(self):
        self.trace = []

    def bind_launch_order(self, plan, dependency):
        self.trace.append(("launch_order", dependency.source, dependency.target))

    def bind_completion(self, plan, dependency):
        self.trace.append(("completion", dependency.source, dependency.target))

    def begin_device_region(self, plan, region):
        self.trace.append(("begin_device", region.name))

    def begin_host_region(self, plan, region):
        self.trace.append(("begin_host", region.name))

    def emit_device_action(self, action, context):
        self.trace.append(("device", context.region.name, type(action).__name__))

    def emit_host_action(self, action, context):
        self.trace.append(("host", context.region.name, type(action).__name__))

    def end_device_region(self, plan, region):
        self.trace.append(("end_device", region.name))

    def end_host_region(self, plan, region):
        self.trace.append(("end_host", region.name))

    def end_execution(self, plan):
        return tuple(self.trace)


def test_region_backend_protocol_dispatches_region_and_dependency_kinds():
    kernel = KernelSpec("region_protocol")
    partial = DeviceRegionPlan("partial", prologue=(HookAction("launch"),))
    collective = HostRegionPlan("collective", (HostCallAction("reduce_scatter"),))
    reduce = DeviceRegionPlan("reduce", epilogue=(HookAction("finish"),))
    plan = ExecutionPlan(
        kernel,
        device_regions=(partial, reduce),
        host_regions=(collective,),
        region_dependencies=(
            RegionDependencyPlan("partial", "collective", "launch_order"),
            RegionDependencyPlan("collective", "reduce", "completion"),
        ),
    )

    trace = TileEmitter(_RegionProtocolBackend()).emit(plan)
    assert trace == (
        ("launch_order", "partial", "collective"),
        ("completion", "collective", "reduce"),
        ("begin_device", "partial"),
        ("device", "partial", "HookAction"),
        ("end_device", "partial"),
        ("begin_host", "collective"),
        ("host", "collective", "HostCallAction"),
        ("end_host", "collective"),
        ("begin_device", "reduce"),
        ("device", "reduce", "HookAction"),
        ("end_device", "reduce"),
    )


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


def test_kernel_lower_uses_default_static_action_backend():
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
