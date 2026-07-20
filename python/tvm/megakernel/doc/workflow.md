<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Megakernel DSL Design

## Goal

The megakernel DSL is a spec-level bridge between a planned megakernel
structure and a TIRX megakernel implementation.

The original input can be Torch-like code,
natural language, an operator graph, pseudocode, or any other description that
contains enough staged computation information for an agent or user to plan the
megakernel structure.

The user-facing DSL is split into two layers:

```text
1. Spec layer: describes tile stages, tile spaces, tensors, events, and
   wait/notify dependencies.
2. Impl layer: connects each logical tile to the concrete implementation of
   that tile.
```

The spec layer corresponds to Step 3 of the workflow.  The impl layer
corresponds to Step 4.

The DSL should not encode low-level event implementation details such as
atomics, barrier layout, spin waits, or runtime scheduling code.

## DSL Responsibility

The DSL starts after the tile/event plan is known.  Its responsibility is to
record the planned structure and attach tile implementations in a stable Python
object model.

The logical spec and implementation layers are not responsible for:

```text
1. parsing natural language or source programs
2. deciding the tile partitioning
3. deciding which logical events are required
4. implementing wait/notify mechanics
5. selecting or emitting a concrete backend
```

Those tasks belong to input interpretation, planning, validation, and transform
layers around the logical model.  This package's `transform/` module includes a
reference single-device static TIRX backend; custom backends may consume the
same `ExecutionPlan` contract.

## Compiler Plan Boundaries

The compiler does not use one catch-all plan object:

```text
KernelSpec
  -> SemanticPlan
  -> ExecutionPlan
  -> backend-private TIRXLoweringPlan
  -> TIRX
```

`SemanticPlan` owns backend-independent meaning and validation.  It includes
logical tile/event edges and optional tensor access regions.  `ExecutionPlan`
owns explicit physical regions and source-ordered `ProgramStep` sequences.
The default backend's private plan owns sanitized parameter bindings, bounded
event-workspace offsets, packed job IDs, and static queue phases.  Backends may
prepare different private plans without changing the semantic or physical
contracts.

## Agent Workflow

The intended workflow is:

```text
1. User provides a staged computation description.
2. Agent plans tile partitioning and event dependencies.
3. Agent/user writes the upper DSL spec layer from the complete plan.
4. User/agent fills in the lower DSL impl layer for each tile.
5. The compiler runs necessary validation.
6. The compiler lowers the DSL and tile implementations to a TIRX megakernel.
```

## Step 1: Staged Computation Input

Input is any high-level staged computation description, such as natural
language, Torch-like code, pseudocode, an operator graph, or a written staged
dataflow description.  The input only needs to contain enough information for
an agent or user to identify the staged computation and the relevant shapes,
block sizes, or symbolic dimensions.

## Step 2: Agent Plan

The agent preserves the visible staged dataflow from the input and plans the
megakernel structure.

The plan should determine:

```text
1. logical tile stages
2. tile instance space for each stage
3. tensors read and written by each tile stage
4. events required between producer and consumer tiles
5. wait/notify coordinate mappings
```

Use [plan.md](plan.md) as the Stage 1 planning specification.

The output of this step is still a logical plan.  It must not contain CUDA,
TIRX bodies, runtime event implementation, atomics, spin waits, or encoded event
counter formulas.

## Step 3: Plan To Spec Layer

After Step 2, all structural information is known.  The agent or user
converts that plan into the upper DSL spec layer.  Use
[dsl_api.md](dsl_api.md) as the API reference for this step.

The spec layer mainly records:

```text
1. tensors in the megakernel spec
2. logical events and their initial counts
3. tile stages and their tile_num
4. each tile's input tensors
5. each tile's output tensors
6. each tile's wait/notify dependencies
7. optional bounded symbolic expressions
8. optional tile-specific tensor access regions
```

The current core API is:

```text
KernelSpec:
  container for one megakernel spec

VarSpec and ExprSpec:
  bounded symbolic values and logical integer expressions

TensorSpec:
  logical tensor or buffer name, shape, dtype, and optional access-region view

EventSpec:
  logical event name, shape, init_count, dtype, attrs

Dependency tuple:
  (EventSpec, coord_map) wait/notify endpoint

TileSpec:
  tile name, TileImpl, tile_num, reads, writes, waits, notifies, attrs

TileImpl:
  local implementation for one tile kind
```

Example shape:

```python
row_ready = kernel.event(
    "row_ready",
    shape=(NUM_BLOCK_M,),
    init_count=NUM_BLOCK_N,
)

stage1 = kernel.tile(
    "stage1_partial_reduce",
    Stage1ReduceTile(),
    tile_num=(NUM_BLOCK_M, NUM_BLOCK_N, 1),
    reads=[A],
    writes=[B],
).notify(row_ready, coord_map=lambda m, n, k: (m,))

stage2 = kernel.tile(
    "stage2_final_reduce",
    Stage2ReduceTile(),
    tile_num=(NUM_BLOCK_M, 1, 1),
    reads=[B],
    writes=[C],
).wait(row_ready, coord_map=lambda m, n, k: (m,))
```

## Step 4: Fill Impl Layer

Once the spec layer is written, the user or agent fills in the lower DSL impl
layer by providing each `TileImpl`.  A tile implementation describes the local
computation for one tile instance.

`TileImpl` may define:

```text
1. init_shared_resources: resources shared by all instances of this tile class
2. finalize_shared_resources: release shared class-level resources
3. device_init: device-side state owned by one tile instance
4. host_init: host-side state owned by one tile instance
5. prefetch: optional prefetch before the main tile body
6. run: required tile body for one (m, n, k) tile instance
```

`TileImpl` should not encode global dependency policy.  Dependencies are already
represented in the DSL through `wait` and `notify` endpoints.

## Step 5: Validation

Before lowering, the compiler should run necessary validation on the DSL and the
attached implementations.

Validation should check at least:

```text
1. tile_num uses the canonical (m, n, k) convention
2. all tensors referenced by tiles are registered in the KernelSpec
3. all events referenced by waits/notifies are registered in the KernelSpec
4. event coord_map rank is compatible with the event shape when statically known
5. producer/consumer tensor flow is complete enough for lowering
6. required TileImpl hooks and signatures are compatible with the selected lowering
7. event notify multiplicity matches each event coordinate when exactly provable
8. static tensor regions are in bounds and producer/consumer coordinates agree
9. bounded storage and packed task fields fit backend limits
```

Validation should produce diagnostics that are useful for either the user or an
agent to repair the DSL.

## Step 6: Lower To TIRX Megakernel

After semantic validation and physical-policy selection, the backend prepares
its private lowering plan and lowers it together with the tile implementations
to a TIRX megakernel.

This lowering decides concrete implementation details, including:

```text
1. scheduler strategy
2. event storage layout
3. event initialization
4. wait/notify implementation
5. shared/global memory allocation
6. TIRX function structure
7. runtime integration
```

This package includes the reference executable static lowering described above.
It is an implementation integration rather than part of the logical spec or an
agent's planning output; other integrations may lower the same execution plan.
