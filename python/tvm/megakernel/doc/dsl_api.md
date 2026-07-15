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

# Megakernel DSL User API

This document lists the user-facing API in `tvm.megakernel.dsl`.

## DSL Layers

The DSL has two layers:

- Spec layer: `KernelSpec`, `VarSpec`, `TensorSpec`, `EventSpec`, and
  `TileSpec`.  This layer describes tile stages, tensors, logical events, and
  tuple-form wait/notify dependencies.
- Impl layer: `TileImpl`.  This layer connects a logical tile to the concrete
  implementation of that tile.

The spec layer corresponds to Step 3 in [workflow.md](workflow.md).  The impl
layer corresponds to Step 4.

## `KernelSpec`

```python
kernel = KernelSpec(name: str, attrs: dict[str, Any] | None = None)
```

Creates one megakernel spec.  Tensors, events, and tiles are registered on this
object.

Parameters:

- `name`: unique name for the megakernel spec.
- `attrs`: optional metadata reserved for later passes.


Example:

```python
kernel = KernelSpec("two_stage_reduce", attrs={"target": "sm90"})
```

## `KernelSpec.var`

```python
var = kernel.var(name: str, dtype: str = "int32")
```

Registers a symbolic variable owned by this kernel.  Shapes and three-axis
tile counts may reference the returned `VarSpec`.

Example:

```python
rows = kernel.var("rows", "int32")
```

## `KernelSpec.tensor`

```python
tensor = kernel.tensor(name: str, shape: ShapeType, dtype: str)
```

Registers a logical tensor.

Parameters:

- `name`: tensor name, unique inside the kernel.
- `shape`: tensor shape.  Each dimension can be an `int` or `VarSpec`.
- `dtype`: tensor element type.

Returns: `TensorSpec`.

Example:

```python
rows = kernel.var("rows")
A = kernel.tensor("A", shape=(rows, 1024), dtype="float32")
```

## `KernelSpec.event`

```python
event = kernel.event(
    name: str,
    shape: ShapeType,
    init_count: int | Callable[[tuple[int, ...]], int],
    dtype: str = "int32",
    attrs: dict[str, Any] | None = None,
)
```

Registers a logical event tensor.

Parameters:

- `name`: event name, unique inside the kernel.
- `shape`: event tensor shape.
- `init_count`: logical count for each event coordinate.  This can be a single
  integer for a uniform count, or a callable that returns the count for a given
  event coordinate.
- `dtype`: event storage dtype.  Defaults to `"int32"`.
- `attrs`: optional metadata reserved for later passes.

Returns: `EventSpec`.

Examples:

```python
evt1 = kernel.event(
    "evt1",
    shape=(100,),
    init_count=88,
)

evt2 = kernel.event(
    "evt2",
    shape=(100, 200),
    init_count=lambda coord: coord[0] + coord[1],
)
```

## `KernelSpec.tile`

```python
tile = kernel.tile(
    name: str,
    impl: TileImpl,
    tile_num: TileNumType,
    reads: list[TensorSpec] | None = None,
    writes: list[TensorSpec] | None = None,
    attrs: dict[str, Any] | None = None,
)
```

Registers one logical tile stage.

Parameters:

- `name`: tile stage name, unique inside the kernel.
- `impl`: local tile implementation object.
- `tile_num`: tile count on `(m, n, k)` axes.  Use `1` for unused axes.
- `reads`: logical tensors read by this tile.
- `writes`: logical tensors written by this tile.
- `attrs`: optional metadata reserved for later passes.

Returns: `TileSpec`.

Example:

```python
bs = kernel.var("bs")
tile_a = kernel.tile(
    "tile_a",
    tile_a_impl,
    tile_num=(bs, 16, 1),
    reads=[A],
    writes=[B],
)
```

## `TileSpec.wait`

```python
tile.wait(event: EventSpec, coord_map: CoordMapType)
```

Declares that this tile waits on `event` at the coordinate produced by
`coord_map`.

Parameters:

- `event`: event to wait on.
- `coord_map`: callable mapping tile index `(m, n, k)` to an event
  coordinate.  If it is a tuple/list instead of a callable, it is used directly
  as the event coordinate.

Returns: `TileSpec`.

Example:

```python
tile_b.wait(
    event=evt1,
    coord_map=lambda m, n, k: (m,),
)
```

## `TileSpec.notify`

```python
tile.notify(event: EventSpec, coord_map: CoordMapType)
```

Declares that this tile notifies `event` at the coordinate produced by
`coord_map`.

Parameters:

- `event`: event to notify.
- `coord_map`: callable mapping tile index `(m, n, k)` to an event
  coordinate.  If it is a tuple/list instead of a callable, it is used directly
  as the event coordinate.

Returns: `TileSpec`.

Example:

```python
tile_a.notify(
    event=evt1,
    coord_map=lambda m, n, k: (m,),
)
```

## `TileImpl`

Users subclass `TileImpl` to define the local implementation for one tile kind.
Only `run()` is required.

```python
class MyTile(TileImpl):
    def run(self, m_idx, n_idx, k_idx):
        ...
```

### `TileImpl.init_shared_resources`

```python
@classmethod
def init_shared_resources(cls, smem_manager: SmemManager): ...
```

Optional.  Initializes resources shared by all instances of this tile class.
For example, this hook can allocate shared memory, mbarriers, or tensor memory
used by all tile instances of the same class.

Example:

```python
@classmethod
def init_shared_resources(cls, smem_manager):
    cls.workspace = smem_manager.alloc((128,), "uint8", policy="persistent")
```

### `TileImpl.finalize_shared_resources`

```python
@classmethod
def finalize_shared_resources(cls, smem_manager: SmemManager): ...
```

Optional.  Finalizes class-level resource declarations after every tile class
has run `init_shared_resources()` and before the shared-memory manager commits
its allocation plan.


### `TileImpl.device_init`

```python
def device_init(
    self,
    smem_manager: SmemManager,
    m_idx,
    n_idx,
    k_idx,
): ...
```

Optional.  Initializes device-side state owned by one tile instance.  For
example, this hook can allocate managed shared-memory buffers used by that tile
instance.

### `TileImpl.host_init`

```python
def host_init(self): ...
```

Optional.  Initializes host-side state for one tile instance.  For example,
this hook can set the cuTensorMap used by that tile instance.

Example:

```python
def host_init(self):
    T.call_packed("runtime.cuTensorMapEncodeTiled", ...)
```

### `TileImpl.prefetch`

```python
def prefetch(self, m_idx, n_idx, k_idx): ...
```

Optional.  Prefetches data for one tile instance before `run()`.  This hook
may run before the tile dependency is satisfied, after the tile has been
dispatched to an SM.  For example, it can prefetch weights that do not depend
on activations while previous tasks are still incomplete.

### `TileImpl.run`

```python
def run(self, m_idx, n_idx, k_idx): ...
```

Required.  Defines the computation for one logical tile instance at index
`(m_idx, n_idx, k_idx)`.

## DSL Example

```python
stage1 = kernel.tile(
    "stage1",
    Stage1Tile(),
    (NUM_BLOCK_M, NUM_BLOCK_N, 1),
    reads=[A],
    writes=[B],
).notify(row_ready, lambda m, n, k: (m,))

stage2 = kernel.tile(
    "stage2",
    Stage2Tile(),
    (NUM_BLOCK_M, 1, 1),
    reads=[B],
    writes=[C],
).wait(row_ready, lambda m, n, k: (m,))
```

## Physical Execution Plan

Lowering policies produce one `ExecutionPlan`.  Its device regions contain
ordered `TileProgram` objects, and each program contains the `ProgramStep`
sequence that a backend lowers.  For example, the default static policy emits
`HookStep`, `WaitStep`, `RunStep`, and `NotifyStep` values in source order:

```python
from tvm.megakernel.transform import make_static_execution_plan

plan = make_static_execution_plan(kernel)
for program in plan.device_regions[0].tile_programs:
    print(program.tile.name, program.smem_scope, program.steps)
```

`TileProgram.smem_scope` is one of `"none"`, `"program"`, or `"run_to_end"`.
Logical edge locations are not stored separately.  `plan.edge_placements()`
validates the programs and returns the read-only `EdgePlacement` values derived
from the steps carrying each edge.

A custom backend implements a single entry point and owns traversal of regions
and programs:

```python
from tvm.megakernel.transform import ExecutionPlanBackend, lower_execution_plan


class MyBackend(ExecutionPlanBackend):
    def lower(self, plan):
        for region in plan.regions_in_dependency_order():
            ...


result = lower_execution_plan(plan, backend=MyBackend())
```

## `KernelSpec.lower`

```python
prim_func = kernel.lower(options=None)
```

Validates and lowers the graph with the default single-device static policy,
or with the execution plan and backend supplied through `LoweringOptions`.
