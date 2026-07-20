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

The user-facing DSL has two layers:

- Spec layer: `KernelSpec`, `VarSpec`, `TensorSpec`, `EventSpec`, and
  `TileSpec`.  This layer describes tile stages, tensors, logical events, and
  tuple-form wait/notify dependencies.
- Impl layer: `TileImpl`.  This layer connects a logical tile to the concrete
  implementation of that tile.

The spec layer corresponds to Step 3 in [workflow.md](workflow.md).  The impl
layer corresponds to Step 4.

Compiler transforms keep the following contracts separate:

```text
KernelSpec -> SemanticPlan -> ExecutionPlan -> backend-private lowering plan
```

`SemanticPlan` is the validated, implementation-independent meaning of the
spec.  `ExecutionPlan` records explicit physical regions and ordered programs.
The default TIRX backend then prepares its own bindings, event layout, job IDs,
and queue phases; that last object is intentionally not a user API.

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
var = kernel.var(
    name: str,
    dtype: str = "int32",
    range: tuple[int, int] | None = None,
)
```

Registers a symbolic variable owned by this kernel.  Shapes, event extents,
and three-axis tile counts may reference the returned `VarSpec`.  `range` is
an optional inclusive positive lower/upper bound.  Static backends use it to
reserve bounded storage while keeping the variable as a runtime parameter.

Example:

```python
rows = kernel.var("rows", "int32", range=(1, 8192))
```

## Integer expressions

`VarSpec` and `ExprSpec` support `+`, `-`, `*`, `//`, `%`, unary `-`, and
`.ceildiv(...)`.  Expressions remain logical until a backend binds their
variables.

```python
rows = kernel.var("rows", range=(1, 8192))
row_tiles = rows.ceildiv(128)
workspace = kernel.tensor("workspace", (row_tiles, 128), "float16")
```

A bounded static allocation requires every variable contributing to its size
to have a range.

## `KernelSpec.tensor`

```python
tensor = kernel.tensor(name: str, shape: ShapeType, dtype: str)
```

Registers a logical tensor.

Parameters:

- `name`: tensor name, unique inside the kernel.
- `shape`: tensor shape.  Each dimension can be an `int`, `VarSpec`, or
  `ExprSpec`.
- `dtype`: tensor element type.

Returns: `TensorSpec`.

Example:

```python
rows = kernel.var("rows")
A = kernel.tensor("A", shape=(rows, 1024), dtype="float32")
```

## `TensorSpec.region`

```python
access = tensor.region(region_map)
access = tensor.region(dynamic=True, reason="data-dependent routing")
```

Returns an access view of the registered tensor for use in one tile's
`reads` or `writes`.  A static region map receives `(m, n, k)` tile indices
and returns `R[...]`, a `RegionSpec`, or a tuple/list of point indices.

```python
producer = kernel.tile(
    "producer",
    producer_impl,
    tile_num=(row_tiles, 1, 1),
    writes=[workspace.region(lambda m, n, k: R[m, 0:128])],
)
```

Static regions let semantic validation check bounds, disjoint writers, and
producer/consumer ordering.  Validation proves these properties when it can
exactly enumerate at most 4,096 bounded symbolic environments, 65,536 tile
points or tile pairs for one check, and 4,194,304 combined environment-point
operations.  It collects only variables referenced by that check.  A larger
bounded space uses deterministic boundary, midpoint, and fixed-seed samples
instead of failing.  Variables used by a static region still require explicit
ranges.  If a region is data-dependent rather than symbolically bounded,
declare it dynamic and give a non-empty reason.  Dynamic access still requires
an event dependency between its producer and consumer.

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

Event cardinality validation exactly enumerates at most 262,144 event and tile
points for each concrete environment.  Larger point spaces, bounded symbolic
spaces above 4,096 environments, combined exact work above 4,194,304
environment-point operations, and unbounded symbolic variables use the same
deterministic best-effort sampling policy.  Sampled validation still rejects
concrete out-of-bounds coordinates and proven count mismatches, but does not
claim an exhaustive proof of an opaque Python coordinate map.

The reference static backend stores each count in a signed `int32` semaphore
using a stride of 65,537, so every per-coordinate `init_count` must be at most
32,767.  A callable count is exhaustively checked across a bounded event domain
of at most 262,144 coordinates; larger callable domains require a custom
backend with its own count representation or proof.

A symbolic event shape denotes its full runtime domain, so every coordinate in
that domain must receive its declared `init_count`.  A constant event shape may
instead reserve capacity for a symbolic runtime-active tile domain; in that
case only coordinates reached by active producers or consumers require
notifications.  This supports persistent pipelines whose runtime work count is
smaller than their statically allocated event workspace.

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

Every tile that allocates transient managed shared memory must bracket its use
with `smem_manager.acquire_all()` and `smem_manager.release_all()`, then call
`smem_manager.advance()` so the next task waits on the opposite mbarrier phase.
The reference backend validates this order for every complete phase cycle and
for each allocation owner.

### `TileImpl.host_init`

```python
def host_init(self): ...
```

Optional.  Initializes host-side state for one tile instance.  For example,
this hook can set the cuTensorMap used by that tile instance.

Example:

```python
def host_init(self):
    T.evaluate(T.call_packed("runtime.cuTensorMapEncodeTiled", ...))
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

`kernel.semantic_plan()` builds and validates a `SemanticPlan` before any
physical policy is selected.  The plan snapshots variables, tensors, events,
tiles, and logical event edges; validation rejects a stale snapshot.

Lowering policies produce one `ExecutionPlan`.  Its device regions contain
ordered `TileProgram` objects, and each program contains the `ProgramStep`
sequence that a backend lowers.  Host regions explicitly list their logical
ownership in `HostRegionPlan.owned_tiles`.  Validation requires every logical
tile to belong to exactly one device or host region.  For example, the default
static policy emits device initialization, prefetch, waits, `RunStep`, and
notifications in source order:

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

`DeviceRegionPlan.fetch_steps` belongs to the general execution-plan contract
for custom schedulers and distributed backends.  The reference static backend
rejects it explicitly; it is distinct from a tile's early `prefetch` hook.
`RuntimeEventInitStep` is likewise reserved for custom backends because the
reference backend owns one built-in event initialization phase.  Each logical
edge in the reference backend must bind exactly one CTA-wide wait and notify,
reuse the logical coordinate maps, and publish the notification with a
device-scope release fence before its atomic signal.  Remote ranks and batched
endpoint counts are not supported by this single-device backend.

Each default tile program contains exactly one canonical `RunStep`: waits must
precede it, notifications must follow it, and predicate, repeat, index-map, and
profiling modifiers require a custom backend that validates their mapping to
logical tile instances.  The device epilogue ends with exactly one
`HookStep("smem_commit")`, which commits the dynamic shared-memory extent used
by the persistent scheduler and tile programs.

The reference static persistent queue also requires the concrete
tile-instance/event-coordinate dependency graph to be acyclic.  It checks a
coarse logical cycle at concrete coordinates and accepts it only when exact
enumeration proves that the coordinate-level graph and its tile-phase
projection are acyclic.  Other backends may implement cyclic protocols.  For a
coarse acyclic graph, waiting tile phases are emitted in stable topological
order so a persistent CTA does not
block on a later producer phase.

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
