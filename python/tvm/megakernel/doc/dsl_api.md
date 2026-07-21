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

- Spec layer: `KernelSpec`, `VarSpec`, `ScalarSpec`, `TensorSpec`,
  `EventSpec`, and `TileSpec`.  This layer describes tile stages, tensors,
  runtime scalars, logical events, and tuple-form wait/notify dependencies.
- Impl layer: `TileImpl`.  This layer connects a logical tile to the concrete
  implementation of that tile.

The spec layer corresponds to Step 3 in [workflow.md](workflow.md).  The impl
layer corresponds to Step 4.

The compiler works directly on the spec in two actions:

```text
verify: validate_kernel(spec)  -> all backend-independent checks
build:  lower_to_tirx_module(spec, LoweringOptions(...)) -> TIRX module
```

There is no intermediate plan object.  The build step derives its parameter
bindings, event-workspace layout, job IDs, and static queue phases from the
validated spec through private helpers that are not a user API.

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

Static regions let spec validation check bounds, disjoint writers, and
producer/consumer ordering.  Validation proves these properties when it can
exactly enumerate at most 4,096 bounded symbolic environments, 65,536 tile
points or tile pairs for one check, and 4,194,304 combined environment-point
operations.  It collects only variables referenced by that check.  A larger
bounded space uses deterministic boundary, midpoint, and fixed-seed samples
instead of failing.  Variables used by a static region still require explicit
ranges.  If a region is data-dependent rather than symbolically bounded,
declare it dynamic and give a non-empty reason.  Dynamic access still requires
an event dependency between its producer and consumer.

## Runtime scalars

```python
scalar = kernel.scalar(
    name: str,
    source: tuple[TensorSpec, tuple[ExprLike, ...]],
    dtype: str = "int32",
    range: tuple[int, int] | None = None,
)
```

Registers a runtime scalar: a symbolic integer whose value is read from a
device buffer at kernel runtime.  It is neither a host constant nor a kernel
parameter.  `source` is a `(tensor, index)` pair: `tensor` must be a
registered base tensor (not a region view), and `index` a static coordinate
into it whose rank matches the tensor rank.  Index entries are `int`,
`VarSpec`, or `ExprSpec` values whose variables are owned by this kernel;
they must not reference runtime scalars.

```python
num_tokens_post_pad = kernel.tensor("num_tokens_post_pad", (1,), "int32")
routed_rows = kernel.scalar(
    "routed_rows",
    source=(num_tokens_post_pad, (0,)),
    range=(1, 8192),
)
```

A `ScalarSpec` participates in integer expressions exactly like a `VarSpec`
(`+`, `-`, `*`, `//`, `%`, unary `-`, `.ceildiv(...)`), but it never
evaluates to a concrete value during validation: concrete evaluation of an
expression containing a runtime scalar is unresolved, and bound proofs use
its `range` when present (unbounded otherwise).  `range` follows
`VarSpec.range` semantics: an inclusive `(minimum, maximum)` pair with
`0 < minimum <= maximum`, used only for validation bound proofs.

Allowed contexts:

- `tile_num` axes.
- `wait`/`notify` coord_map return values.
- `init_count` callables, which may capture a runtime scalar (the callable
  itself must still return concrete positive integers during validation).

Forbidden contexts, rejected by validation:

- tensor shapes and event shapes, which must stay statically bounded so the
  backend can reserve storage;
- the scalar's own source index;
- tensor region starts and extents — use a dynamic tensor region with a
  non-empty reason for data-dependent access instead.

Tiles whose `tile_num` depends on a runtime scalar cannot be pre-enumerated
on the host.  The dynamic scheduler dispatches them at kernel runtime (see
*Dynamic scheduling* below); the static scheduler enumerates the grid at the
scalar upper bound and requires the tile to gate execution with a declared
`megakernel.run_predicate` attr.  Validation degrades gracefully around
runtime scalars: instance-enumeration proofs (event counts, static region
bounds, disjoint writers, event-coordinate ordering) are skipped for tiles
and events that depend on them, and tensor dataflow through them is checked
against the coarse logical event ordering instead.

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

### Endpoint scopes

```python
class MyTile(TileImpl):
    wait_level: str = "cta"                     # "thread" | "warp" | "warpgroup" | "cta"
    wait_mask: int = 0xFFFFFFFF                 # 32-bit thread mask
    notify_scope: tuple[str, int] = ("cta", 0)  # (scope, scope_id)
    pre_notify_scope: tuple[str, int] | None = None  # dynamic pre-notify; defaults to notify_scope
    notify_release: bool = True                 # release-fenced completion notifies
```

These optional class attributes declare the physical sync granularity the
tile implementation uses for its event endpoints: the scope at which it
waits on events (`wait_level` plus the participating `wait_mask`) and the
scope at which it signals them (`notify_scope`, a `(scope, scope_id)` pair
with the same legal scope values and a non-negative integer id).  One
consistent set applies to all of the tile's endpoints.  `pre_notify_scope`
overrides the scope of the dynamic scheduler's pre-notify-and-push step when
it must differ from the completion notify, and `notify_release` controls the
release fence on completion notifies.

Scope is impl metadata rather than spec data because it describes the tile's
internal cooperative granularity: a warpgroup-cooperative GEMM signals at
warpgroup scope regardless of which spec it runs in.  The defaults match the
reference static backend (CTA-wide waits with a full mask, CTA-scope
notifies), so the simple path stays correct without declaring anything.

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

## Lowering

One build system lowers every validated spec: the runtime-library builder in
`transform.runtime_build`, which assembles the persistent kernel from the
`tvm.megakernel.runtime` building blocks.  Two actions on a spec:

```python
from tvm.megakernel.transform import (
    LoweringOptions,
    build_runtime_kernel,
    lower_to_tirx_module,
    validate_kernel,
)

validate_kernel(kernel)  # verify: every backend-independent check
mod = lower_to_tirx_module(  # build: the persistent device kernel module
    kernel,
    LoweringOptions(scheduler="static"),  # default; or "dynamic"
)
```

`validate_kernel` runs all validation directly on the `KernelSpec` and returns
the spec for chaining; `KernelSpec.validate()` is the same call.
`lower_to_tirx` returns just the persistent device kernel and
`lower_to_tirx_module` wraps it in an `IRModule` — the module carries only
the kernel because queue construction is host-side.  `LoweringOptions` carries
the scheduler kind (`scheduler="static"` by default, or `"dynamic"`), the
managed shared-memory chunk size (`smem_chunk_size`), and backend `attrs`:
hardware overrides (`sm_count`, `num_threads`, `warps_per_warpgroup`,
`warpgroup_count`, `warp_size`, `max_dynamic_smem`), `profiler`,
`tile_coalescing`, and `debug_scheduler`.

The richer entry point additionally derives everything the host must upload:

```python
build = build_runtime_kernel(kernel, LoweringOptions(scheduler="static"),
                             var_values={"rows": 12})
build.module                # the device kernel IRModule
build.exec_queue            # static: (sm_count, 128) int32 central queue
build.queue_tasks/head/tail # dynamic: MPMC seed arrays (int32)
build.event_workspace_size  # int32 cells to allocate and ZERO before launch
build.drain_events          # dynamic drain-event metadata
```

`var_values` gives concrete integers for symbolic `VarSpec`s; it is required
only when a *seeded* `tile_num` is symbolic (the device kernel itself is
symbolic-safe).  Host contract: allocate the event workspace with
`event_workspace_size` int32 cells and **zero it before every launch** (the
two-phase event protocol starts from zeroed cells), upload the queue arrays
from the build result, launch, and re-upload the queue arrays plus re-zero
the workspace between launches — the device mutates both.

The static build emits one persistent kernel around a central per-SM task
queue: event initialization (job 29), waiting-free tile grids, the event-init
barrier (job 30), waiting tile grids in stable topological order, and the end
marker (job 31), striped round-robin over the SMs with END padding.
Inside the persistent loop, each tile instance runs `device_init`, `prefetch`,
its declared waits, `run`, and its declared notifications in that order;
every notification is published with a device-scope release atomic.

The builder concretizes all buffer shapes: tensor params take their declared
(possibly var-symbolic) shapes and event regions take their upper-bound
shapes, rather than passing extents as symbolic shape parameters the way the
hand-written kernels do.  This is proven perf-neutral: across the production
MoE batch range the runtime-built kernels match the hand-written ones within
3% median latency on all 24 benchmark cells (20 of 24 within 1%), with
byte-exact queue contents.

The static persistent queue also requires the concrete
tile-instance/event-coordinate dependency graph to be acyclic.  It checks a
coarse logical cycle at concrete coordinates and accepts it only when exact
enumeration proves that the coordinate-level graph and its tile-phase
projection are acyclic.  For a coarse acyclic graph, waiting tile phases are
emitted in stable topological order so a persistent CTA does not block on a
later producer phase.

## Dynamic scheduling

`LoweringOptions(scheduler="dynamic")` emits the MPMC persistent scheduler:
a device-side multi-producer multi-consumer queue (`exec_task`/`exec_head`/
`exec_tail`, 32768 slots) replaces the static central queue, events use the
two-phase dynamic semaphore, and only the event-init tasks (job 29, one per
event-workspace region) and the entry tiles (no waits) are seeded on the
host.  Every other tile is *dynamically dispatched*: when the last producer
task for an event cell starts, it pushes the consumer's tasks into the queue
from the device.

### The two-phase event protocol

Each dynamic event cell is initialized to `count * (base + 1)` where `count`
is the number of tasks notifying that cell and `base = 1 << 16`.  A task
notifies twice: a *pre-notify* (−1) at the start of its dispatch branch and a
completion *notify* (−base) after its run.  The pre-notify that observes
`old % base == 1` is the last one for the cell and *triggers the push* of the
downstream tile's tasks, so the scheduler is fed before the pushed work
exists (this is what keeps the persistent workers deadlock-free).  Event
cells are written by the init tasks with a plain store, and racing
pre-notifies spin until the store lands, so initialization order is safe.

### Dispatch synthesis

The builder requires a single-chain graph shape, matching the production
kernels it generalizes:

- exactly one terminal tile (a tile without notifies);
- every non-entry tile has exactly one wait, its waited event has exactly one
  producer tile, and that producer notifies the event exactly once;
- one outgoing dispatch rule per source tile (no fan-out/fan-in);
- the tile-level event dependency graph is acyclic (persistent workers block
  on waits);
- entry tiles are host-seeded, so their `tile_num` must not depend on
  runtime scalars.

For each dispatched tile the rule is derived mechanically:

- **Pusher**: the producer of the tile's waited event (the event edge).
- **Push scope**: the pre-notify and push run at the source impl's
  `pre_notify_scope` (falling back to `notify_scope`); the push granularity
  must not exceed the pre-notify scope.
- **Push indices**: wait/notify coord maps are probed as tuples of integer
  constants and bare `m`/`n`/`k` axis references.  An axis shared by the
  producer's notify coord and the consumer's wait coord *pins* that target
  axis to the source task index; remaining *free* axes are enumerated by the
  push index `m`-major, exactly reproducing the production
  `(push_idx // 12, push_idx % 12, 0)`-style maps.
- **Push count**: the product of the free axes' extents, evaluated at kernel
  runtime — extents may load runtime scalars from their source buffers.
- **Scalar ancestry**: every runtime scalar in the count must be host-planted
  or produced by the pusher or one of its event-DAG ancestors, so the value
  is written before the push can fire.

### Drain synthesis and kernel termination

The terminal tile gets a synthesized drain event (shape `(1,)`, laid out
after the user events in the workspace; there is no completion cell in
dynamic mode).  Its count is `(base + 1) * grid_count`: for a static or
var-dependent terminal grid the init tasks write it (vars lower against the
kernel parameters); for a scalar-dependent grid it is *runtime-initialized*
on device by the tile producing the scalar's source tensor (or by the
terminal tile's pusher when the scalar is host-planted), after its run and a
CTA barrier.  The terminal's last drain pre-notify pushes one END task per
SM, ending every worker's scheduler loop.  A spec may instead declare its
drain event explicitly with `event.attrs["megakernel.drain"] = True` (shape
`(1,)`, no wait/notify edges); the static builder then treats it as an
ordinary event, keeping one workspace layout across both schedulers.

### `tile_coalescing`

`LoweringOptions(attrs={"tile_coalescing": {"down": 4}})` groups `q` n-axis
slices of a dynamically dispatched tile per pushed task: the scheduled grid
becomes `(m, n // q, k)`, the push count and drain count use the divided
grid, and the tile's `run` hook is invoked `q` times per task with
`n = n_idx * q + i`.  Requires a non-entry tile with a statically known
n extent divisible by `q`.

### `megakernel.*` attribute namespace

- `tile.attrs["megakernel.dispatch"]`: explicit push rule (escape hatch)
  overriding synthesis.  A dict with required `source` (pusher tile name),
  `count` (int/ExprLike or `callable(m, n, k)`), and `indices`
  (`callable(push_idx, m, n, k) -> (m, n, k)`); optional `event` (default:
  the tile's single waited event), `pre_scope` (default: the source impl's
  `pre_notify_scope`), and `push_level` (default: the pre-notify scope).
  Validation rejects unknown/missing keys, unknown tiles/events, an event
  the tile does not wait on, malformed scopes, and a `push_level` wider than
  the pre-notify scope.
- `tile.attrs["megakernel.run_predicate"]`: `(axis, op, expr)` with `op`
  currently always `"lt"`; legalizes a *static* scalar grid by gating the
  tile's run to instances with `indices[axis] < expr` while the host queue
  enumerates the scalar upper bound.
- `event.attrs["megakernel.drain"]`: marks a spec-declared drain event
  (shape `(1,)`, no wait/notify edges); see *Drain synthesis*.
- `tile.attrs["megakernel.job_id"]`: pins the tile's physical job id (a
  non-negative integer) in the packed task wire format; unpinned tiles take
  their spec registration order.  Job ids must stay clear of the reserved
  event-init (29), wait-init (30), and END (31) ids.

### Profiler and endpoint scopes

Endpoint scopes (`wait_level`/`wait_mask`/`notify_scope`/
`pre_notify_scope`/`notify_release`) are documented under *Endpoint scopes*
and work identically in dynamic mode; both semaphores implement `cta` and
`warp` waits.  The profiler is duck-typed: a tile impl may define a
`profile_event` attribute (an `Enum` member or int) to be wrapped in
start/stop pairs when `attrs["profiler"]` is on; the dynamic scheduler
itself additionally emits FETCH/PUSH events around its queue operations.

### Host contract (dynamic)

`build_runtime_kernel(..., scheduler="dynamic")` returns the MPMC seed
arrays (`queue_tasks`/`queue_head`/`queue_tail`) plus
`event_workspace_size` and `drain_events` metadata.  Seeds are the
event-init tasks followed by the entry-tile grids in spec order.  Upload all
three queue arrays and zero the event workspace before every launch; the
drain cell is initialized by the kernel itself (seed-written for static
grids, device-written for scalar grids), so the host never computes it.

## `KernelSpec.lower`

```python
prim_func = kernel.lower(options=None)
```

Validates and lowers the spec with the runtime-library builder (static by
default); equivalent to `lower_to_tirx(kernel, options)`.
