<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# tirx tile-primitive dispatch: capability index

What each dispatch variant supports: memory scopes, directions, exec scopes,
distributed-register (frag) operands, and acceptance constraints. Consult
this BEFORE concluding that a case is unsupported or that a new primitive is
needed; the executable ground truth is the set of `register_dispatch(...)`
sites under `python/tvm/backend/cuda/operator/tile_primitive/` and their
`predicate` functions, and the normative correctness arguments live in
`tile_primitive_dispatch_proofs/`.

Dispatch selection (`dispatcher.py:273`): candidates sorted by
`(-priority, variant_name)`; the first whose predicates pass and whose impl
does not `fail()` wins. An explicit `dispatch="<variant>"` config filters to
that variant only (`dispatcher.py:263-266`).

Scope shorthand: `G` = global, `S` = shared (`shared` / `shared::cta` /
`shared::cluster`), `L` = local (per-thread registers / frag), `T` = tmem.
`DEFAULT_ALLOWED_PAIRS` (`copy/utils.py:75`) = G↔S, G↔L, S↔L.

Exec-scope prefixes in TVMScript: `Tx.warp.*` / `Tx.wg.*` (warpgroup) /
`Tx.cta.*` / `Tx.cluster.*` are scope-qualified forms of every tile
primitive (`python/tvm/tirx/script/builder/tirx.py:114-121`); a bare
`Tx.copy(...)` runs in the ambient scope.

## copy (op `copy`)

| Variant | Registration | Scopes / direction | Exec scope | Frag side | Key constraints | Config knobs |
| --- | --- | --- | --- | --- | --- | --- |
| `vec_256b/128b/64b/32b/16b` | `copy/vec_forced.py:194` | DEFAULT_ALLOWED_PAIRS, symmetric | thread only | no (single thread) | explicit `dispatch=` required; same dtype; each region exactly `num_bytes*8/elem_bits` elems; cache hints require global src | `dispatch=`, `cache="nc"`, `l1_evict`, `l2_evict`, `prefetch_size` |
| `vec_auto` (reg path) | `copy/vec_auto.py:39`, impl `vec_auto_reg.py:473` | L↔S, L↔G | thread/warp/warpgroup/cta, all threads active | **yes** — local side carries thread-axis TileLayout | R layout non-swizzle `TileLayout`; thread-axis subscope ≤ exec scope; sliced thread-axis offset provably 0; auto-vectorizes 128/64/32/16/8b by alignment | none (auto) |
| `vec_auto` (gmem↔smem path) | `vec_auto_gmem_smem.py:104` | G↔S | thread/warp/warpgroup/cta, all threads active | no (partition synthesized from `sctx.intra` thread count) | region elem count divisible by thread count; vec fits one swizzle chunk | none (auto) |
| `ldstmatrix` | `copy/ld_stmatrix.py:439` | L↔S (L→S = `stmatrix`, S→L = `ldmatrix`, `.b16 m8n8`) | warp/warpgroup/cta, all threads active | **yes** — warp-collective; exactly one lane axis ∈ {`laneid`,`tid_in_wg`,`tx`} | no replica on either side; R thread-section total % 32 == 0; S swizzle `per_element >= 3`; `num ∈ {4,2,1}` | `dispatch="ldstmatrix"` |
| `fallback` | `copy/fallback.py:108` | any valid copy | any (warp+ scopes elect a single thread) | ignores distribution (scalar loop) | last resort, priority 0, warns | none |

## copy_async (op `copy_async`)

| Variant | Registration | Scopes / direction | Exec scope | Frag side | Key constraints | Config knobs |
| --- | --- | --- | --- | --- | --- | --- |
| `tma` | `copy_async/tma.py:1615` | G↔S | single thread | no (hardware distributes) | same dtype; total elem counts equal (permuted regions allowed); planner: rank ≤ 5, swizzle atom, unit inner stride. Proof: `tile_primitive_dispatch_proofs/tma.md` | `mbar`, `cta_group`, `cta_mask`, `cache_hint`, `oob`, `gather_axis`/`indexer`, `mbarrier_addr`, `use_tma_reduce`, `tensor_map`, `tma_dtype`, `prefetch_tensormap`, `tensormap_l2_promotion` |
| `ldgsts` | `copy_async/ldgsts.py:330` | G→S only | thread/warp/warpgroup/cta, all threads active | no | elem count divisible by thread count; vec ∈ {128,64,32}b; caller commits/waits | `direct`, `prefetch_size`, `predicate`, `fill_mode` |
| `dsmem` | `copy_async/dsmem.py:203` | S→S cross-CTA (`cp.async.bulk`) | single thread | no | `remote_cta_id` + `mbar` required; contiguous ≥16B and %16 == 0 | `remote_cta_id`, `mbar` |
| `smem->tmem` (tcgen05.cp) | `copy_async/tcgen05_cp.py:883` | S→T | single thread | tmem dst; distribution from (shape, multicast) | routes: no `shape` → legacy `32x128b.warpx4`; `shape=` → generic planner; `shape=`+`desc_*` → explicit. Shapes: `128x256b`, `4x256b`, `128x128b`, `64x128b`(+`warpx2::*`), `32x128b`(+`warpx4`). Proof: `tile_primitive_dispatch_proofs/tcgen05_cp.md` | `shape`, `multicast`, `cta_group`, `decompress`, `desc_*`, tile/subtile strides |
| `tmem<->local` (tcgen05.ld/st) | `copy_async/tcgen05_ldst.py:454` | T↔L | **warpgroup only** | **yes** — local side must match a `tcgen05_atom_layout` | frag layout structurally equal to `tcgen05_atom_layout(shape,(rows,K),dtype)` for `16x64b/16x128b/16x256b`, or the `32x32b` default `(128,K):(1@tid_in_wg,1)`; TMEM datapath compat enforced; 32b-aligned col slice | frag shape implied by buffer layout |

## gemm / gemm_async

| Variant | Registration | Operand scopes | Exec scope | Key constraints | Config knobs |
| --- | --- | --- | --- | --- | --- |
| `gemm` / `mma.m16n8k*` | `gemm/mma_m16n8k_.py:260` | D/A/B/C all `local` (register frags) | warp/warpgroup/cta, full active set | instr ∈ `MMA_INSTRUCTIONS` (`m16n8k16`/`m16n8k8`, f16/bf16 in, f32 accum); no operand replica; stage A/B via e.g. `ldstmatrix` | `transpose_A/B`, `alpha`, `beta` |
| `gemm_async` / `tcgen05` | `gemm_async/tcgen05.py:1436` | C in tmem; A shared or tmem; B shared | single thread or warp | C fp32; dense dtypes f16/bf16/fp8/tf32; block-scaled fp4/fp8 with SF in tmem; `cta_group ∈ {1,2}`. Proof: `tile_primitive_dispatch_proofs/gemm_async.md` | `weight_stationary`, `cta_group`, `smem_desc`, `is_AB_tf32`, `descI` (block-scaled) |

## reduction (ops `sum` / `max` / `min`)

| Variant | Registration | Scopes | Exec scope | Frag | Key constraints | Config |
| --- | --- | --- | --- | --- | --- | --- |
| `local` | `reduction/local.py:472` | L→L | thread (sequential); warp/warpgroup (layout-driven) | **yes** — thread-axis TileLayouts partition src/dst; warp scope may `shfl_xor` | non-swizzle TileLayouts both sides; no zero-stride thread dims; no thread axis in replica; `thread_reduce=True` warp-only | `thread_reduce`, `accum` |
| `shared` | `reduction/shared.py:284` | S→S | thread/warp/warpgroup/cta | no | needs `threadIdx.x` only; dst size = spatial product | `accum` |
| `packed_add_sum` / `3input_maxmin` | `reduction/sm100_packed.py:238` | L→L | thread only | per-thread | fp32, 1-D src len ≥ 8, dst len 1, SM100+; priority 20 (beats `local`) | `accum` |

## elementwise (ops: `zero fill reciprocal sqrt exp exp2 silu add sub mul fdiv maximum cast fma`)

| Variant | Registration | Scopes | Exec scope | Frag | Key constraints |
| --- | --- | --- | --- | --- | --- |
| `reg` | `elementwise/register.py:33` | all operands `local` | thread/warp/warpgroup/cta, all threads active | **yes** — partition induced by anchor operand's thread axes | anchor non-swizzle TileLayout; scope-level anchor for collective scopes; NumPy-broadcast compat; replica signatures agree |
| `smem` | `elementwise/register.py:45` | all operands `shared*` | thread/warp/warpgroup/cta | no | operands have layouts; broadcast compat; scope sync emitted |

## permute_layout

| Variant | Registration | Scopes | Exec scope | Key constraints |
| --- | --- | --- | --- | --- |
| `warp_xor_swizzle` | `permute_layout/warp_xor_swizzle.py:410` | in-place S↔S transpose (register-staged) | warp only | dtype width ∈ {1,2,4,8,16}B; plain TileLayouts; volume % 32 == 0; per-thread P power-of-2 ≤ 32; XOR-swizzle must be bank-free both phases |

## Buffer views (dim surgery)

Tile-primitive operands are tensor + region; derived tensors come from
`Buffer` view methods (`python/tvm/tirx/buffer.py`), which never move data —
each returns a view sharing `data`, with the layout (iters + swizzle)
carried automatically, or raises. Prefer these over hand-written
`T.decl_buffer` restating strides/swizzle, and over `a*k:(a+1)*k` region
arithmetic (unflatten the dim, then index).

| Method | Semantics (torch-aligned name) |
| --- | --- |
| `view(*shape)` | layout-preserving reshape; can split inside an iter |
| `permute(*dims)` | reorder dims; swizzle-composed layouts supported |
| `unflatten(dim, sizes)` | split a dim row-major; one `-1` inferred |
| `flatten(start_dim, end_dim)` | merge adjacent dims |
| `select(dim, index)` | fix + drop a dim; `index` may be a dynamic PrimExpr |
| `narrow(dim, start, length)` | sub-range of a dim; multi-iter dims need inner-block alignment |
| `sub[...]` | numpy basic indexing: int drops, `a:b` narrows, `a::s` strides |
| `rearrange(pattern, **sizes)` | einops-style bijective rearrangement |

Statically known arguments are validated loudly; dynamic PrimExprs are the
caller's responsibility (region semantics). On swizzle-composed layouts the
folded offset goes into `elem_offset` only when it provably commutes with
the swizzle (a multiple of the swizzle period
`2^(per_element + atom_len + swizzle_len)`); otherwise it stays inside the
derived tile layout's `m`-axis offset (printed as `T.S[...] + offset`),
where `ComposeLayout` applies the swizzle to it. Behavior and address
equivalence are pinned by the `test_buffer_*` tests in
`tests/python/tirx/test_parser_printer.py`.

## Distributed register tiles (frags)

A `local`-scope buffer is a **frag** when its `TileLayout.shard` contains
iters whose axis `is_thread()` (`python/tvm/tirx/layout.py:475-560`).
Thread axes: `laneid`, `tid_in_wg`, `wid_in_wg`, `warpid`, `tx`, `wgid`;
memory axes: `m` (local storage), `TLane`/`TCol` (tmem), etc. A thread-axis
iter states which thread owns which logical coordinate; the `m` iters
describe each thread's private register storage.
`get_layout_thread_local_partition` (`tile_primitive/layout_utils.py:52`)
validates the partition (no zero-stride thread dims, no thread axis in
replica); `get_local_region` derives per-thread storage shape/region.

Declared with plain layout syntax, e.g. a warpgroup 64x64 tile where lane
bits and warp bits address the tile:

```python
TileLayout(S[(8, 8, 2, 4, 2, 4) : (1 @ laneid, 1, 2 @ wid_in_wg, 8, 1 @ wid_in_wg, 8 @ laneid)])
```

Frag-accepting variants and their extra requirements: `vec_auto` reg path
(most general L↔S/L↔G), `ldstmatrix` (L↔S, m8n8 b16 fragment pattern),
`tcgen05 ld/st` (T↔L, atom layouts only), `gemm mma.m16n8k*`,
`elementwise reg`, `reduction local`. `vec_forced`/`fallback` accept local
operands but do not interpret thread axes.

Allocation entry points:

- `T.alloc_buffer(shape, dtype, scope="local", layout=<thread-axis TileLayout>)`
  — generic custom frag; `Buffer.local()` (`python/tvm/tirx/buffer.py:383`)
  returns the per-thread storage view.
- `T.alloc_tcgen05_ldst_frag(instr_shape, (rows, K), dtype)`
  (`python/tvm/tirx/script/builder/ir.py:1852`) — tcgen05 ld/st atom frags;
  shapes `32x32b` (rows=128, reps 1..128), `16x64b`, `16x128b`, `16x256b`
  (rows=64/128).
- `T.alloc_cast_frag(src, dtype)` (`ir.py:1908`) — same distribution as
  `src`, dtype-cast, no cross-lane movement.
- Layout factories: `tcgen05_atom_layout`, `tmem_datapath_layout`,
  `wg_local_layout` (`python/tvm/tirx/layout.py`).

Note the two distinct tcgen05 shape tables: ld/st frag shapes
(`32x32b`/`16x*b`, TMEM↔register) vs `tcgen05.cp` copy shapes
(`128x256b`/`4x256b`/`128x128b`/`64x128b`/`32x128b`, SMEM→TMEM).
