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

# Placement-Correctness Proof of the TMA copy_async Dispatch

Subject: `python/tvm/backend/cuda/operator/tile_primitive/copy_async/tma.py`
(hereafter the **planner**; all line numbers refer to the current version of that
file unless another file name is noted).

Proposition (informal): for every `(gmem view, smem layout, region, config)`
combination accepted by the planner, the device-side instruction sequence
(`cp.async.bulk.tensor.*`) emitted by `copy_tma_impl` together with the host-side
`cuTensorMapEncodeTiled` call implements, under the instruction axioms and the trust
base (§2, §5), exactly the logical copy semantics of
`Tx.copy_async(dst_region, src_region)`; for rejected combinations, the planner throws
`DispatchFail` / `ValueError`, the dispatcher falls back to other variants or reports
an error as a whole, and never emits incorrect IR.

**Key difference from the sibling documents**: the `tensor_map_dim_order` config knob
has been **removed** (`_assemble_plan` retains only the single
chain-DESC ordering path; passing the old key triggers the
`"tensor_map_dim_order was removed"` rejection, lines 1251–1256). The main theorem is
therefore **unconditional** on the acceptance domain — there is no conditional branch
of the form "the caller contract selects the natural ABI mode"; the descriptor
dimension order is always derived from the contiguous chain of the declared smem
layout, and this (under AX-FILL) is the **only** placement-correct order
(end note of §3 L5). The remaining completeness boundary is pinned down by the test
`test_copy_tma_declines_non_derivable_fold_layout`: a hand-written rope-fold layout
whose placement is not chain-derivable is **loudly rejected**
("TMA innermost dim must have unit stride");
this is a reject-safe completeness limitation, not a bug (completeness discussion in §4).

Suggested reading order: §0 overview → §1 definitions → §2 axioms and trusted lemmas →
§3 step-by-step lemmas → §4 theorems → §5 trust base → Appendix A (the full
code-checkpoint ↔ lemma cross-reference table).

Sibling documents: `tcgen05_cp.md` (hereafter the
**cp proof**) and `gemm_async.md` (hereafter the
**gemm proof**). This document reuses the notation and the proven / trusted lemmas of
the cp proof (T-SLICE / T-APPLY / T-CANON / T-GROUP / T-ARITH, swizzle semantic
equivalence) without re-proving them.

---

## 0. Overview: the TMA instantiation of P1 / P2 / P3

### 0.1 Problem shape

A `Tx.copy_async` call carries `(dst_region, src_region, config)`, with direction
global↔shared (lines 1220–1231 determine `g2s` / `s2g`). The gmem side is an arbitrary
`TileLayout` folded view (e.g. FlashMLA's 5D Q fold:
`(64,128,2,4,3):(1,512,256,64,65536)`); the smem side is an arbitrary declared layout
(possibly with swizzle). The output is:

- on the host side, one `cuTensorMapEncodeTiled` (a CUtensorMap of rank ≤ 5,
  deduplicated by cache key);
- on the device side, an unrolled issue loop, each iteration emitting one
  `cp.async.bulk.tensor.<r>d`.

The semantics of the logical operation is a **pair set** Pairs (which smem physical
address ↔ which gmem physical address, §1.3 Def 4); the semantics of each TMA
instruction is also a pair set (given by AX-TILE + AX-FILL).

> **Correctness criterion**: the union of the pair sets of the emitted instructions
> = Pairs (as set equality, and under g2s every smem destination location is written
> exactly once; dually for s2g).

### 0.2 Instantiation of the generic three-stage decomposition

- **P1 (gmem view normalization and grouping, the L1 layer)**: gmem is grouped by
  buffer shape, canonicalized within each group, and multi-iter groups are split under
  an alignment condition (`_split_multi_iter_group`); smem is sliced, canonicalized,
  and regrouped by the "ext>1 copy shape" (`_build_l1_result`). Every step is a
  bijective reindexing of the pair set, ultimately reducing both sides to a common
  mixed-radix digit space (§3 L1/L2).
- **P2 (single-transaction placement lemma, the technical core of this proof)**:
  AX-FILL says TMA fills smem in box linear order (descriptor dim 0 fastest); the
  chain invariant of `_find_contiguous_chain_prefix` gives
  `stride(chain[m]) = ∏_{m'<m} extent(chain[m'])`; `_assemble_plan` arranges the box>1
  dims in descending chain order (outer→inner), so **the fill stride
  `∏_{m'<m} box_{m'}` of the m-th (counting from the inside) box>1 dim equals exactly
  the declared smem stride of that chain shard** — the hardware fill ≡ the declared
  layout (§3 L5; host-side mechanical recomputation in the table at the end of L5).
- **P3 (tiling completeness + semantic preservation of shrink)**: all unselected
  shards become issue axes, and the emission loop enumerates them via a mixed-radix
  bijection; the plan produced for every j (chain prefix length) satisfies P1/P2/P3
  on its own (j only moves the dividing line between "covered by the box ↔ covered by
  the loop"), so `_build_plan_with_shrink`, taking the first plan that passes
  validation as j decreases from max to 0, is a **semantics-preserving search**, not an
  approximation (§3 L8/L9).

### 0.3 Safety criterion

The acceptance decision and the emission share the same code path (`copy_tma_impl` →
`_build_l1_result` → `_build_plan_with_shrink`, lines 1208–1265); any precondition
failure leaves via `DispatchFail` (`fail()`, dispatcher.py lines 78–81) or
`ValueError`; the dispatcher (`python/tvm/tirx/operator/tile_primitive/dispatcher.py`
lines 304–329) catches it and tries other variants, aggregating an error if all fail.
There is no "half-accepted" state.

---

## 1. Definitions

### 1.1 Notation and index spaces

- gmem region: per dim `[g_st_d, g_st_d + g_ext_d)` (lines 1233–1236); likewise for
  the smem region. `N = ∏_d s_ext_d = ∏_d g_ext_d` (guaranteed by the
  total-element-count equality check of the predicate `_validate_tma_copy_op`,
  lines 1590–1605; note that the TMA predicate **relaxes** the generic copy
  validation's "non-unit dims equal position by position" requirement, to accommodate
  permuted regions such as RoPE `global(32,64,1) → shared(64,32)`,
  docstring at lines 1568–1574).
- row-major flattening `n ∈ [0, N)`: last dim fastest; the `SplitCoord` convention
  matches cp proof §1.1.
- dtype byte width `w_B = dtype_bits / 8`; `E₁₆ = 16 / w_B` (elements per 16B unit).

### 1.2 Planner data types (definition anchors, with line numbers)

- `GmemIter(shape, stride, copy_start, copy_ext)` (lines 70–87): one gmem logical dim
  after multi-iter group splitting; `is_ext1 ⟺ copy_ext = 1`.
- `SmemShard(extent, smem_stride)` (lines 90–95): one smem shard after slicing +
  regrouping.
- `SmemGroup(shards, bound_gmem_iter_idx)` (lines 98–107): the shard sequence
  (outer→inner) paired with one ext>1 gmem iter; the product of the extents = that
  iter's `copy_ext`.
- `Segment` (lines 110–134): one reshape segment produced by cutting at the chain
  prefix (§3 L4).
- `DescDim(shape, stride, box, coord_base)` (lines 137–144): one cuTensorMap dim;
  `stride` is the gmem element stride (in plan-element units).
- `IssueAxis(extent, dim_idx, coord_advance, smem_stride)` (lines 147–159): a loop
  axis arising from an unselected shard: each step advances the coordinate of dim
  `dim_idx` by `coord_advance` and the smem region by `smem_stride`.
- `TmaPlan(swizzle_mode, dims, issue_axes, tensor_ptr, elem_bytes, elem_dtype)`
  (lines 162–217): `dims` is stored in cuTensorMap **outer→inner** order (at emission
  it is `reversed` so that dim0 = innermost, lines 1525–1527); `offsets_and_coords`
  (lines 199–217) decodes the flat loop variable via mixed radix into
  `(smem offset, per-dim coordinates)`.

### 1.3 Logical copy semantics (Pairs) and OOB extension

**Def 1 (gmem address)**: the gmem view is a pure-memory `TileLayout` (enforced at
lines 405–410); `Φ_g : [0,N) → element address`: the row-major index `n` within the
region decomposes into the digits `x_i ∈ [0, copy_ext_i)` of each (post-split) iter,
and `Φ_g(n) = Σ_i (copy_start_i + x_i)·stride_i`.

**Def 2 (smem address)**: `L_s = _to_tile_layout(s_buf.layout).canonicalize()`
(the linear part of the swizzle, lines 225–232 + 413–414); after slicing to the region,
`Φ_s : [0,N) → linear (pre-swizzle) element address`; the physical address =
`Swz(Φ_s(n))` (`Swz` is the swizzle permutation of the declared layout, cp proof §1.3).
Planning and emission share the same `_to_tile_layout` (planning: line 414; the
emission's `decl_buffer` layout: lines 1473/1506), so both sides interpret the
"linear part" identically.

**Def 3 (digit correspondence convention)**: after slicing and canonicalization, the
smem-side flat index is **identified**, via `sliced_smem.group(extgt1_shape)`
(lines 544–547), with the row-major flat index of the gmem-side ext>1 iters — the
group succeeds if and only if the smem canonical iter structure can be cut at the gmem
copy_ext boundaries (T-GROUP). This identification is exactly the operational contract
of `Tx.copy_async` for TMA, and also the way the GPU tests reconstruct the host
expectation (`test_tma.py` lines 2354–2366: source value = linear gmem offset, read
back through the declared layout and compared element by element).

**Def 4 (Pairs)**:

```
Pairs = { ( Swz(Φ_s(n)),  Φ_g(n) )  :  n ∈ [0, N) }
```

Under g2s the first component is the write location and the second the read location;
dually for s2g.

**OOB extension**: when the caller's region extends past the declared shape of the
gmem view (a legal usage; the `oob` config selects the fill value), the "source value"
at an out-of-bounds `Φ_g(n)` is defined as the fill constant (0 or NaN,
`_normalize_oob_mode` lines 244–256); §3 L4 proves that the hardware OOB decision
agrees **element by element** with this logical out-of-bounds set (relying on the
`u | G` alignment check). In the s2g direction, the hardware suppresses out-of-bounds
writes (AX-TILE).

**Def 5 (side-effect semantics and equivalence)**: the effect of the emitted IR = the
union of the pair sets produced by each TMA instruction per AX-TILE/AX-FILL.
**Equivalence** := that union = Pairs and (under g2s) each destination location is
written exactly once. The asynchrony contract is as in the cp proof: mbarrier arrival /
waiting is the caller's responsibility (`mbar` is merely passed through,
lines 1303–1306); the proposition speaks only of the final writes.

### 1.4 Well-formedness assumptions

- **WF-DECL (declaration truthfulness, caller contract)**: the declared smem layout is
  the layout by which the consumer actually reads, and (as an address map) it is
  injective on the region. **The planner cannot in principle falsify this
  assumption** — any bijective layout declaration is compatible with any physical byte
  intent; the semantics is fixed by the agreement between the writing and reading ends
  (the isomorphism argument in the gemm proof's Theorem 1, condition 2). The small_topk stride-lie incident is an
  instance of violating WF-DECL: a lying declaration is faithfully executed and
  surfaces as a numerical error — the uniqueness note of §3 L5 rules out any
  alternative dim order that could mask it; the remedy is to fix the declaration, not
  to add guardrails.
- **WF-REGION**: the region lies within the buffer's logical shape (out-of-bounds is an
  upstream legality contract; deep-dim overrun of the gmem view is legalized by the OOB
  extension, see Def 4).
- **WF-ALIGN**: the smem destination base satisfies the period alignment of the swizzle
  mode (an allocator contract, not validated by the planner — §5 item 4); the gmem
  base is 16B-aligned (wrapper ICHECK,
  cuda_device_api.cc line 665).

---

## 2. Axioms and trusted lemmas

The `T-` prefix refers to cp proof §2.1 (T-SLICE / T-APPLY / T-CANON / T-GROUP /
T-ARITH). The `AX-` prefix denotes hardware / driver / codegen axioms, grounded in the
CUDA Driver API documentation + PTX ISA 8.8 + B200 probe measurements.

### 2.1 AX-ENC (cuTensorMapEncodeTiled field semantics)

The semantics of the CUDA driver's `cuTensorMapEncodeTiled(tensorMap, dtype, rank, ptr,
globalDim, globalStrides, boxDim, elementStrides, interleave, swizzle, l2, oobFill)`
(Driver API §CUDA_TENSOR_MEMORY; this repo's FFI wrapper
`src/backend/cuda/runtime/cuda_device_api.cc` lines 400–772 mirrors it item by item
with upfront validation):

1. **Dim order**: dim 0 = the innermost (element-contiguous) dim. `globalStrides` has
   only rank−1 entries; `globalStrides[i]` = the **byte** stride of dim i+1; **the
   stride of dim 0 is implicitly = the element size** (wrapper doc lines 410–412).
   ⟹ the planner must prove that the innermost dim has element stride = 1 (L7).
2. **Domain validation** (wrapper ICHECKs, all failing loudly at host time):
   rank ∈ [1,5] (lines 431–433); `globalDim > 0 ∧ ≤ 2^32` (lines 458–460);
   `globalStrides % 16 == 0 ∧ < 2^40` (lines 462–469); `boxDim ∈ [1, 256]`
   (lines 471–475); `elementStrides ∈ [1,8]` (lines 477–481);
   `boxDim[0]·elemSize % 16 == 0` (lines 692–695); swizzle mode vs
   `boxDim[0]·elemSize ≤ 32/64/128` bytes (lines 711–724); `oobFill = NaN` requires a
   floating-point dtype (lines 697–704); `tensor_ptr` 16B-aligned, `tensor_map`
   64B-aligned (lines 665–666).
3. **dtype mapping**: DLDataType → CUtensorMapDataType (lines 493–581; fp8 → UINT8,
   lines 562/566); an optional trailing argument `force_cu_dtype` overrides it
   (e.g. TFLOAT32 = 11, lines 583–587 + tma.py 1283–1289) — same byte width, changing
   only the rounding semantics on load.
4. **OOB semantics**: box elements whose coordinates exceed `[0, globalDim)` are filled
   with 0 on load (`OOB_FILL_NONE`) or NaN (`OOB_FILL_NAN_REQUEST_ZERO_FMA`); stores
   are suppressed.

The planner's emission (lines 1516–1537) passes `*reversed(plan.shape)`,
`*reversed(tma_g_strides_for_map)`, `*reversed(plan.box_dim)` — the plan stores
outer→inner, the API numbers inner→outer, and the double reversal is consistent
(checked in L10). Unit conversion:
`tma_global_strides = stride · plan.elem_bytes` (elements→bytes, line 1323),
`element_strides = [1]*rank` (line 1326, no strided sampling).

### 2.2 AX-TILE (the gmem walk of a single dense tile instruction)

`cp.async.bulk.tensor.<r>d ... [smem_ptr], [tensormap, {c_0..c_{r-1}}] ...`
(PTX ISA 8.8 §9.7.9.24.9): with the coordinate vector `c` (c_0 = dim0) as the origin,
for every multi-dimensional coordinate `x` within the box (`x_k ∈ [0, boxDim_k)`):

- g2s: read one element from gmem address `ptr + Σ_k (c_k + x_k)·S_k`
  (`S_0 = elemSize`, `S_{k≥1} = globalStrides[k-1]`, in bytes), filling per AX-ENC.4
  when out of bounds; write it to smem (the placement is given by AX-FILL).
- s2g: the dual direction — read from smem in the same placement order and write gmem,
  suppressing out-of-bounds writes.
- `.reduce` (`use_tma_reduce`, line 1320 + lines 1448–1459): the write side becomes a
  reduction read-modify-write (scope note §2.5).

### 2.3 AX-FILL (the placement axiom — this proof's hardware anchor, pinned by probes)

**Content**: a tile instruction places data on the smem side in **box linear order**:
the element at box coordinate `x = (x_{r-1},…,x_0)` (x_0 = descriptor dim 0, fastest)
lands at the linear offset

```
smem_linear_off(x) = Σ_k x_k · ∏_{k' < k} boxDim_{k'}      (element units)
```

relative to the destination pointer; the swizzle mode then merely **permutes 16B
blocks** within its repetition period (`Swizzle<s,4,3>` semantics as in cp proof §1.3;
when the base is period-aligned the phase agrees with `SwizzleLayout`). That is:
physical byte address = `Swz(dst_linear + off·elemSize)`.

**Evidence chain**:
1. A B200 raw-smem read-back probe (an on-device probe that copies a
   FlashMLA-Q-shaped 5D folded view into smem with declared strides (64,64,2,4), dumps
   the raw linear smem, and decodes each logical element's landing spot; run with both
   sw=none and sw128). Its conclusions: ① TMA fills smem in box linear order
   (dim0 fastest); ② the chain-DESC reordering is exactly the correspondence derived
   from the declared smem layout strides, correct-by-construction; ③ the
   gmem-positional ("natural") order is the unfaithful mode.
2. GPU element-level test
   `test_copy_tma_optimized_folded_view_placement_matches_declared_layout`
   (test_tma.py lines 2291–2366): the same gmem fold, against two different declared
   strides (interleaved `(1,64,4096,8192)` and half-major `(1,64,16384,4096)`), matches
   the declared layout element by element in each case — if the fill were not box
   linear order, or the dim order were not derived from the declared strides, the two
   middle dims of the half-major case would necessarily be scrambled (the docstring at
   lines 2297–2304 states this criterion explicitly).
3. Host structural pin
   `test_copy_tma_optimized_dim_order_derives_from_declared_smem_layout`
   (lines 2268–2288): the two declarations produce **different** descriptor dim
   orders / stride encodings (golden at lines 2240–2266), ruling out the interpretation
   that "the dim order is independent of the declaration".

### 2.4 AX-GATHER4 (tile::gather4 semantics)

`cp.async.bulk.tensor.2d....tile::gather4` (PTX §9.7.9.24.9.1): a 2D TensorMap; the
coordinates are (column coordinate c_0, four runtime row coordinates r_0..r_3); the
hardware reads the `[c_0, c_0+boxDim_0)` segment of gmem row `r_i`, and writes the four
rows **contiguously** starting at the smem destination pointer (boxDim_0 elements per
row, swizzle as in AX-FILL). The planner's per-chunk emission is covered in L9.
**Row-pitch contract (caller contract)**: the declared layout's stride along
`dst_gather_axis` must equal the hardware row width (the boxDim_0 row bytes / swizzle
atom convention); the planner addresses only chunk starts through the declared layout
(4-row granularity, lines 1340–1343), and the placement of the four rows within a
chunk is not validated at dispatch — the GPU gather4 smoke (test_tma.py
lines 1370–1401) pins it under standard layouts; exotic layouts remain a caller
contract.

### 2.5 AX-MC / AX-EXEC

- **AX-MC**: `cta_mask` multicast (g2s only, lines 1297–1301) writes the same data to
  the corresponding smem of the masked CTAs within the cluster; `cta_group::2`
  (sm_100a) paired-CTA semantics + the `sm100_2sm_leader_smem_addr` mbarrier conversion
  (lines 1357–1362). Both lie outside the layout algebra; this document reads them
  under the single-CTA projection (the same convention as the cp proof's asynchrony
  contract, §1.4 there). Likewise outside the Pairs
  algebra: `use_tma_reduce` (the write side read as the read-modify-write
  generalization of the plain store) and an `external tensor_map` (equivalent to the
  explicit-descriptor path, at the caller's own risk — the same
  compatibility-retained surface as the gemm proof's block-scaled descI path, L0
  there);
  `prefetch_tensormap` is purely performance.
- **AX-EXEC (codegen fidelity, trusted)**: `T.unroll(total)` fully unrolls;
  `T.meta_var` inlines;
  `T.decl_buffer(..., elem_offset = base + s_offset, layout = tile part)`'s
  `ptr_to(s_st)` produces a smem pointer of "linear-part address + offset";
  `T.ptx.cp_async.bulk.tensor.*` generates a single PTX instruction argument by
  argument (coordinate order = argument order = inner→outer). Compile-level pins: the
  `impl_spec` structural golden of `test_copy_tma_codegen` (`_build_expected_impl`,
  test_tma.py lines 264–379) + the various codegen tests (lines 1990–2131).

---

## 3. Step-by-step lemmas

Each lemma notes the planner code lines and its role within P1/P2/P3. "Reject" means
leaving by throwing `DispatchFail` / `ValueError` (safety in Theorem 2).

### L0 (routing and config normalization; scoping)

- Direction determination (lines 1220–1231): `global→shared* = g2s`,
  `shared*→global = s2g`, otherwise rejected with `ValueError`.
- `oob` (lines 244–264 + 1249–1250): `None/'zero' → fill 0`, `'nan' → fill 1`
  (restricted to floating dtypes, rejected at lines 254–255); the `raise` at line 264
  is unreachable after normalization (defensive). Once the plan is settled, it is
  re-checked against `plan.elem_dtype` (lines 1267–1275: when merge+promote retypes the
  descriptor to uintN, this rejects early at dispatch rather than blowing up late at
  host init; pinned by `test_copy_tma_oob_nan_declined_after_promotion`, test_tma.py
  lines 2530–2550).
- **removed knob** (lines 1251–1256): `"tensor_map_dim_order" in config` ⟹
  `fail("copy_async(tma) tensor_map_dim_order was removed: ...")`. Negative test:
  `test_copy_tma_rejects_removed_tensor_map_dim_order` (test_tma.py lines 2214–2237).
  Loudness: this rejection is a `DispatchFail`, so when the call site does not pin
  `dispatch="tma"` and another viable variant exists, the fallback absorbs the
  migration hint silently (the copy semantics remain correct — the other variant is
  also a correct copy; the aggregated error carries the reason when all variants
  fail).
- `tma_dtype` (lines 1283–1289): only `tf32/tfloat32`, and requires `plan.elem_dtype`
  to be in the float32 family (automatically rejected after promotion retyping,
  lines 1287–1288); mapped to `force_cu_dtype = 11` (AX-ENC.3).
- `tensormap_l2_promotion` (lines 267–288): domain `[0,3]` / an alias table, otherwise
  rejected.
- `cache_hint` (lines 291–296): a string (static hint) or an expression (runtime cache
  policy; `_normalize_cache_config` passes other types through as cache-policy
  expressions, backstopped by codegen-level type checking).
- `cta_group` default (lines 1293–1295: sm_100a → 1, otherwise −1 = no qualifier);
  `cta_mask` restricted to g2s (assert at lines 1297–1301); `mbar` required for g2s
  (lines 1303–1306); `mbarrier_addr` has the three states bool/IntImm/PrimExpr
  (lines 1307–1319, rejected when not g2s).
- gather normalization (lines 323–332 + 335–384): indexer type, g2s only
  (lines 355–356), `gather_axis = 0` only (lines 357–361), length a multiple of 4
  (lines 362–363), exactly one dst extent = len(indexer) (lines 365–371), source
  gather-axis min = 0 (absolute coordinates, lines 373–377). Plan-input rewrite:
  `g_ext[gather] = 1`, `s_ext[dst_gather] = 1` (lines 379–384) — the planner plans a
  "single-row copy", leaving the row dim to AX-GATHER4's runtime coordinates (L9). ∎

### L1 (gmem grouping and multi-iter splitting = mixed-radix factorization of the region; P1)

First half of `_build_l1_result` (lines 502–529):

1. Swizzle-mode extraction (lines 510–512, `get_swizzle_mode_from_layout`, tma_utils.py
   lines 98–142): undecidable ⟹ reject; flipped layouts with `swizzle_len ≥ 1` and
   `swizzle_inner=False` are **loudly rejected** (tma_utils.py lines 126–134; the
   docstring carries the exhaustive argument that the two permutation directions
   coincide only on blocks whose swizzle bits are all zero, plus the sw=0 identity
   exemption; pinned by `test_copy_tma_rejects_flipped_swizzle_inner`, test_tma.py
   lines 2185–2211); `per_element/atom_len` remain only a structural trust convention
   (§5 item 4).
2. Pure-memory axis assertion on both sides (`_assert_memory_only`, lines 235–241).
3. gmem grouped by buffer shape (lines 417–422, T-GROUP: `Φ_g` invariant, group
   product = dim length; reject on failure).
4. `_split_multi_iter_group` (lines 430–481): in-group canonicalization (T-CANON) +
   dropping unit shards yields t iters. t = 0 (degenerate dim length 1) ⟹ placeholder
   `GmemIter(1, 0, copy_start, copy_ext)` (lines 445–448; in that case necessarily
   `copy_ext = 1`, since group product = dim length = 1). t = 1 passes through.
   t ≥ 2: let `u = ∏_{k≥1} extent_k` (the product excluding the outermost iter);
   require `u | copy_start ∧ u | copy_ext` (lines 464–469, otherwise reject).

**Lemma (splitting preserves pairs)**: when the alignment holds, between the dim
coordinate `x ∈ [copy_start, copy_start+copy_ext)` and the split digits
`(x', x_inner)`,

```
x = (outer_start + x')·u + x_inner ,   x' ∈ [0, copy_ext/u),  x_inner ∈ [0, u) full range
```

is a bijection, and the in-region offset `x − copy_start = x'·u + x_inner` = the
row-major flat of the split iters ✓; the address is preserved term by term by the
linearity of Def 1 (the outer iter's stride = original stride·u is given by the
layout's iter structure; the inner iters cover their full ranges). That is, the split
refines a "partial interval of a single dim" into a "multi-dimensional box", exactly —
no more, no less. ∎

### L2 (smem slicing + regrouping = the common digit space and Pairs pinned down; end of P1)

Second half of `_build_l1_result` (lines 531–560):

1. `_slice_and_canonicalize_smem` (lines 484–491): T-SLICE (the region is shifted into
   the offset) + T-CANON (the flat map is unchanged). Reject on failure.
2. `extgt1_shape` = the `copy_ext` of the ext>1 iters in gmem positional order
   (lines 536–537). If all ext=1, return empty `smem_groups` directly
   (lines 539–542, the trivial branch of L8).
3. `_regroup_smem_by_extgt1_shape` (lines 494–499 + 544–547): T-GROUP cuts the smem
   canonical flat by the gmem copy shape; failure (the declared layout cannot be
   factorized by that shape) ⟹ reject.
4. Each group drops unit shards to yield a `SmemGroup` (lines 549–558).

**Lemma**: after a successful group, the common digit space is

```
n  ↔  ( x_1, …, x_M ), x_i ∈ [0, copy_ext_i)  (ext>1 iters, row-major)
x_i ↔  ( y_{i,1}, …, y_{i,q_i} ), y_{i,k} ∈ [0, e_{i,k})  (in-group shards, outer→inner)
```

and `Φ_s(n) = Σ_{i,k} y_{i,k} · S_{i,k}` (S = the declared smem strides),
`Φ_g(n) = Σ_i (copy_start_i + x_i)·stride_i + Σ_{ext=1 iters} copy_start·stride`.
Def 4's Pairs is thereby fully expressed in the planner's data structures. ∎

### L3 (contiguous chain prefix: the invariant; the smem half of P2)

`_find_contiguous_chain_prefix` (lines 568–603): flattens the shards of all groups
(group order × in-group order) and walks the chain greedily: `expected_stride` starts
at 1; at each step it finds an unconsumed shard whose `smem_stride` is provably equal
to `expected_stride`, consumes it, then `expected_stride *= extent`.

**Invariant (chain identity)**: the chain entries `chain[0..j-1]` (inner→outer) satisfy

```
stride(chain[0]) = 1 ,   stride(chain[m]) = ∏_{m' < m} extent(chain[m'])
```

— given directly by the loop construction (T-ARITH guarantees `can_prove_equal` is
sound; if it cannot be proven the chain stops early, conservative in direction). The
chain may interleave across groups (the half-major case:
`chain = [(0,0),(1,0),(3,0),(2,0)]`, host recomputation in the L5 table). An empty
chain (no stride-1 shard) ⟹ j can only take 0. ∎

### L4 (alignment + segmentation: the digit partition of each gmem iter; the gmem half of P2)

Given the chain prefix `chain[:j]`, `_distribute_selection` (lines 606–621) distributes
the selected positions per group. For each ext>1 iter (G = shape, s = stride,
copy_start, copy_ext, group with q shards, selected positions `p_0 < … < p_{j_i-1}`):

- **`_check_alignment`** (lines 624–646): `u_{p_0} = ∏_{m > p_0} e_m`, requiring
  `u_{p_0} | G` and `u_{p_0} | copy_start` (unprovable ⟹ this j is discarded).
  Note that `u_{p_0} | copy_ext` **holds automatically**
  (copy_ext = ∏ e_m = E_0·u_{p_0}).
- **`_build_segments`** (lines 649–788) produces the segments (outer→inner):
  - Segment 0 (j≥1): positions `[0, p_0]`, `DescDim(shape = G/u_{p_0}, stride = s·u_{p_0},
    box = e_{p_0}, coord_base = copy_start/u_{p_0})`;
  - Segment i (1≤i<j): positions `(p_{i-1}, p_i]`, `(shape = E_i, stride = s·u_{p_i},
    box = e_{p_i}, coord_base = 0)`;
  - Tail segment (p_last < q−1): positions `(p_last, q−1]`, `(shape = E_j, stride = s,
    box = 1, coord_base = 0)`;
  - j_i = 0: the whole axis becomes a single Case-2 segment
    `(G, s, box 1, coord copy_start)` (lines 698–716).
  - Every **unselected** position m within a segment becomes an issue axis:
    `(extent = e_m, coord_advance = u_m / u_{seg}, smem_stride = S_m)`
    (lines 681–694; `u_{seg}` is the segment's anchor `u_{p_i}`, and 1 for the tail
    segment / j=0). For m outside the anchor ⟹ `u_m = u_{seg}·∏_{m<k≤p_i} e_k` ⟹
    the divisibility is exact.

**Lemma (the gmem side matches term by term)**: the gmem contribution
`Σ_m y_m·s·u_m` of the iter coordinate digits `y_1..y_q` regroups by segments into

```
Σ_seg [ (coord_base + Σ_{m∈seg unselected} y_m·(u_m/u_seg)  +  y_{p_seg}·1 ) · (s·u_seg) ]
```

That is: the segment's DescDim coordinate = `coord_base + issue advance`, the box
digit = the selected shard's digit, and the sum of per-dim coordinate × dim stride is
exactly `Φ_g`'s term for that axis (matching AX-TILE's address formula term by term).
Segment digits do not overlap and their union covers `[0, q)` ✓ (the segmentation is a
partition of position intervals).

**Lemma (OOB exactness)**: `u_{p_0} | G` makes `x ≥ G ⟺ x' ≥ G/u_{p_0}` — the
logical out-of-bounds set agrees element by element with the hardware OOB decision
(segment 0's dim coordinate ≥ globalDim); inner / tail segment coordinates always lie
within `[0, E_i)` and produce no overrun; an ext=1 dim's coordinate = copy_start, out
of bounds ⟺ logically out of bounds ✓. If `u ∤ G`, floordiv truncation would
misclassify **in-bounds** elements near the boundary as OOB — the planner rejects via
the alignment check (this check is a load-bearing correctness condition, not mere
conservatism). ∎

### L5 (assembly + chain-DESC reorder = the placement lemma; the crux of P2)

`_assemble_plan` (lines 796–916):

1. First pass: ext=1 iters (positional order) → `DescDim(shape, stride, box=1,
   coord_base)` (lines 829–836; the degenerate placeholder iter's `(shape=1, stride=0)`
   satisfies the AX-ENC domain ✓).
2. Second pass: each group's segments produce DescDims and issue axes per L4
   (lines 838–883); selected segments record their chain indices (lines 846–852 +
   865–871; the i-th selected segment anchors `selected_positions[i]` — the segments'
   emission order is isomorphic to the selection order, correct by construction).
3. **Reorder** (lines 885–905): box=1 dims (ext1 + tail segments + j=0 segments) come
   first in construction order (outer), and box>1 dims follow in **descending chain
   index order** (inner) — i.e. `plan.dims` reversed (= descriptor dim order, dim0
   innermost) is exactly `chain[0], chain[1], …, chain[j-1], (box=1 dims...)`. The
   issue axes' `dim_idx` is remapped through `old_to_new` (lines 894–905).

**Placement lemma (the crux of P2)**: for the m-th (counting from the inside) box>1
dim, AX-FILL's fill stride

```
B_m = ∏_{k inner to it} boxDim_k = ∏_{m' < m} box_{m'} = ∏_{m' < m} extent(chain[m'])
    = stride(chain[m])                                (chain identity, L3)
```

is **exactly the declared smem stride of that chain shard**; box=1 dims have no fill
footprint (a single coordinate) and all sit on the outside, entering no B_m product.
Hence within a single instruction, the element for the box digits `(y_{chain[m]})_m`
lands at linear offset `Σ_m y·S_{chain[m]}` = the declared layout's address for those
digits ✓. Combined with L4 (the gmem side): **the single-transaction pair set = the
restriction of Pairs to the subspace "issue digits fixed, selected digits free"**. ∎

**Uniqueness note** (the basis of the error message "the only placement-correct
order"): given the same set of box>1 dims (all boxes >1), AX-FILL's fill-stride
sequence `1, b_0, b_0·b_1, …` is strictly increasing and uniquely determined by the dim
order; the declared strides are exactly this family of products ⟹ any other
permutation would assign some dim a fill stride ≠ its declared stride ⟹ chain-DESC is
the only placement-correct dim order. The removed "natural" order (gmem group
positional order) is correct exactly when it coincidentally agrees with the chain
order — under the half-major declaration the two diverge, and the GPU test
(§2.3 evidence 2) pins the divergence down as an error on the natural side.

**Host-side mechanical recomputation** (a pure-host probe driving the real planner
against the cases below):

| case | chain (inner→outer) | per-dim `fill_stride == declared_stride` | P3 enumeration |
|---|---|---|---|
| interleaved `(1,64,4096,8192)` | `[(0,0),(1,0),(2,0),(3,0)]` | `1/64/4096/8192` all equal ✓ | 32768 pairs, exactly = the declared address set ✓ |
| half-major `(1,64,16384,4096)` | `[(0,0),(1,0),(3,0),(2,0)]` | `1/64/4096/16384` all equal ✓ | 32768 pairs ✓ |
| stride-gap `(1024,64,1)` (with an issue axis) | `[(2,0),(1,0)]` | `1/64` ✓ | 2048 `(smem,gmem)` pairs, each = Pairs ✓ |
| rope-fold (decline case) | — | `DispatchFail: ... unit stride; got 576` ✓ | — |

(The encoded values also have host pins: the two `encode_head` goldens of
`_FLASHMLA_Q_SMEM_CASES`, test_tma.py lines 2240–2288.) ∎

### L6 (semantic preservation of merge + promote; P2's finishing transformation)

`_merge_contig_full_box_dims` (lines 935–1119) is activated only when
`_plan_needs_alignment_fix` (lines 919–932: some non-innermost dim's byte stride ∤ 16)
holds (lines 981–982 — an already-compliant plan is returned untouched, so golden
shapes are not perturbed).

- **merge** (`try_merge_at`, lines 998–1024): merging an adjacent `(outer, inner)` pair
  requires ① no issue-axis binding, ② both `coord_base = 0`, ③ both `box = shape`
  (full box), ④ `outer.stride = inner.shape·inner.stride` (gmem physically
  contiguous), ⑤ merged box ≤ 256.
  **Pair-preservation proof**: the merged digit is `y = y_out·inner.shape + y_in`.
  gmem: `y·inner.stride = y_out·outer.stride + y_in·inner.stride` ✓ (condition ④).
  smem (AX-FILL): the merged dim's fill stride = the original inner's B;
  `y·B = y_in·B + y_out·(inner.box·B)` = the contributions of the two original dims ✓
  (inner full-box ⟹ inner.box = inner.shape). Moreover, by L3/L5, adjacent chain dims
  satisfy `S(chain[m+1]) = extent(chain[m])·S(chain[m])`, cohering with the ④ /
  full-box conditions — after merging, the placement lemma still holds dim by dim.
  Coordinates identically 0 (conditions ②+①) ⟹ no coordinate re-encoding issue. ∎
- **promote** (`try_promote`, lines 1028–1087): `elem_bytes` doubles
  (uint8→16→32→64); the innermost dim's shape/box/coord_base halve, non-innermost
  strides halve, and the coord_advance of issue axes bound to the innermost dim halves
  (that branch is in fact unreachable — having any issue axis already rejects promote,
  lines 1034–1035; defensive redundancy). **Byte equivalence**: the per-dim byte
  contributions `coord·stride·elem_bytes` and the box byte span are unchanged —
  provided the divided quantities are even. The code checks `inner.shape % 2`
  (line 1039), non-innermost `stride % 2` (lines 1050–1052), and `inner.box % 2` /
  `inner.coord_base % 2`
  (lines 1046–1049, with the comment at lines 1041–1045: an odd box would silently
  under-copy, an odd coord_base would misalign by one original element; if evenness
  cannot be proven, promote is abandoned and left to the shrink/variant fallback).
  Pinned by `test_copy_tma_merge_promote_positive_pin` (test_tma.py
  lines 2407–2426), `test_copy_tma_promote_declines_odd_box_and_coord`
  (lines 2429–2472), and `test_copy_tma_declines_odd_box_promotion_end_to_end`
  (lines 2475–2489); Theorem 1 therefore needs no separate evenness premise.
- The loop (lines 1089–1110): greedy merge to a fixed point; when a box blocks, try
  promote and retry. Termination: merge reduces the dim count, promote strictly raises
  elem_bytes (≤ 8). ∎

### L7 (hardware constraint validation ⟺ AX-ENC preconditions; rejection direction)

`_validate_hw_constraints(plan)` (lines 1122–1157; all checks are in units of
`plan.elem_dtype / plan.elem_bytes`, so they stay consistent after promote; pinned by
`test_copy_tma_validate_hw_constraints_uses_promoted_dtype`, test_tma.py
lines 2492–2527):

- rank ∈ [1, 5] (lines 1132–1135) ↔ AX-ENC.2.
- **Innermost dim stride provably = 1** (lines 1138–1140) ↔ AX-ENC.1's implicit dim0
  stride. This is precisely the trigger of the rope-fold decline (last row of the L5
  table; test lines 2133–2182). If the smem shard declared innermost-contiguous
  corresponds to non-contiguous gmem elements, no j can produce a unit-stride innermost
  dim — the planner cannot express that placement with a dense tile, and rejects
  loudly.
- `_swizzle_inner_box_fits` (lines 299–304 + 1143–1144): `boxDim[0] ≤` the swizzle atom
  row width (in elements, `tma_atom_shape`) ↔ AX-ENC.2's swizzle-span constraint. The
  atom row width is computed from `plan.elem_dtype`, keeping the check in promoted
  units.
- **Post-merge 16B stride recheck** (lines 1151–1155): if
  `_plan_needs_alignment_fix` still holds after merge+promote (e.g. some partially
  boxed dims cannot merge) ⟹ reject right at dispatch (allowing shrink / other-variant
  fallback) rather than leaving it to the wrapper ICHECK to fail late; pinned by
  `test_copy_tma_declines_unfixable_alignment_at_dispatch` (test_tma.py
  lines 2553–2568).

**Still backstopped only by wrapper ICHECKs** (fail-late): box ≤ 256,
globalDim ≤ 2^32, globalStrides < 2^40 and the other AX-ENC domain bounds (all loud,
never silent). ∎

### L8 (the shrink iteration = a semantics-preserving search; P3 precondition)

`_build_plan_with_shrink` (lines 1160–1200):

- Empty `smem_groups` (all ext=1): a trivial plan (all box=1) + validation, reject on
  failure (lines 1171–1176).
- Otherwise j goes from `len(chain)` down to 0: for each j, first check
  `_check_alignment` iter by iter (lines 1183–1192, failure moves on to the next j),
  then `_assemble_plan` + `_validate_hw_constraints` (lines 1194–1197), returning on a
  full pass; if j=0 still fails ⟹
  `fail("TMA plan: all chain prefix lengths rejected; last reason: …")` (line 1200).

**Lemma (j-independence)**: the constructions and proofs of L4/L5 hold for **any** j
that passes the alignment check — j only decides which shards are covered by the box
(within a single instruction) and which by the issue loop (across instructions); the
two coverages produce the same pair set (L9). Hence shrink is not a "degrading
approximation" but a search, within a family of equi-semantic plans, for the largest
hardware-encodable box. ∎

### L9 (the emission layer = P3 tiling; exactly once)

Emission (lines 1322–1512):

1. `flat_total_extent = ∏ issue extents` (lines 193–197 + 1328);
   `offsets_and_coords` (lines 199–217): `iter_val_a = ⌊flat / cum_a⌋ mod extent_a`
   (cum = product of the inner axes) — the standard mixed-radix **bijection**;
   `s_offset = Σ iter_val·smem_stride`, `coords[dim] += Σ iter_val·coord_advance`
   (summing multiple axes on the same dim = the mixed-radix expansion of that dim's
   issue digit; L4's divisibility guarantees the coefficients are exact).
   `_simplify_with_var_ranges` (lines 312–320) only simplifies (T-ARITH
   value-preserving).
2. Each iteration: `s_buf_w_offset = decl_buffer(..., elem_offset = s_buf.elem_offset +
   s_offset, layout = tile part)` (lines 1500–1507),
   `ptr_to(s_st)` = the region start's **linear** address + s_offset (AX-EXEC) —
   consistent with Def 2's linearization convention: the hardware applies the swizzle
   itself (AX-FILL), the declared layout's physical address = `Swz(linear)`, both sides
   use the same `Swz` (same `swizzle_len`, same inner direction — the latter
   checked by tma_utils.py lines 126–134; the structural identity of
   `per_element/atom_len` remains a trust convention, §5 item 4) and the same anchor
   (WF-ALIGN) ⟹ verifying in the linear space suffices (the same lemma as cp proof L2).
3. Coordinate order: `compute_offsets_and_tma_coords` returns `reversed(coords)`
   (line 1338, outer→inner → inner→outer); the host encoding likewise `reversed`
   (lines 1525–1527) — **double reversal consistent**, PTX coordinate c_0 and
   descriptor dim0 both refer to the innermost dim ✓.
4. **P3 lemma**: fixing the issue digits = one instruction = one slice of Pairs (L5);
   the issue enumeration bijectively traverses the unselected digit space ⟹ the
   union = Pairs, and the slices are pairwise disjoint (digit combinations unique +
   WF-DECL injectivity ⟹ destination addresses never collide) ⟹ every destination is
   written exactly once. The emission order has no semantic effect (the instructions
   are mutually independent; completion is aggregated by the mbarrier).
5. **gather4** (lines 1345–1360 + 1466–1501): one chunk per 4 indexer rows; a chunk's
   smem start = `s_st + chunk·4 @ dst_gather_axis` through the declared layout's
   `ptr_to` (lines 1345–1348) — chunk-granular placement follows the declared layout ✓;
   the 4 rows within a chunk are written contiguously by AX-GATHER4 (row-pitch
   contract, §2.4); coordinates = `[c_inner, r_0..r_3]` (lines 1350–1360; rank≠2 /
   coordinate count≠2 rejected, lines 1352–1353 / 1358–1359). When
   `flat_total_extent > 1` the chunk is the outermost level and each chunk body is an
   issue loop over the remaining iters (lines 1490–1501; a single chunk uses its body
   directly — a length-1 SeqStmt is invalid IR); the emission order does not affect
   the set semantics. Pinned by `test_copy_tma_gather4_multi_iter_gpu_smoke`
   (single-chunk multi-iter roundtrip) alongside the single-tile smoke.
6. **mbarrier / multicast operands** (lines 1357–1459): with `cta_group=2` or a
   static/predicated `mbarrier_addr`, convert the shared address; s2g goes through the
   `s2g` / `s2g_reduce` builtins (lines 1437–1459). All are operand packaging that does
   not touch the pair set (within AX-MC / AX-TILE scope). ∎

### L10 (host encoding + tensormap cache)

Lines 1390–1542:

- **cache key** (lines 1399–1407): `hash(tensor_ptr) : g_buf.dtype :
  plan.elem_dtype : rank : shape : strides(bytes, innermost dropped) : box : swizzle :
  oob_fill : force_cu_dtype : l2`. The fields in the key = the actual argument set of
  the encode call minus the constant entries (interleave=0, element_strides=1) —
  **`plan.elem_dtype` is in the key**, ruling out promotion collisions in which the
  numeric fields (shape/byte-stride/box) of two copies of the same buffer coincide
  while their promotion levels differ (the comment at lines 1394–1398 records the
  collision construction; pinned by
  `test_copy_tma_tensormap_cache_key_includes_promotion_dtype`, test_tma.py
  lines 2571–2618). `hash(plan.tensor_ptr)` is an object hash — always identical for
  the same buffer var; cross-var hash collisions are assumed negligible. A hit ⟹
  reuse the already-encoded var (lines 1414–1417); the
  `external tensor_map` config bypasses directly (lines 1409–1412, at the caller's own
  risk, scope note §2.5).
- **Encode emission** (lines 1514–1542): `tvm_stack_alloca("tensormap")` + call_packed
  with the arguments in AX-ENC order (`*reversed(shape)`, `*reversed(strides)`,
  `*reversed(box)`, `element_strides`, interleave 0, swizzle, l2, oob, optional
  `force_cu_dtype` — appended only when requested, keeping the default path
  byte-stable, lines 1529–1536); attached to host init and registered in the cache
  (lines 1541–1542). Structural golden: `_build_expected_host_init` (test_tma.py
  lines 232–261) compares every case IntImm by IntImm.
- **prefetch** (lines 1544–1560): requires the `warp_id_in_cta` launch param (otherwise
  reject, lines 1545–1546); `elect_sync` has a single lane issue
  `prefetch_tensormap`; a performance-only statement that does not change the pair
  set. ∎

### L11 (predicate and execution domain)

`register_dispatch` (lines 1610–1630, variant="tma", priority=10):

- `_validate_tma_copy_op` (lines 1565–1605): layouts exist on both sides, dtypes equal,
  scope envelope, (after the gather correction) total element counts provably equal —
  a failure is a **predicate failure** (silently yields to other variants, not an
  exception).
- `single_thread` (exec_scope_utils): single-thread execution domain ⟹ the
  instruction sequence is issued exactly once (the execution-model half of
  "exactly once"). ∎

---

## 4. Theorems

### Theorem 1 (acceptance ⟹ Pairs equivalence; unconditional on the acceptance domain)

Suppose a `Tx.copy_async(..., dispatch="tma")` call satisfies:

1. `copy_tma_impl` returns `impl` without exception (acceptance);
2. the well-formedness assumptions WF-DECL / WF-REGION / WF-ALIGN (§1.4);
3. the trust base holds (§5), in particular: AX-FILL, AX-ENC, AX-EXEC.

Then the side-effect semantics (Def 5) of `impl` + host init is exactly `Pairs`: under
g2s, each `(p, a) ∈ Pairs` has its destination `p` written exactly once, with value =
the element at address `a` (or the `oob` fill value when logically out of bounds);
there are no other smem writes. Dually for s2g (out-of-bounds writes suppressed;
`use_tma_reduce` generalizes with the reduction semantics, scope note §2.5).

**Note (no side premises)**: condition 3 carries no promote-evenness side premise —
evenness of every divided quantity is unconditionally guaranteed by `try_promote`'s
runtime checks (lines 1046–1049, L6). Likewise, the no-collision cache-key convention
holds by construction (the key includes `plan.elem_dtype`, L10).

**Proof**: L0 (config normalization and scoping) → L1/L2 (P1: both sides reduced to
the common mixed-radix digit space, Pairs expressed in the planner's data structures)
→ L3/L4/L5 (P2: chain identity + segment decomposition + chain-DESC reorder ⟹ the
single-transaction pair set = a slice of Pairs; AX-FILL is the hardware half) →
L6 (merge/promote preserve pairs) → L7 (the AX-ENC preconditions hold ⟹ the host
encoding is executable and faithful) → L8 (the j search preserves semantics) →
L9 (P3: the issue enumeration is bijective, the slices are mutually exclusive, the
union is complete) → L10 (the encode arguments correspond to the plan item by item;
the cache key has no false sharing) → L11 (exactly-once execution).

**Emphasis (unconditionality)**: unlike before the removal, this theorem contains no
conditional branch on "the value of `tensor_map_dim_order`" — every acceptance within
the acceptance domain takes the same chain-DESC path, and that path is proven by the
placement lemma (L5) + the uniqueness note to be the **only** dim order consistent with
the declared layout. ∎

**Empirical note**: the end-to-end composition is verified field-by-field /
element-by-element by the following tests — the structural golden matrix
`test_copy_tma_codegen` (test_tma.py lines 1103–1142; case table lines 495–1100:
2D–5D, swizzle 0–3, int8/uint8/fp8/bf16/fp16/fp32, partial/offset/transpose/
multiphase/atom/non-prefix chain, the s2g series, the three oob states and two classes
of oob negative cases); GPU smoke g2s/s2g (lines 1657/1786), symbolic dimensions
(1403–1407), 3D view (1501–1505), gather4 (1282–1401), uint32 shape/base
(1924/1957), dynamic cache hint / cta_group2 codegen (1990–2131); the two folded-view
cases with host pin + GPU element-level (2268–2366); the merge+promote positive pin
and the rejection matrix (2407–2618); and the raw-smem probe of §2.3.

### Theorem 2 (rejection safety)

When `copy_tma_impl` or any layout operator it calls throws (`DispatchFail` /
`ValueError` / ICHECK), this variant produces no IR: the dispatcher catches it
(dispatcher.py lines 304–321) and continues trying other variants, aggregating an error
if all fail (lines 324–329); a predicate failure (L11) likewise skips silently. When
`dispatch="tma"` pins the variant, a rejection becomes a direct compile error (the
decline test and the removed-knob test take exactly this path). There is no path that
"accepts a wrong combination and emits wrong IR"; the residual fail-late surface is
only the wrapper-ICHECK-backstopped AX-ENC domain bounds (end of L7, all loud). ∎

### Completeness discussion (conservative rather than incorrect)

All of the following are cases of **rejecting legal or potentially legal copies** (the
safe direction):

- **Placement not chain-derivable** (the pinned completeness boundary): the
  hand-written rope-fold layout `Compose(Swz(3,2,3), (64,2,32):(32,2048,1))` backing
  the permuted copy `global(32,64,1) → shared(64,32)` — the smem stride-1 chain
  corresponds to non-contiguous gmem elements, and j=2/1/0 are all rejected with
  "unit stride" (`test_copy_tma_declines_non_derivable_fold_layout`,
  lines 2133–2182). This case used to be accepted under the old "natural" mode
  (encoded by gmem positional order, relying on an external ABI coincidence); after the
  removal it becomes a **loud completeness limitation**: either change the declared
  layout so the chain becomes derivable, or use a non-TMA variant. This is a
  reject-safe boundary, not a bug.
- Single-element / all-ext=1 copies whose innermost gmem iter has non-unit stride
  (L8's trivial branch + L7).
- Multi-iter gmem groups with `u ∤ copy_start / copy_ext` (L1), segment 0's
  `u_{p_0} ∤ G / copy_start` (L4) — the chain auto-shrinks; if j=0 still fails the
  unit-stride condition, reject.
- The declared smem layout cannot be grouped by the gmem copy shape (L2);
  undecidable swizzle mode / flipped `swizzle_inner` (L0/L1); symbolic stride/extent
  equalities unprovable (T-ARITH conservative, chain early stop or alignment failure).
- Promote preconditions unprovable (shape/box/coord_base/stride evenness, L6) — the
  rewrite is abandoned and handed to shrink / other variants.
- gather4's narrow domain (axis=0, rank2, len%4, absolute coordinates) (L0/L9).

---

## 5. Trust base (summary)

1. **Layout algebra**: T-SLICE / T-APPLY / T-CANON / T-GROUP / T-ARITH
   (cp proof §2.1/§5, implemented as library-wide shared infrastructure).
2. **Driver / PTX axioms**: AX-ENC (cuTensorMapEncodeTiled field semantics + wrapper
   upfront validation, cuda_device_api.cc lines 400–772); AX-TILE (PTX §9.7.9.24.9
   tile semantics + OOB fill/suppression); **AX-FILL (box linear fill order + swizzle
   periodic permutation within 16B blocks) — where the spec text is underspecified,
   the B200 raw-smem probe (§2.3) and the GPU element-level test
   (test_tma.py lines 2291–2366) are authoritative**; AX-GATHER4; AX-MC (single-CTA
   projection).
3. **codegen**: AX-EXEC (unroll/meta_var/decl_buffer+elem_offset/ptr_to/
   PTX-builtin per-argument mapping), pinned by the structural goldens
   (the full `_build_expected_impl` matrix) and codegen string spot checks.
4. **Conventions**: canonicity of the swizzle family's `per_element/atom_len`
   (structural identity trusted, not compared; the `swizzle_inner` direction is
   checked, tma_utils.py lines 126–134) + the smem base's swizzle period alignment
   (WF-ALIGN, an allocator contract); the gather4 row-pitch contract (§2.4). Promote
   evenness and cache-key completeness are enforced by runtime checks (L6, L10);
   unit-consistent validation and the post-merge stride recheck reject at dispatch
   (L7), leaving only the wrapper-ICHECK-backstopped AX-ENC domain bounds as the
   fail-late surface (end of L7, all loud).
5. **Caller contracts**: WF-DECL (declared layout truthfulness — where the small_topk
   incident lives; not falsifiable by dispatch), WF-REGION, mbarrier completion
   synchronization, the self-assumed semantics of external tensor_map / multicast /
   reduce.
6. **Empirical anchors**: the B200 probe + the GPU test matrix (Theorem 1's empirical
   note); the host-side pure-Python mechanical recomputation (a pure-host probe
   driving the real planner: crux per-dim equalities + full P3 enumeration + the
   decline negative case, L5 table).

---

## Appendix A: code checkpoint ↔ lemma cross-reference table

Every explicit rejection / validation point in `tma.py` and its attribution
(line numbers refer to the current version):

| Line | Check | Lemma |
|---|---|---|
| 238–241 | non-pure-memory axes (shared/global) | L1/L2 precondition |
| 253 | unknown oob mode | L0 |
| 255 | oob='nan' with non-float dtype | L0 (AX-ENC.2 precondition) |
| 264 | oob fallback raise (unreachable) | L0 (defensive) |
| 275 / 285–288 | l2_promotion domain | L0 |
| 332 | indexer type | L0 |
| 355–356 | gather not g2s | L0 |
| 357–361 | gather_axis missing / ≠0 | L0 |
| 362–363 | indexer length ∤ 4 | L0 |
| 365–371 | dst gather extent not unique | L0 |
| 373–377 | source gather axis min ≠ 0 | L0 |
| 409 | gmem not a TileLayout | L1 |
| 422 | gmem group-by-buffer-shape failure | L1 |
| 464–469 | multi-iter group `u ∤ copy_start / copy_ext` | **L1 (load-bearing)** |
| 490 | smem slice failure | L2 |
| 511–512 | swizzle mode undecidable | L0/L2 |
| tma_utils.py 126–134 | `swizzle_inner=False` flipped-direction rejection | L1/L9 |
| 546 | smem regroup by ext>1 shape failure | L2 |
| 642–645 | `u_{p_0} ∤ G / copy_start` (alignment) | **L4 (load-bearing, incl. OOB exactness)** |
| 926–932 | `_plan_needs_alignment_fix` determination (gating, not rejection) | L6 |
| 1000–1013 | the five merge conditions (skip if unsatisfied) | L6 |
| 1034–1052 | promote preconditions (no issue axis / unit stride / shape even / box even / coord_base even / stride even) | L6 |
| 1132–1135 | rank ∈ [1,5] | L7 |
| 1138–1140 | **innermost-dim unit stride** (the trigger of the decline boundary) | L7 |
| 1143–1144 | innermost box exceeds swizzle atom (computed with `plan.elem_dtype`) | L7 |
| 1151–1155 | post-merge 16B stride recheck | L7 |
| 1171–1176 | all-ext=1 plan validation failure | L8 |
| 1200 | all chain prefixes rejected (aggregated reason) | L8 |
| 1229–1231 | illegal scope combination | L0 |
| **1251–1256** | **`tensor_map_dim_order` removed ("was removed" rejection)** | **L0 (loudness note)** |
| 1271–1275 | oob='nan' recheck against `plan.elem_dtype` | L0 |
| 1285–1286 | unknown tma_dtype | L0 |
| 1287–1288 | tma_dtype on a non-float32-family descriptor (per `plan.elem_dtype`) | L0 |
| 1299 | cta_mask not g2s (assert) | L0 |
| 1306 | g2s missing mbar | L0 |
| 1317 / 1319 | mbarrier_addr type / direction | L0 |
| 1347–1348 / 1353–1354 | gather4 rank ≠ 2 / coordinate count ≠ 2 | L9 |
| 1546 | prefetch missing warp_id_in_cta | L10 |
| 1580–1605 | predicate: layout/dtype/scope/total element count | L11 |
| wrapper 431–724 | AX-ENC domain ICHECKs (residual fail-late backstop) | L7/L10 |

Emission side (no rejection, construction only): 568–621 (chain + distribution) → L3;
649–788 (segmentation) → L4; 796–916 (assembly + chain-DESC reorder) → **L5 (crux)**;
935–1119 (merge/promote) → L6; 1160–1200 (shrink) → L8; 199–217 + 1322–1512
(issue loop / gather4 / mbarrier operands) → L9; 1390–1560 (cache key / host encoding /
prefetch, key includes `plan.elem_dtype`) → L10.

Cross-reference test anchors (test_tma.py): structural golden driver 1103–1142
(case table 495–1100); decline boundary 2133–2182; flipped swizzle_inner 2185–2211;
removed-knob 2214–2237; folded-view host pin 2268–2288; folded-view GPU element-level
2291–2366; merge/promote / validation-unit / cache-key regressions 2407–2618;
gather4 1282–1401; GPU smoke 1657/1786; symbolic / uint32 1403/1924/1957.
