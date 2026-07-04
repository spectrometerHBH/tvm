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

# IR Equivalence Proof of the Generic tcgen05.cp Dispatch

Subject: `python/tvm/backend/cuda/operator/tile_primitive/copy_async/tcgen05_cp.py`
(hereafter the **planner**; all line numbers refer to the current version of that file
unless another file name is noted).

Proposition (informal): for every `(tensor shape, layout, region)` combination accepted
by the planner, the IR emitted by `copy_smem_tmem_impl`, under the instruction axioms and
the trust base (§2, §5), implements exactly the logical copy semantics of
`Tx.copy_async(t_region, s_region)`; for rejected combinations, the planner raises
`ValueError` (or another exception), the dispatcher falls back to other variants or fails
as a whole, and never emits incorrect IR.

Suggested reading order: §0 overview → §1 definitions → §2 axioms and trusted lemmas →
§3 step-by-step lemmas → §4 theorems → §5 trust base → Appendix A (full table of
code checkpoints ↔ lemmas).

---

## 0. Overview: the general form of a tile-primitive dispatch

### 0.1 Problem shape

The input to a tile-primitive dispatch is one logical operation

```
Op(dst_region, src_region, config)
```

where the two regions each carry a free triple `(buffer shape, layout, region)`: the
buffer shape is arbitrary, the layout is an arbitrary (well-formed)
`TileLayout / SwizzleLayout / ComposeLayout`, and the region is an arbitrary
sub-rectangle. The output of the dispatch is a piece of IR composed of fixed-shape
hardware instructions. The semantics of the logical operation is a **pairing set**
(which physical destination location receives the value from which physical source
location); the semantics of each hardware instruction is also a pairing set (given by
the instruction axioms). Hence:

> **Correctness criterion**: the union of the pairing sets of the emitted IR = the
> pairing set of the logical operation (as an equality of sets of
> "location ← value source", with every destination location written exactly once).

### 0.2 The generic three-part decomposition

Any proof of this kind can be split into three parts:

- **P1 (re-indexing)**: apply pairing-set-preserving transformations synchronously to
  the dst/src layouts — slicing, canonicalization, dimension permutation, elimination of
  degenerate (stride-0 / extent-1) dimensions — to bring both sides into a **common
  digit space**. Each step is a bijective re-indexing or an idempotent deduplication of
  the pairing set; the pairing set is unchanged.
- **P2 (atom)**: carve out a subspace of the common digit space (here the `(lane, col)`
  digit groups), and prove that the hardware pairing set of a **single instruction**
  with given parameters (descriptor fields, taddr column offset) equals exactly the
  restriction of the two layouts to that subspace. This is the technical core of the
  proof: matching the layout's stride structure term by term against the address-walk
  formulas of the instruction axioms, and **solving** for the instruction parameters
  from the match.
- **P3 (tiling)**: enumerate the remaining digits (middle) one by one; each value yields
  one instruction with affinely translated parameters; prove that the enumeration is a
  bijection (exactly once), that the destination sets of distinct instructions are
  pairwise disjoint, and that their union is exactly the full pairing set.

The planner's steps A–I correspond to the three parts as: A/C/D/E ↔ P1, F (grouping +
validation) and G (alignment) ↔ P2, F.5/H/I ↔ P3. §3 unfolds along this outline.

### 0.3 Safety criterion

Beyond equivalence we also require: **the acceptance decision and the emission share the
same code path** (`_build_plan`, lines 303–636); any failed precondition leaves via an
exception, the dispatcher (`python/tvm/tirx/operator/tile_primitive/dispatcher.py`,
lines 298–329) catches it and keeps trying other variants, and if all fail it raises a
`RuntimeError` with an aggregated report. Hence there is no "half-accepted" state:
either the IR is emitted and (Theorem 1) correct, or nothing is emitted.

---

## 1. Definitions

### 1.1 Notation and index spaces

- region `R = [(b_1, b_1+e_1), …, (b_K, b_K+e_K)]`, index space `I(R) = ∏_k [0, e_k)`,
  `N = ∏_k e_k`.
- row-major flattening `⌊i⌋ ∈ [0, N)`: the last dimension is fastest. The inverse of the
  multi-dim → one-dim flattening is `SplitCoord`
  (`layout.py` lines 45–63; C++ mirror `src/tirx/ir/layout/utils.cc::SplitCoord`):
  the last extent takes `%` first, and the first takes the remaining `//`.
- dtype bit width `w = dtype_bits`; `ε₃₂ = 32/w` (elements per 32-bit unit, line 331),
  `ε₁₂₈ = 128/w` (elements per 16B unit, line 330), `epa = atom_bits/w` (line 332).

### 1.2 TileLayout semantics (definition anchor)

`TileLayout L = (shard, replica, offset)`, `shard = [(E_1,S_1,A_1), …, (E_p,S_p,A_p)]`
(extent, stride, axis), `replica = [(F_1,T_1,A'_1), …]`, `offset = {A ↦ O_A}`.

**Def 1 (apply)**: for a multi-dimensional digit tuple `d = (d_1,…,d_p) ∈ ∏[0,E_i)`,

```
⟦L⟧(d)_A = O_A + Σ_{i: A_i = A} d_i · S_i        (summed separately per axis)
```

This is the direct semantics of `TileLayoutNode::Apply(Array<PrimExpr>)`
(`src/tirx/ir/layout/tile_core.cc` lines 187–210). For a one-dimensional (flat) index
`n ∈ [0, ∏E_i)`, first obtain `d` via `SplitCoord(n, [E_i])`, then apply
(tile_core.cc lines 146–148). Write the flat form as `Φ_L(n)`.

**Def 2 (replica semantics / multicast set)**:

```
Rep(L) = { Σ_j c_j · T_j on axis A'_j  :  c_j ∈ [0, F_j) }      (Minkowski sum)
```

For a destination layout, the set of physical coordinates occupied by logical element
`n` is `⟦L⟧_rep(n) = { Φ_L(n) ⊕ δ : δ ∈ Rep(L) }`. This convention is adopted directly
by the tests' host-side expected-value reconstruction
(`tests/python/tirx/operator/tile_primitive/cuda/copy_async/test_tcgen05_cp_shapes.py`
lines 193–208: `rep_offs` expansion + `t_full.apply` + masked comparison), which serves
as this document's semantic anchor for replica.

**Def 3 (physicalization)**:
- smem: the single memory axis `m`; value = linear address in element units (relative to
  the buffer base). If the layout is `ComposeLayout(Swz, T)`, the physical element
  address = `Swz(Φ_T(n)_m)` (see §1.3).
- tmem: axes `TLane` (rows 0–127) and `TCol` (column coordinate in **element** units).
  Hardware 32-bit unit column `c₃₂ = TCol ÷ ε₃₂`, intra-unit bit offset
  `(TCol mod ε₃₂)·w` (low element in low bits; the `val << (sub * bits)` at test
  lines 206–207 is exactly this convention). The uint32 encoding of taddr:
  bits 16–31 = lane, bits 0–15 = 32-bit column
  (`intrinsics/header.py` lines 605–618, `get_tmem_addr`).

### 1.3 Swizzle / Compose semantics

`SwizzleLayout(per_element = pe, swizzle_len = s, atom_len = a, swizzle_inner = true)`
acts on a **one-dimensional element index** x as
(`src/tirx/ir/layout/swizzle_layout.cc` lines 69–84):

```
u  = x >> pe                      (16B unit index, when pe = log2(ε₁₂₈))
u' = u XOR ((u >> a) & (2^s − 1)) (inner mode; outer_mask = ((1<<s)−1) << a)
Swz(x) = (u' << pe) + (x mod 2^pe)
```

That is: blocks of `2^pe` elements (16B units) are indivisible, and the low `s` bits of
the block index are XORed with the bit field `[a, a+s)`. `mma_atom_layout`
(`python/tvm/backend/cuda/operator/tile_primitive/tma_utils.py` lines 37–44) fixes
`pe = log2(128/w)`, `a = 3`, `s = swizzle mode`, which is exactly the PTX-spec-level
`Swizzle<s, 4, 3>` (in byte space, bit field `[7, 7+s)` is XORed into `[4, 4+s)`;
PTX ISA 8.8 §9.7.16.3.3, table "Canonical Layouts", printed page p630). `Swz` is a
self-inverse permutation within each aligned window of `8·2^s` 16B units
(= `2^(7+s)` bytes = one swizzle repeat period), and never crosses a 16B unit boundary.

`ComposeLayout(Swz, T)`: `Apply = Swz ∘ Φ_T` (`compose_layout.cc` lines 67–74); `Slice`
slices only the `T` part and re-wraps (lines 108–115); `Canonicalize` canonicalizes `T`,
and if it degenerates to trivial only `Swz` remains (lines 76–82 — this is exactly the
origin of the bare-swizzle case handled by the planner at lines 349–357).

### 1.4 Logical copy semantics and well-formedness assumptions

**Def 4 (Pairs)**: let `L_t = t_buf.layout.slice(t_shape, t_region).canonicalize()` and
`L_s` (likewise; for the swizzle case see the linearization convention in §3 L2); both
are maps on `[0, N)` (that N is equal on the two sides after slicing is given by op
legality; if unequal, subsequent checks necessarily fail, see L6/L11).

```
Pairs = { (Φ_{L_t}(n) ⊕ δ ,  Φ_{L_s}(n))  :  n ∈ [0, N),  δ ∈ Rep(L_t) }
```

I.e. pairing one by one along the region's row-major flat index — this is exactly the
reduction performed by `zip(t_coords, s_coords)` in the test's `_expected_readback`
(lines 196–199), and it is the operational contract of `Tx.copy_async`.

**Well-formedness assumptions**:
- **WF-dst (injectivity of the unicast part)**: the map `(n, δ) ↦ Φ_{L_t}(n) ⊕ δ` is
  injective, or weakens to "if `Φ_{L_t}(n) ⊕ δ = Φ_{L_t}(n') ⊕ δ'` then
  `Φ_{L_s}(n) = Φ_{L_s}(n')`" (the value received by any single destination cell is
  unique). **The planner does not check this assumption**; calls that
  violate it have no well-defined semantics of their own and are excluded from the
  theorem's scope.
- **WF-src (totality)**: `Φ_{L_s}` is defined everywhere on `[0,N)` and lands within the
  buffer allocation (a layout is a total function, so this holds automatically;
  out-of-bounds belongs to the upstream region-legality contract). The source side is
  **not** required to be injective: reading the same source element multiple times
  (broadcast to distinct destination cells) is semantically harmless.

**Def 5 (side-effect semantics and equivalence)**: the effect of the emitted IR = the
union of the "write (location, source address)" sets produced by each `tcgen05.cp` in it
according to the instruction axioms. **Equivalence** := that union = `Pairs`, with each
destination location written exactly once (under WF-dst this follows from pairwise
disjointness, see L12).

**Asynchrony contract (explicit exclusion)**: `tcgen05.cp` is inherently asynchronous;
the planner only emits the cp loop, and the completion signal (`tcgen05.commit` against
an mbarrier) is the caller's responsibility — the file docstring, lines 20–23, states
this explicitly. Hence "when the writes become observable" is not part of the
equivalence proposition; the proposition speaks only of "the finally written pairing
set". Likewise, the peer-CTA replication of `cta_group::2` lies outside the layout
algebra (the planner passes `cta_group` through unchanged, line 826; PTX specifies
that the two CTAs receive the same data); the theorem's scope is `cta_group::1` (or
the single-CTA projection of ::2).

---

## 2. Axioms and trusted lemmas

We first draw the line between "what is proven" and "what is trusted". Items prefixed
`T-` are layout-algebra facts: of these, T-CANON / T-GROUP / T-PERM come with proof
sketches (mechanically verifiable directly from the cited code), while
T-SLICE / T-APPLY / arith are fully trusted. Items prefixed `AX-` are hardware
instruction / codegen axioms, grounded in the PTX ISA 8.8 text plus B200 measurements
(`test_tcgen05_cp_shapes.py`).

### 2.1 Layout algebra

**T-APPLY (trusted, definition anchor)**: `TileLayout.apply` / the FFI reflection
implements the semantics of Def 1 (tile_core.cc lines 146–210; the group-first path of
`apply(coord, shape)` at lines 150–185 is numerically equivalent to the flatten+split
path — self-attested by code comments, trusted).

**T-SLICE (trusted)**: if `L.slice(shape, region)` returns non-None, then for all
`i ∈ I(R)`:

```
⟦slice(L)⟧(i) = ⟦L⟧(i + R.min)      (per-dimension translation; offset absorbs the contribution of R.min)
```

and replica is preserved as-is. Implementation in `src/tirx/ir/layout/tile_slice.cc`:
`Slice` (lines 144–172) groups by region dimension and calls `SlicePerGroup` per group
(lines 29–142). There, decomposing `begin` into each iter's initial value `d0[k]` and
accumulating it into the offset (lines 59–80) is a direct mixed-radix translation; the
single-shard special case (lines 86–90), the per-iter peeling (lines 92–107), and the
2-split rewrite that crosses a boundary once (lines 121–139, constructing a delta
stride) all maintain the invariant "the index map within the region is unchanged". This
document takes the whole of T-SLICE as a trusted lemma (it is infrastructure shared
library-wide, and is indirectly corroborated by the same test suite via host-side
`apply` expected values). A failed slice returns None (on the planner side this leaves
via an `AttributeError: 'NoneType'` exception, lines 337–339 — safe but with an
unfriendly error message).

**T-CANON (proof sketch)**: `canonicalize()` preserves (a) the flat index map `Φ_L`,
(b) the set `Rep(L)`, (c) offset semantics. It is implemented as the composition of five
passes (`src/tirx/ir/layout/tile_canonicalize.cc` lines 120–132):

1. `RemoveUnitIters` (lines 31–42): removes extent=1 iters. A unit digit is constantly
   0, contributes nothing to `Φ`, and does not change the row-major decoding of the
   remaining digits. ∎
2. `RemoveZeroOffsets` (lines 44–54): removes zero entries. ∎
3. `FuseAxesByScope` (lines 81–118): rewrites only when the layout contains thread axes
   and a fuser is registered. The axes involved in this dispatch, `m / TCol / TLane`,
   have no fuser (in `axis_registry.cc`, `set_fuser` appears only for
   `tx / warpid / laneid / wgid / tid_in_wg / wid_in_wg`), and with no thread axis
   `GetScope()` has no value and the pass returns immediately (lines 84–86). Hence it is
   the identity here. ∎
4. `FuseContiguousShardIters` (lines 56–79): merges adjacent same-axis iters into
   `(E_k·E_{k+1}, S_{k+1})` only when `E_{k+1}·S_{k+1} = S_k` (provably equal). By the
   identity `d_k·S_k + d_{k+1}·S_{k+1} = (d_k·E_{k+1} + d_{k+1})·S_{k+1}` and the fact
   that the row-major decoding `(d_k·E_{k+1}+d_{k+1})` is exactly the merged digit, the
   map is unchanged. ∎
5. `SortReplicaIters` (lines 134–143): replica is a Minkowski sum, order-independent. ∎

**T-GROUP (proof sketch)**: `L.group(shape)` returns `(L', seps)` satisfying:
(i) `Φ_{L'} = Φ_L`; (ii) the product of extents of the d-th group (shard indices
`[seps[d], seps[d+1])`) = `shape[d]`; failure (boundaries misaligned with the iter
factor structure, or shape not exhausted) leaves via an `ICHECK` exception. The
implementation (`src/tirx/ir/layout/tile_tile_ops.cc` lines 28–72) accumulates products
iter by iter, and when a boundary is hit, splits `(E, S)` into `(E/c, S·c), (c, S)` —
this is inverse to the merging in T-CANON.4, the same mixed-radix identity; the map is
unchanged. ∎

**T-PERM (immediate)**: `permute_dims(perm)` (`layout.py` lines 1281–1289) reorders the
shard by a permutation; `⟦·⟧` as a function of the **multi-dimensional digits** is
unchanged (argument reordering); replica and offset are untouched.
`permute_by_groups(seps, perm)` (lines 1291–1308) is the group-level special case. ∎

**T-ARITH (trusted)**: `arith.Analyzer.can_prove_equal(e, 0)` being true ⟹ `e ≡ 0`
(sound but incomplete: when it cannot prove, the planner rejects — the safe direction).

### 2.2 Instruction axioms

The status of the following three axioms: the PTX ISA 8.8 text provides the skeleton;
where the text is underspecified (warpx2 row assignment, 4x256b lane placement, swizzle
phase anchoring), the facts are pinned down by bit-exact round-trip tests on B200
(`test_tcgen05_cp_shapes.py`: `test_cp_shape_roundtrip_swizzled` / `_nonswizzled` /
`_offsets`, lines 298–339, covering all shape × multicast × sw1–3 × {bf16, f32} ×
{single cp, multiple cps} plus sw0 and offset variants). The planner docstring,
lines 52–60, and the `_cp_lane_replica_pattern` docstring (lines 138–165) record the
same facts.

**AX-DESC (descriptor encoding)**: bit fields of the 64-bit shared-memory descriptor
(PTX §9.7.16.4.1 Table 43, p638): bits 0–13 = `(start_addr & 0x3FFFF) >> 4`,
bits 16–29 = LBO (same encoding), bits 32–45 = SBO, bits 49–51 = matrix base offset,
bits 61–63 = swizzle mode (0 none, 6 = 32B, 4 = 64B, 2 = 128B).
All three fields (start / LBO / SBO) must be 16B aligned (end of p638).
The encoder `ptx_tcgen05_encode_matrix_descriptor`
(`python/tvm/backend/cuda/operator/intrinsics/tcgen05.py` lines 287–317) implements
this bit by bit: swizzle 0/1/2/3 ↦ layout_type 0/6/4/2 (lines 297–303, matching
Table 43), `start_address = cvta(addr) >> 4` (lines 305–306), `base_offset ≡ 0`
(lines 308–309), `sdo/ldo` filled in directly (lines 311–312). `_desc_set_addr`
(planner lines 661–671) replaces the entire 14-bit address field with
`(cvta(addr)>>4) & 0x3FFF` — since the whole field is overwritten, no residue of the
template's original address remains.

**AX-CP-T (TMEM write footprint)**: `tcgen05.cp.cta_group::1.<RxB>{.mc} [taddr], desc`
with taddr (lane field = ρ, 32-bit column field = γ) has the write set: for each data
row `r ∈ [0, R)` and each replication offset `δ ∈ Δ(mc)`, the B bits of that row's data
are written to lane `ρ + λ(r) + δ`, 32-bit column interval `[γ, γ + B/32)`, with the
row's data laid out little-endian by element (element e lands at TCol element coordinate
`γ·ε₃₂ + e`). Here `(λ, Δ)` is given by the table in `_cp_lane_replica_pattern`
(lines 137–177), and that table together with `_CP_SHAPE_TABLE` (lines 126–132)
constitutes part of the axiom's content:

| shape.mc | λ (row → lane, mixed-radix (extent,stride)@TLane) | Δ |
|---|---|---|
| 128x256b / 128x128b | (128, 1): λ(r)=r | {0} |
| 4x256b | (4, 32): λ(r)=32r | {0} |
| 32x128b.warpx4 | (32, 1) | {0,32,64,96} |
| 64x128b.warpx2::02_13 | (64, 1) | {0,64} |
| 64x128b.warpx2::01_23 | (2,64)(32,1): λ(32r₁+r₀)=64r₁+r₀ | {0,32} |

Textual basis: shape = `lane × bits` (§9.7.16.2.3, p622); multicast semantics and warp
pairing (§9.7.16.9.2, p663: "multicasted into warp pairs and each warp in the warp
pair receive half of the data"; 02_13 = {0,2},{1,3}, 01_23 = {0,1},{2,3}).
The text does not fix the row-to-lane order; the orientation taken in the table (02_13
being the datapath organization of Layout E/B, Figures 213/207) is verified bit by bit
by B200 round-trips (docstring lines 52–60).
Each `(λ, Δ)` satisfies: `λ` is injective, and `{λ(r)+δ}` is injective over `(r, δ)`
(warpx4: `[0,32)+{0,32,64,96}` partitions `[0,128)`; 02_13: `[0,64)+{0,64}`; 01_23:
`([0,32)∪[64,96))+{0,32}` partitions `[0,128)`; the rest are immediate) — this
**disjointness** is cited by L12 below.

**AX-CP-S (SMEM read walk)**: let the descriptor be `(start, LBO, SBO, sw)` (start a
byte address, LBO/SBO in units of 16B, sw ∈ {0..3}), shape row count R, `U = B/128`
16B units per row. The **linear** byte address of row `r`, unit `u`:

```
A(r, u) = start + (r mod 8) · K(sw) + (r div 8) · 16·SBO + u · LD(sw)
K(0) = 16,  K(sw>0) = 16·2^sw          (swizzle atom row width atom_K)
LD(0) = 16·LBO,  LD(sw>0) = 16         (when swizzled, LBO is unused and treated as 1)
```

The physical byte address actually read: when `sw = 0` it is `A` itself; when `sw > 0`,
`A` undergoes the `Swizzle<sw,4,3>` XOR (bits `[4,4+sw)` ⊕= bits `[7,7+sw)`), and when
the descriptor's base offset field = 0, the XOR is **anchored at the Table-44 boundaries
of the absolute address** (the 128B/64B/32B swizzles have repeat periods starting at
1024/512/256-byte boundaries respectively; §9.7.16.4.1 Table 44, p638). Shapes with
R < 8 (4x256b) only walk `r ∈ [0,4)`; SBO is never touched.

Textual basis: the K-major canonical layouts table (§9.7.16.3.3, p630): without swizzle
it is `((8,m),(T,2k)) : ((1T, SBO), (1, LBO))` (16B row pitch within a group, LBO
between units), and for sw>0 it is `((8,m),(T,2k)) : ((2^s·T, SBO), (1, T))` composed
with `Swizzle<s,4,3>` (row pitch `atom_K` within a group, the two 16B units linearly
adjacent); LBO under swizzled K-major is
"not used, assumed to be 1" (§9.7.16.3.1.1, p629); SBO = "offset from the
first 8 rows to the next 8 rows" (§9.7.16.3.2, p630).
The **absolute anchoring** point is stated ambiguously in the text but has decisive
experimental evidence: in the swizzled multi-cp tests, per-cp descriptor start points
fall in the interior of a swizzle period yet the results are bit-exact — e.g.
`128x128b, SW128, bf16, n_mid=4`: the s middle stride is 8 elements = 16B, the four
cps' starts are offset 0/16/32/48 bytes in turn from the 1024B-aligned base (interior
of the 1024B period), and the round-trip passes bit-exactly;
`128x256b, SW128, bf16, n_mid=4` likewise (32B stride).
If the phase were anchored at start, these cases would necessarily fail (XOR phase
difference ≠ 0), hence the anchor is the absolute-address boundary
(under the premises base_offset = 0 and a period-aligned base address — the
phase-anchor convention, §5 item 4).

**AX-EXEC (codegen fidelity, trusted)**: the tirx frontend and CUDA codegen faithfully
implement: `T.unroll(total)` fully unrolls `flat = 0..total−1`; `T.meta_var` inlines the
Python-side expression at the unroll point;
`T.ptx.tcgen05.cp(taddr, desc, shape, cta_group, multicast)` generates exactly one
corresponding PTX instruction per call (`intrinsics/tcgen05.py` lines 1345–1387,
asm template `tcgen05.cp.cta_group::G.SHAPE{.MC} [%0], %1`, with taddr packed
identically through `get_tmem_addr(taddr, 0, 0)`); the pointer arithmetic
`T.ptr_byte_offset / ptr_to / cvta_generic_to_shared` has the semantics its names
suggest. Compile-level tests make pinpoint checks on the emitted structure:
`test_cp_4x256b_compile_emits_shape_and_count`
(2 middles → exactly 2 cps of that shape, lines 347–354),
`test_cp_shape_config_routes_to_generic_planner` (template encoded at address 0 +
per-cp 0x3FFF patch ×4, lines 357–368),
`test_cp_default_32x128b_instruction_sequence_unchanged`
(legacy path pinned byte-for-byte: `(ldo,sdo,sw) = (0,8,0)`, 4 cps with tmem columns
0/4/8/12 and smem bytes 0/512/1024/1536, lines 371–430).

---

## 3. Step-by-step lemmas

Each lemma below cites its corresponding planner code lines and states its role within
P1/P2/P3. "Reject" always means leaving `_build_plan` by raising an exception (for
safety see Theorem 2).

### L0 (routing and config resolution; scoping)

`_resolve_cp_shape` (lines 180–208): no `shape` key ⟹ default `"32x128b"`
(lines 184–186, backward compatibility); unknown shape rejected (lines 189–192); a
missing `multicast` can be inferred only when the table entry is unique
(lines 193–200; `64x128b` must be explicit, rejection path lines 197–200); illegal
combinations rejected (lines 203–207).
`_build_plan` lines 319–324: the presence of a **partial** set of `desc_*` keys is
rejected — this guarantees a clean bifurcation between the explicit path
(`_has_explicit_tcgen05_cp_config`, lines 681–688: only a non-default shape with
`desc_ldo/sdo/swizzle` all present goes to `_copy_smem_tmem_explicit_impl`) and the
generic path, so config keys are never silently ignored.
The `decompress` key is likewise explicitly rejected on the generic path
(`copy_smem_tmem_impl` lines 804–807; pinned by
`test_cp_rejects_decompress_on_generic_path`, test lines 554–560).
`_cp_lane_replica_pattern` (lines 137–177) maps `(shape, multicast)` to the `(λ, Δ)`
pattern; unknown multicast rejected (line 177).

Negative cases empirically tested: unknown shape (test lines 518–521), illegal combo
(468–470), missing multicast (473–491), partial desc (524–528), decompress on generic
path (554–560).

### L1 (step A: slice + canonicalize = pairing set translated onto `[0, N)`; P1)

Lines 335–339: both sides `slice(shape, region).canonicalize()`. By T-SLICE, the sliced
layout equals the original layout translated by `R.min` pointwise on `I(R)`; by T-CANON,
canonicalize does not change the flat map. Hence the `Pairs` of Def 4 can be expressed
entirely via `Φ_{L_t}, Φ_{L_s} : [0,N) → coordinates`. The t side's `slice` directly
yields a `TileLayout` (tmem layouts have no swizzle); the s side may be Compose / bare
swizzle, handed over to L2. A slice returning None is rejected via an `AttributeError`
exception (see T-SLICE). ∎

### L2 (swizzle peeling and bare-swizzle recovery; P1 + the linearization convention for P2)

Lines 343–357. Three cases on the s side:
1. `TileLayout`: `s_swizzle_mode_from_layout = 0` (line 343, default);
2. `ComposeLayout(Swz, T)`: record `s = T` (already canonicalized inside Compose's
   canonicalize, compose_layout.cc line 77) and the `swizzle object and mode`
   (lines 345–348);
3. bare `SwizzleLayout`: the Compose canonicalization of a whole-block slice swallows
   the trivial linear part, leaving only Swz (compose_layout.cc lines 78–80); in that
   case recover the linear part from the **uncanonicalized** slice `s_sliced`
   (necessarily a Compose, because `SwizzleLayoutNode::Slice` wraps a Compose first,
   swizzle_layout.cc lines 120–125) and then canonicalize (lines 349–357).
   If this structural expectation is violated, reject (lines 355–356, defensive).

**Lemma**: let the physical map be `Φ_phys = Swz ∘ Φ_s` (§1.3), and the hardware read
address = `Swz_hw ∘ A(·)` (AX-CP-S, with `Swz_hw` the absolutely anchored XOR of
mode `sw`). If (i) `Swz = Swz_hw` as address permutations (same `s`, same
`pe = log2(ε₁₂₈)`, `a=3`, inner, and a consistent phase anchor), then

```
Φ_phys(n) = Swz_hw(A(digit(n)))  ⟺  Φ_s(n) = A(digit(n))
```

I.e.: **it suffices to verify the pairing in the linear (pre-swizzle) space**. The
constituent parts of premise (i) are all discharged by runtime checks except the
phase anchor: the equality of `s` (swizzle_len) is guaranteed by L8's cross-check
(lines 466–471); `pe = log2(ε₁₂₈)` and `a = 3` by the swizzle family check
(lines 472–487; pinned by `test_cp_rejects_non_canonical_swizzle_family`, test
lines 589–616); the `inner` direction by the
`swizzle_inner` check (lines 488–509) — for `sw ≥ 1`, the mirrored permutation with
inner=False coincides with inner=True only on the `2^(3-sw)` blocks, out of each atom
period of `2^(3+sw)` blocks, whose inner `[0,sw)` and outer `[3,3+sw)` bits are all zero
(the code comment contains the exhaustive argument), so a flipped layout is necessarily
rejected (pinned by `test_cp_rejects_flipped_swizzle_inner`, test lines 619–651); for
`sw = 0` both directions are the identity — don't-care.
Only the phase anchor (buffer base aligned to the period + base_offset=0) remains a
trusted convention, recorded in the code comment at lines 808–810 (`alloc_mma`'s
`align=1024` fulfills it for all canonical sources); the planner checks region offsets
(G.2) but never the base address itself. ∎

From here on, `s` always denotes the linear part.

### L3 (step B: replica routing = the destination multicast set is exactly Δ; one half of P2-T)

Lines 359–372: `t.replica` must equal the `(extent, stride, TLane)` list of
`Δ(shape, mc)` **exactly, item by item** (`int()` coercion: symbolic extent/stride is
rejected directly via exception). By Def 2, `Rep(L_t) = Δ`. Each pattern in the table
has at most one replica item, so the sorting of T-CANON.5 does not affect the zip
comparison. Conclusion: the set of `δ` in `Pairs` = the `Δ` of AX-CP-T — the multicast
dimension need not (and must not) be tiled by instructions; it is handed over in its
entirety to the hardware multicast. Negative case: declared 02_13 layout + configured
01_23 → "replica mismatch" (test lines 445–465). ∎

### L4 (steps C/D: common digit space; P1)

Lines 374–379. `perm = _compute_perm(t)` (lines 214–219): the TLane group first, within
groups by descending stride (`int()` coercion ⟹ symbolic stride rejected). Then:

- t side: `t_p = t.permute_dims(perm).canonicalize()`;
- s side: `s.group(t_shape_for_group)` (`t_shape_for_group` = the extents of `t.shard`,
  line 376), yielding per-group products exactly equal to t's corresponding extents
  (T-GROUP (ii)), then `permute_by_groups(seps, perm).canonicalize()`.

**Lemma (digit correspondence)**: before the permutation, the d-th iter of t and the
d-th group of s cover **the same digit segment** of the flat index (T-GROUP cuts s open
with t's extents as boundaries). Applying the same `perm` to both sides (T-PERM) yields
the new common flat order `n'`, and `n ↔ n'` is the same bijection on both sides (the
radix structures agree: the t side by iter extents, the s side by group products = the
same extents; within groups contiguous and row-major). Canonicalize (T-CANON) does not
move the flat map. Hence

```
Pairs = { (Φ_{t_p}(n') ⊕ δ, Φ_{s_p}(n')) : n' ∈ [0,N), δ ∈ Δ }
```

Which specific permutation `perm` is has no bearing on correctness (both sides must
merely agree); this ordering only makes L6's `(lane, middle, col)` group split possible.
A failed group on the s side (factor mismatch) rejects via ICHECK. ∎

### L5 (step E: broadcast isolation; the quotient map of P1)

Lines 381–387 and `_split_by_zero` (lines 222–240). Scan each side's shard in order: a
run of nonzero-stride iters is accumulated by product into one entry, and each
stride=0 iter's extent gets its own separate entry, giving `seq`; keeping the nonzero
iters gives `keep`. Check `seq_t == seq_s` (lines 384–385), else reject. Then
`t_iso / s_iso = from_iters(keep, replica, offset)` (lines 386–387).

**Lemma (soundness of dropping under WF)**: suppose WF-dst holds and `seq_t == seq_s`;
then the stride-0 iters on the two sides occupy **exactly the same flat digit segments**
(same positions and extents), and

```
Pairs(t_p, s_p) = Pairs(t_iso, s_iso)        (as sets; duplicate pairs collapse)
```

Proof: (1) if t has stride=0 on digit segment D while s is nonzero on some digit within
D, then there exist `n ≠ n'` differing only in that digit with `Φ_t` equal and `Φ_s`
unequal — violating WF-dst. So WF-dst ⟹ t's zero segments ⊆ s's zero segments
(digit-wise). (2) If s's zero segments strictly contain t's: the extra zero iter of s
either falls inside a nonzero segment of t (cutting the corresponding single product
entry on the t side into ≥3 entries) or is adjacent to a shared zero segment (changing
the entry count or values); in both cases the length or the values of `seq` must differ
— contradicting `seq_t == seq_s` (note that nonzero entries in `seq` can never be
adjacent: a nonzero run is always accumulated into a single entry). Hence the zero
segments coincide exactly.
(3) With exact coincidence, the quotient map `π : [0,N) → [0,N')` (erasing the zero
digits) satisfies `Φ_{t_p} = Φ_{t_iso} ∘ π`, `Φ_{s_p} = Φ_{s_iso} ∘ π`, `π` is
surjective, and the pairing sets are equal. ∎

**An honest note on the boundary case**: without WF-dst, `seq` equality alone does
**not** imply zero-segment coincidence (there exist interleaved counterexamples of the
`[K,K]` type); but one can verify: any such counterexample either makes the products of
`keep` differ on the two sides — caught by F.5's total-extent equality check
(`_align_middles` lines 266–268) or by F's group failure — or necessarily violates
WF-dst. In other words: **for well-formed inputs, acceptance ⟹ the dropping is sound**;
for inputs violating WF-dst, the planner may accept and emit IR for one particular
reading (garbage-in; WF-dst is not checked, see §1.4). A pure source-side broadcast (s zero segments ⊋ t zero
segments, which is a legitimate copy) is **conservatively rejected** by the `seq`
mismatch — such copies indeed cannot be expressed by the hardware in any way other than
single-instruction multicast anyway. ∎

### L6 (step F: the `(lane, middle, col)` three-segment grouping; the tail of P1 + the stage for P2/P3)

Lines 389–414. `n_lane = atom_rows`, `n_col = epa`,
`n_mid_x = shard_prod(x_iso) / (n_lane·n_col)` (if the division is inexact or yields 0,
the subsequent group rejects via ICHECK). `t_iso.group([n_lane, n_mid_t, n_col])`,
`s_iso.group([n_lane, n_mid_s, n_col])` (lines 396–397); by T-GROUP the segment
boundaries fall at the cumulative products `n_lane` and `n_lane·n_mid` — i.e. the common
flat index decomposition

```
n' = ( r · n_mid + m ) · n_col + e ,   r ∈ [0, atom_rows), e ∈ [0, epa)
```

**with the lane digit highest and the col digit lowest, identically on both sides**.
`n_mid_t = n_mid_s` is guaranteed by F.5 (the total-product equality check in
`_align_middles` lines 266–268), otherwise reject; write the common value as `n_mid`.
The segments are extracted separately: all three t segments go through `_canon_segment`
(lines 401–406, T-CANON preserves the map within a segment); the s lane segment is
**not** canonicalized (line 407 — this introduces conservative rejections, see the
remark in L8), while the middle/col segments are (lines 408–409). By the linearity of
Def 1,

```
Φ_{t_iso}(r,m,e) = O_t + Lt(r) + Mt(m) + Ct(e)
Φ_{s_iso}(r,m,e) = O_s + Ls(r) + Ms(m) + Cs(e)
```

where each term is the mixed-radix linear form of the corresponding segment's iters, and
`O` is the offset. P2 handles `L, C` and `O`; P3 handles `M`. ∎

### L7 (F.1: t-side atom match ⟹ single-cp TMEM footprint; P2-T)

Lines 416–430. Checks:
- `t_lane` (after canonicalization) equals the `λ` pattern item by item (extent, stride,
  axis=TLane all equal, lines 417–425) ⟹ `Lt(r) = λ(r) @ TLane` (the mixed-radix decode
  order agrees with the convention of the AX-CP-T table; e.g. 01_23's `(2,64)(32,1)`:
  `r = 32r₁+r₀ ↦ 64r₁+r₀`).
- `t_col` is a single iter and = `(epa, 1) @ TCol` (lines 426–430) ⟹ `Ct(e) = e @ TCol`.

Combining L3 (`Rep = Δ`) with G.1/G.1b (L10: `O_t` has only a TCol component `O_c`, and
its TLane component is 0), with `m` fixed the t-side destination set of the pairing is:

```
{ (λ(r) + δ  @TLane ,  O_c + Mt(m) + e  @TCol) : r, e, δ }
```

which is **pointwise equal** to AX-CP-T's write footprint at
`taddr = t_addr + (O_c + Mt(m))/ε₃₂` (column field) (the element-coordinate ↔
32-bit-column conversion is exact by G.1's divisibility).
**Corollary (multicast disjointness)**: `(r,δ) ↦ λ(r)+δ` is injective (note after the
AX-CP-T table), so within a single cp distinct `(r,e,δ)` write distinct physical
cells. ∎

### L8 (F.2: s-side row walk ⟹ SDO / atom_K / swizzle mode; P2-S row dimension)

Lines 432–509. `rows_per_group = min(atom_rows, 8)`,
`n_row_groups = atom_rows / rows_per_group`:
- `n_row_groups = 1` (only 4x256b): `s_lane` after canonicalization must be a single
  iter (lines 439–442, else reject), `SDO_byte := 0` (line 443),
  `atom_K_byte = stride·w/8` (line 444).
- Otherwise: `s_lane` is grouped by `[n_row_groups, 8]`, and each of the two blocks must
  be exactly a single iter (lines 446–454, else reject),
  `SDO_byte = inter-group stride·w/8`, `atom_K_byte = intra-group row stride·w/8`
  (lines 455–456); `SDO_byte` must be divisible by 16B, else reject (lines 457–461 —
  the descriptor's SBO is encoded in 16B units, PTX §9.7.16.4.1; pinned by
  `test_cp_rejects_non_16b_aligned_row_group_stride`, test lines 563–586).

By T-GROUP's row-major boundary, `r` decomposes into `(r div 8, r mod 8)`, so

```
Ls(r) = (r div 8)·SDO_e + (r mod 8)·aK_e     (element units; aK_e = atom_K_byte·8/w)
```

**exactly isomorphic to the row terms of AX-CP-S's walk `A(r,·)`** — provided the
descriptor fields are encoded as `SBO = SDO_byte/16` (line 616; the 16B divisibility is
guaranteed by lines 457–461, so the division is exact), swizzle mode `sw`, and
`K(sw) = atom_K_byte`. The latter is pinned down by two checks:
- `atom_K_byte ∈ {16,32,64,128}`, mapped to `derived_sw ∈ {0,1,2,3}` (lines 462–465).
  This simultaneously covers the sw=0 requirement that "the intra-group row pitch must
  be exactly 16B" (the core-matrix row pitch of canonical no-swizzle, `K(0)=16` of
  AX-CP-S) — a row pitch of 32/64/128B would derive `sw>0` and enter the next check.
- `derived_sw == s_swizzle_mode_from_layout` (lines 466–471, else reject):
  the swizzle actually used to store the data (peeled off in L2) and the mode the
  descriptor claims must agree, otherwise the hardware XOR ≠ the layout XOR. This is
  where the equality of `s` in L2's premise (i) comes from.
  The family check that follows (lines 472–487) and the direction check
  (lines 488–509) complete the `pe/a/inner` parts of premise (i) (see L2).

**Remark (conservativeness)**: `s_lane` is taken from the uncanonicalized group segment
(L6); zero-dimension dropping (L5) may leave "fusable but unfused" row iters
(e.g. `[(2, 4aK), (4, aK)]`), so after grouping `blk_8` is not a single iter and gets
rejected, even though the row pitch is semantically uniform. Conservative; harmless to
correctness. ∎

### L9 (F.3: s-side column footprint ⟹ LDO; P2-S column dimension)

Lines 511–548.
- `atom_bits = 128` (U = 1 16B unit): `s_col` must be a single iter `(ε₁₂₈, 1)`
  (lines 517–522, else reject) ⟹ `Cs(e) = e`, consistent with `u ≡ 0` in AX-CP-S and
  contiguous 16B at the row address; `LDO_field := 0` (line 523 — u is constantly 0,
  the hardware does not read LDO; keeps the legacy encoding).
- `atom_bits = 256` (U = 2): `s_col` grouped by `[2, ε₁₂₈]`, each of the two blocks a
  single iter (lines 525–534); within the block `(ε₁₂₈, 1)` contiguity is required
  (lines 535–536); `ldo_byte = inter-unit stride·w/8` must be divisible by 16B
  (lines 537–539, an encoding precondition of AX-DESC). Then
  `Cs(e) = (e div ε₁₂₈)·ldo_e + (e mod ε₁₂₈)`, matching the `u·LD(sw)` term of
  `A(·,u)`:
  - `sw = 0`: `LDO_field = ldo_byte/16` (lines 540–541), hardware `LD(0) = 16·LBO` ✓.
  - `sw > 0`: the hardware forces `LD = 16` (LBO ignored, AX-CP-S / §9.7.16.3.1.1),
    so require `ldo_byte == 16` (the two units adjacent in linear space, lines 542–547,
    else reject); `LDO_field := 1` is a mere placeholder (line 548). ∎

### L10 (step G: alignment ⟹ parameters exactly encodable and anchor legal; the divisibility glue of P2)

Lines 550–589 and 616–618.
- **G.1** (lines 552–560): the TCol component `O_c` of `O_t` must be provably
  `≡ 0 (mod ε₃₂)`, else reject (a 16-bit element landing in the middle of a 32-bit unit
  is unaddressable). `t_col0 = O_c/ε₃₂` (line 618) is exact. Negative case: bf16 column
  offset of 1 element (test lines 494–515).
- **G.1b** (lines 562–569): the TLane component of `O_t` must be provably 0 — the
  emitted cps anchor their multicast footprint at the taddr lane field (= lane 0 of the
  allocation base), and a nonzero lane offset is inexpressible.
  **Executed only when `"shape" in config`**: the legacy default path deliberately
  skips it for byte compatibility (comment at lines 564–565) — a documented equivalence
  waiver: a legacy (no `shape` key) call whose destination region has a nonzero TLane
  offset emits cps anchored at lane 0, which is **not** equivalent to the logical
  semantics, so Theorem 1's scope is restricted to shape-configured calls plus legacy
  calls with zero TLane offset. Negative case: lane offset 64 (test lines 531–551).
- **G.2** (lines 571–589): alignment of the m component of `O_s`: `sw = 0` requires 16B
  (the lossless precondition of the descriptor start's `>>4` encoding, AX-DESC);
  `sw > 0` requires `8·atom_K` bytes (= the swizzle period). The latter, under
  AX-CP-S's absolute-anchoring semantics + base-address period alignment (the
  phase-anchor convention, §5 item 4), is
  **stronger than necessary** (16B suffices; that the middle strides are only checked
  for 16B is the living proof) — purely conservative margin: phase correctness is
  borne by the base alignment and the absolute-anchoring axiom, not by this check.
  `init_off_16B = O_s·w/8/16` (line 617) is exact by that alignment.

The integer divisions of the five quantities
`t_col0 / init_off_16B / SDO_field / LDO_field / each middle stride` are all free of
truncation under the above checks (the 16B precondition of `SDO_field` is guaranteed
by the check at lines 457–461). ∎

### L11 (F.5 + H: 1–1 refinement of the middles; P3 prerequisite)

`_align_middles` (lines 243–297, call site line 414) and H (lines 591–614).
- Take the union of the cumulative extent boundaries of the two sides' middle segments
  (lines 262–270), obtaining the common refinement `shape`; non-dividing boundaries
  reject (lines 273–277); unequal total products reject (lines 266–268, which at the
  same time closes L6's `n_mid_t = n_mid_s`). Each side is then sub-grouped by `shape`
  (T-GROUP, map-preserving), and each segment after canonicalization must be a single
  iter (lines 290–293, else reject — mixed radices admitting no common refinement are
  conservatively rejected). Result: both middles are lists of single iters of length k,
  with the i-th segment covering **the same** flat digit segment on both sides.
- H: the count (lines 592–596) and per-position extent (lines 598–600) checks — after
  F.5 returns successfully these are **always true** (defensive redundancy; when one
  side's middle is empty and the other's is not, `_align_middles` already rejects via
  group's ICHECK, lines 281–295, rather than the readable error at line 268).
  Segments with extent=1 are skipped outright (lines 601–603; digit constantly 0,
  legal). The substantive checks: the t middle axis must be TCol (lines 604–605; tiling
  along the lane direction is inexpressible, conservative rejection); s strides must be
  divisible by 16B (lines 606–608); t strides must be divisible by `ε₃₂`
  (lines 609–613). Output `middle_iters = [(n_i, s_step_16B_i, t_step_32b_i)]`
  (line 614).

**Lemma**: after alignment, `Mt(m) = Σ_i m_i·(t_step_i·ε₃₂)` (elements),
`Ms(m) = Σ_i m_i·(s_step_i·ε₁₂₈)` (elements) = `16·Σ m_i·s_step_i` bytes,
where `(m_1..m_k)` are the middle digits **shared by both sides** (the i-th taking
values in `[0, n_i)`). Per-position divisibility ⟹ divisibility of every combination;
the two sums land exactly on 32-bit-column units and 16B units respectively. ∎

### L12 (step I: emission = P3 tiling)

`_build_plan` returns the plan (lines 620–636), and `copy_smem_tmem_impl`
(lines 800–874) emits:

1. **descriptor template** (`_get_or_create_desc`, lines 644–658): encoded once at
   virtual address 0 keyed by `(LDO_field, SDO_field, sw)` (AX-DESC); the
   `AllocBuffer + Evaluate` is attached after the s_buf definition via
   `add_post_buffer_def_stmt` (lines 655–656) and cached by `(ldo, sdo, sw)` (the key
   does not include buffer identity — value semantics are safe because `_desc_set_addr`
   fully rewrites the address field, but the template's encode statement is attached
   after the **first** matching buffer's definition, so program order must place every
   same-keyed copy after that definition; in real kernels the smem buffer definitions
   always come first).
2. **each cp**: `_cp_desc(off) = _desc_set_addr(desc, ptr_byte_offset(s_base, 16·off))`
   (lines 832–834 + 661–671) replaces the start field wholesale with
   `cvta(s_base + 16·off) >> 4 & 0x3FFF`; on the tmem side
   `t_addr[0] + t_col0 + t_off` is added directly in the column field (get_tmem_addr
   encoding; the absence of carry overflow is guaranteed by the physical bound of
   TMEM ≤ 512 columns — a trusted physical bound, §5 item 4).
3. **enumeration** (lines 840–872): `total = ∏ n_i`; when `total == 1` a single cp is
   emitted (lines 843–849, equivalent to the general term at flat=0); otherwise
   `for flat in T.unroll(total)`, with `compute_offsets` (lines 852–861) decoding
   `idx_i = (flat // ∏_{j<i} n_j) mod n_i` and computing `t_off = Σ idx_i·t_step_i`,
   `s_off = Σ idx_i·s_step_i`.

**Lemma (P3)**:
(a) **Decoding correct and exactly-once**: `flat ↦ (idx_1..idx_k)` is a bijection
`[0, total) → ∏[0,n_i)` (standard mixed radix; note that `middle_iters[0]` varies
fastest in flat, opposite to the layout's row-major digit order — but the enumeration is
over a **set**: each middle value occurs exactly once, the instructions are mutually
independent and asynchronous, and the emission order has no semantic effect). The
positions with extent=1 were skipped by L11 and do not affect bijectivity (those digits
are constantly 0).
(b) **Single cp correct**: fix `m`; by L7 that cp's TMEM write footprint =
`{(λ(r)+δ, O_c + Mt(m) + e)}`; by L8/L9/L10 + AX-CP-S that cp's read address =
`Swz(O_s + Ms(m) + Ls(r) + Cs(e))` (linear space converted back to physical via the L2
lemma); by the row coupling of AX-CP-T/S (row r's source data is written to row r's
lanes), the pairing is `(r, e, δ)` against `(r, e)` — exactly the slice of
`Pairs(t_iso, s_iso)` at `m`.
(c) **Disjointness**: within a single cp by the corollary of L7; across cps by WF-dst
(`(r,m,e,δ) ↦` destination cell is injective; in particular the column intervals of
distinct `m` do not overlap — the planner does not separately check this; violating it
violates WF-dst, see §1.4).
(d) **Union**: `⋃_m slice(m) = Pairs(t_iso, s_iso)`, which via L5 = `Pairs(t_p, s_p)`,
and via L4/L1 = `Pairs`. ∎

### L13 (predicates and execution domain)

`register_dispatch` (lines 878–889) attaches two predicates:
- `_is_valid_smem_tmem_or_explicit_copy` (lines 691–710): with a `shape` key it checks
  only the memory-scope envelope (shared→tmem, layouts on both sides, **equal dtypes**,
  tmem has allocated_addr, lines 700–710) — all fine-grained validation is left to
  `_build_plan`'s readable exceptions. Without a `shape` key it goes through the legacy
  predicate `_is_valid_smem_tmem_copy` (`copy/utils.py` lines 28–65: envelope +
  `R[4:32@TLane]` replica precheck; on mismatch the **predicate fails** and yields to
  other variants — unlike exception-based rejection, this is a silent fall-through;
  that predicate does not check dtype equality — a trusted convention, §5 item 4).
- `_single_thread_exec` (`copy/utils.py` lines 68–72): the exec scope must be
  single-threaded ⟹ the emitted cp sequence is executed exactly once (the other half of
  "exactly once", on the execution-model side). ∎

---

## 4. Theorems

### Theorem 1 (acceptance implies equivalence)

Suppose a call
`Tx.copy_async(tmem[t_region], smem[s_region], shape=…, multicast=…, cta_group=1)`
satisfies:
1. it takes the generic path (`shape ∈ _CP_SHAPE_TABLE`, no `desc_*` keys, no
   `decompress` key) and `_build_plan + copy_smem_tmem_impl` return an `impl` without
   exception;
2. WF-dst / WF-src (after Def 4);
3. the trust base holds (§5: T-SLICE/T-APPLY/T-ARITH, AX-DESC/AX-CP-T/AX-CP-S/AX-EXEC,
   plus the conventions of §5 item 4: phase anchor, smem axis identity, physical
   bounds).

Then the side-effect semantics of `impl` (Def 5) is exactly `Pairs`: for every
`(p, a) ∈ Pairs`, the destination location `p` is written exactly once, with the written
value being the element at source address `a`; and there are no other TMEM writes.

**Note (premise discharge)**: "`SDO_byte ≡ 0 (mod 16)`" and "the swizzle belongs to
the canonical mma atom family" are not premises — both are unconditionally guaranteed
by the planner's runtime checks (lines 457–461 and 472–509). The remaining trusted
conventions are only the phase anchor (base-address period alignment + base_offset=0),
the smem axis identity, and the physical bounds (§5 item 4).

**Proof**: chain L1 (slice translation) → L2 (linearization) → L3 (multicast set = Δ)
→ L4 (common digit space) → L5 (degenerate-dimension quotient) → L6 ((lane, mid, col)
decomposition) → L7/L8/L9/L10 (single-instruction pairing = restriction, P2) →
L11/L12 (tiling exactly once with complete union, P3) → L13 (single-threaded execution
exactly once). Every step is either a pairing-set-preserving transformation (P1) or a
term-by-term match between the instruction axioms and the layout linear forms
(P2/P3). ∎

**Empirical note**: the end-to-end composition of the theorem (including all axioms) is
verified bit by bit by the GPU round-trip matrix of `test_tcgen05_cp_shapes.py`:
6 (shape, multicast) combinations × sw∈{1,2,3} × {bf16,f32} × {n_mid 1, 4}
(lines 298–306), six sw=0 cases including nontrivial LDO (lines 309–324), and three
cases of nonzero smem row offset / nonzero TMEM column offset (lines 327–339). The host
expected values are reconstructed entirely from the destination layout semantics
(lines 189–208), isomorphic to Def 4.

### Theorem 2 (rejection safety)

If `_build_plan` or any layout operator it calls raises an exception, this variant
produces no IR at all: the dispatcher catches it (dispatcher.py lines 313–322 "keep
searching other variants") and continues trying lower-priority variants; if all fail it
raises `RuntimeError` with an aggregated report (lines 324–329), so the compile failure
is visible. Predicate failures (L13) are likewise merely recorded and skipped
(lines 273–295). Hence there is no path that "accepts a wrong combination and emits
wrong IR" — the only exceptions are inputs violating the unchecked WF-dst assumption
(§1.4) and the legacy-path TLane-offset waiver (L10, G.1b).
Negative-case tests (lines 445–651) verify that all eleven classes of rejection surface
as compile errors with readable messages. ∎

### Completeness discussion (conservative, not incorrect)

All of the following are cases where a **legal copy is rejected** (the safe direction):
- t middle axis containing TLane (lines 604–605): tiling multiple cps along the lane
  direction (e.g. covering 8 rows with 4x256b) is unsupported — the hardware taddr lane
  field is actually addressable; purely conservative.
- TLane offset of `t_iso` ≠ 0 (lines 566–569): same as above; the anchor is fixed at
  lane 0.
- Symbolic (non-constant) stride / extent: the `int(…)` coercions everywhere
  (e.g. lines 217, 229, 444, 455–456, 537, 599–613) reject via exception; only
  **offset** may be a symbolic expression (alignment proven via T-ARITH).
- Pure source-side broadcast copies (remark in L5); middles whose mixed radices admit no
  common refinement across the two sides (lines 273–293);
- Unfused s row structure left over after zero-dimension dropping (remark in L8);
- `64x128b` without an explicit multicast choice (lines 197–200); replica declarations
  not matching the pattern exactly (L3);
- 64-bit dtypes (`ε₃₂ = 0` triggers a division-by-zero exception); non-dividing `n_mid`
  (L6);
- swizzle mode inconsistent with the row pitch (lines 466–471), non-canonical swizzle
  family/direction (lines 472–509) — these are in fact mostly **incorrect** copies, so
  rejection is all the more necessary.

---

## 5. Trust base (summary)

1. **layout operators**: T-SLICE (all of tile_slice.cc), T-APPLY (FFI/Apply
   reflection), T-ARITH (soundness of arith.Analyzer). T-CANON/T-GROUP/T-PERM come with
   sketch proofs.
2. **PTX axioms**: AX-DESC (§9.7.16.4.1 Tables 43/44), AX-CP-T (§9.7.16.2.3,
   §9.7.16.9.2 + the row→lane table), AX-CP-S (§9.7.16.3.1–.3.3 + absolute anchoring).
   Where the text is underspecified (row-assignment orientation, 4x256b lane placement,
   anchoring semantics), the bit-exact B200 round-trips of `test_tcgen05_cp_shapes.py`
   are authoritative (the same tests also serve as the empirical evidence of the
   end-to-end composition).
3. **codegen**: AX-EXEC (tirx unroll/meta_var/pointer builtins/verbatim asm template
   generation), spot-checked by the compile-level pin tests (lines 347–430).
4. **conventions**: (i) **phase anchor** — the smem base address is aligned to the
   swizzle period and the descriptor base offset is 0 (encoder,
   intrinsics/tcgen05.py lines 308–309); the planner checks region offsets (G.2) but
   never the base address; the convention is recorded in the code comment at
   lines 808–810 (`alloc_mma`'s `align=1024` fulfills it for all canonical sources).
   (ii) **smem axis identity** — smem layouts carry only the memory axis `m`, in
   element units; F.2/F.3/H compare extent/stride and never check `axis == m` (the
   corresponding t-side checks exist at lines 418, 429, 604). (iii) **physical
   bounds** — the taddr column-field arithmetic `t_addr[0] + t_col0 + t_off` relies on
   TMEM ≤ 512 columns to exclude 16-bit column-field overflow; the lane field of
   `allocated_addr` is trusted to be anchored at lane 0; the legacy predicate
   (copy/utils.py lines 28–65) does not check src/dst dtype equality (the shape-keyed
   path does, at line 708). (The swizzle-family canonicality and the SDO 16B
   precondition are not conventions: they are discharged by the runtime checks at
   lines 472–509 and 457–461.)
5. **caller contract**: WF-dst / WF-src; the asynchronous completion signal
   (`tcgen05.commit`); region legality (within buffer bounds).

---

## Appendix A: code checkpoint ↔ lemma correspondence table

Every explicit rejection / validation on the generic path in `tcgen05_cp.py` belongs to
exactly one lemma (line numbers re-verified against the current working tree):

| Lines | Check | Lemma |
|---|---|---|
| 189–192 | unknown shape | L0 |
| 197–200 | 64x128b missing multicast | L0 |
| 203–207 | illegal (shape, multicast) | L0 |
| 177 | unknown multicast pattern | L0 |
| 319–324 | partial desc_* keys | L0 |
| 804–807 | `decompress` rejected on the generic path | L0 |
| 355–356 | anomalous bare-SwizzleLayout slice structure | L2 |
| 359–372 | replica does not match Δ | L3 |
| 375/217 | `int(stride)` in perm (symbolic rejected) | L4 |
| 384–385 | zero-dimension split sequences mismatch | L5 |
| 394–397 | n_mid divisibility / group feasibility (ICHECK) | L6 |
| 417–425 | t_lane ≠ λ pattern | L7 |
| 426–430 | t_col ≠ (epa, 1@TCol) | L7 |
| 439–442 | 4-row group not a single iter | L8 |
| 446–454 | s_lane not (n_grp, 8) single-iter blocks | L8 |
| 457–461 | SDO_byte not a multiple of 16B | L8 |
| 462–465 | atom_K ∉ {16,32,64,128} | L8 |
| 466–471 | swizzle mode contradicts atom_K | L8 |
| 472–487 | swizzle family (per_element/atom_len) non-canonical | L2/L8 |
| 488–509 | swizzle_inner=False direction flip rejected (waived for sw=0) | L2/L8 |
| 517–522 | 128b s_col not a (ε₁₂₈, 1) single iter | L9 |
| 525–534 | 256b s_col not of (2, ε₁₂₈) structure | L9 |
| 535–536 | discontiguous within a 16B unit | L9 |
| 537–539 | ldo not a multiple of 16B | L9 |
| 542–547 | swizzled 256b units not adjacent | L9 |
| 552–560 | TCol offset not 32b aligned | L10 |
| 562–569 | TLane offset ≠ 0 (shape-configured only) | L10 (legacy waiver) |
| 571–589 | s offset alignment (16B / period) | L10 (conservative) |
| 592–596 | middle count (dead code) | L11 |
| 598–600 | middle extents (dead code) | L11 |
| 604–605 | t middle axis not TCol | L11 |
| 606–608 | s middle stride not a multiple of 16B | L11 |
| 609–613 | t middle stride not a multiple of ε₃₂ | L11 |
| 266–268 | `_align_middles` total products unequal | L11 (also closes L6) |
| 273–277 | boundary does not divide | L11 |
| 290–293 | refined segment not a single iter after canonicalization | L11 |
| 616 | `SDO_field = SDO_byte // 16` (divisibility guaranteed by lines 457–461) | L8/L10 |
| 691–710 | memory-scope envelope predicate | L13 |
| utils.py 68–72 | single-threaded exec scope | L13 |

Emission side (no rejection, construction only): 620–636 (plan), 644–658 (template
cache), 661–671 (address patch), 800–874 (cp loop) → L12. The explicit path 713–792 and
the legacy routing 691–695 are outside the scope of this theorem (covered respectively
by the caller's hand-computed contract and the historical byte-compatibility pin
tests).
