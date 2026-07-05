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

# Semantic Correctness Proof of the Self-Encoded Instruction Descriptor (descI) in the gemm_async tcgen05 Dispatch

Subject: `python/tvm/backend/cuda/operator/tile_primitive/gemm_async/tcgen05.py`
(hereafter the **dispatch**; all line numbers refer to the current
version of this file unless another file name is noted).

Proposition (informal): for every dense (non-block-scaled, `descI` defaulted)
`Tx.gemm_async(C, A, B, transA, transB, accum, …)` call accepted by the
dispatch, the instruction descriptor `descI_value` self-encoded at compile time
by the dispatch (lines 1381–1391), together with the SMEM descriptors
`(descA, descB)` it constructs alongside and the emitted `tcgen05.mma[.ws]`
instruction sequence, implements exactly the logical GEMM, under the
instruction axioms and the trust base (§2, §5):

```
C[m, n]  (+)=  Σ_{k=0}^{K-1}  A'[m, k] · B'[k, n]
```

(`(+)=` denotes accumulate when `accum` is true, overwrite otherwise; `A'/B'`
are the logical matrices under the transA/transB convention, see §1.2). In
particular: **kernels no longer need a hand-written `descI`** — the encoded
values before and after the head64 kernel fold (deleting the hand-written
`descI`) are bit-for-bit equal (§3 L9, §4 Theorem 2), and the dense path
**explicitly rejects** a hand-written `descI` (lines 526–529).

Suggested reading order: §0 overview → §1 definitions → §2 axioms and trusted
lemmas → §3 step-by-step lemmas → §4 theorems → §5 trust base →
Appendix A (full table of code checkpoints ↔ lemmas).

Sibling document: `tcgen05_cp.md` (hereafter the
**cp proof**). This document reuses its notation and already-proven lemmas
(T-SLICE / T-APPLY / T-CANON / T-GROUP / swizzle semantic equivalence) without
re-proving them.

---

## 0. Overview: The Triple-Consistency Problem

### 0.1 Shape of the Problem

A single issue of `tcgen05.mma` is driven by three encoded values:

- **descA / descB** (64-bit SMEM matrix descriptors): the physical layout of
  the matrix in SMEM (start, LBO, SBO, swizzle mode);
- **idesc** (32-bit instruction descriptor): shape (M, N), dtype formats,
  majorness (bits 15/16), negate/saturate/sparsity, ws max-shift
  (bits 30–31).

The hardware uses the **majorness bits of the idesc** as the switch for
interpreting the **LBO/SBO fields of the descriptors**
(PTX §9.7.16.3.1/.3.2: the same field means different things under K-major
vs. MN-major). Correctness is therefore not the separate correctness of two
independent encodings but a **triple-consistency** proposition:

> **Correctness criterion**: for every issued MMA there exists a unique
> "hardware read walk" (jointly determined by
> (idesc.majorness, desc.LBO/SBO/swizzle, desc.start), axiom AX-SMEM-WALK)
> whose reads of `A'[m,k]`, `B'[k,n]` are exactly the logical elements
> physically stored by the caller; and the (M-tile, N-tile, K-iter) tiling of
> all the MMAs covers the summation of the logical GEMM exactly once.

The historical majorness bug (the no-swizzle col-major-view branch returned
`not is_transposed`; the old value is visible at HEAD line 724) is precisely
an instance of triple inconsistency: `(ldo, sdo)` was constructed per the
K-major ABI while the majorness bit was returned as MN-major — each half
looks "reasonable" in isolation, but the pairing is wrong. The technical core
of this proof is therefore **L4 (majorness–field co-origination)**: the
`(swizzle_mode, ldo, sdo, mn_major)` quadruple returned by each branch of
`compute_canonical_params` must be the two halves of **the same physical
access** under the AX axioms.

### 0.2 Proof Decomposition

- **P1 (encoding fidelity)**: the compile-time Python mirror
  `_encode_instr_descriptor_dense_uint32` ≡ the runtime C bit-field packing
  ≡ the PTX Table 45 bit table (L1, L2); the four construction paths of the
  SMEM descriptors are mutually equal and ≡ PTX Table 43 (L3).
- **P2 (single-instruction consistency)**: the quadruple of each
  `compute_canonical_params` branch is co-original with AX-SMEM-WALK (L4);
  dimension and dtype cross-checks (L5, L6); the C-side TMEM footprint
  matches the datapath layout (L7).
- **P3 (tiling)**: the one-to-one correspondence between the descriptor
  stepping / TMEM address stepping / accumulation chain of the
  `(mi, ni, ki)` triple loop and the GEMM summation (L8).
- **Instantiation**: the encoded values at the two classes of head64 sites
  (P/O) and in the unit tests = the deleted hand-written values (L9).

### 0.3 Safety Criterion

Same as the cp proof: the acceptance decision and the emission share the same
code (`gemm_async_tcgen05_impl`, lines 370–1399); any precondition failure
leaves via `ValueError` / `AssertionError`, which the dispatcher
(`python/tvm/tirx/operator/tile_primitive/dispatcher.py` lines 298–329)
catches and then tries other variants, aggregating an error if all fail.
There is no "half-accepted" state.

---

## 1. Definitions

### 1.1 GEMM Semantics and Operand Conventions

A dense call carries `(C_region, A_region, B_region, transA, transB, accum)`
(the arg layout of `GemmAsync`: `ops.py` lines 282–333; for dense,
args[3:6] = transA/transB/accum). Region shape conventions (`_mat_dim_vals`
takes the non-unit dims, lines 554–560):

- `C`: `[M, N]` (always 2D, lines 550–552);
- `A`: `transA=False → [M, K]`; `transA=True → [K, M]` (lines 886–889,
  909–911);
- `B`: `transB=False → [N, K]`; `transB=True → [K, N]` (lines 913–921:
  `B_K = B_dim1 if not transB else B_dim2`).

**Def 1 (GEMM semantics)**: let `A'[m,k]` = the logical element read from the
A region under the above convention (i.e. `A'[m,k] = A[m,k]` or `A[k,m]`),
and likewise for `B'`. Then the prescribed effect of the call is: for every
`(m, n) ∈ [0,M) × [0,N)`,

```
accum = false:  C[m,n] ← Σ_k A'[m,k]·B'[k,n]
accum = true :  C[m,n] ← C[m,n] + Σ_k A'[m,k]·B'[k,n]
```

Multiply-accumulate follows the hardware dtype semantics (`kind::f16`, etc.);
numerical precision is outside the proposition, but the pairing structure is
inside it. The asynchrony contract is the same as in the cp proof: the
completion signal (`tcgen05.commit`) is the caller's responsibility; the
proposition speaks only of the final writes.

**Def 2 (majorness terminology, PTX §9.7.16.10.6, printed page p678 / PDF
page 690)**:

> "If the bit Transpose A Matrix / Transpose B Matrix in the Instruction
> descriptor is 0, then K-major is used for matrix A / B respectively. …
> we will use MN-Major and K-Major throughout this section."

- **K-major**: in SMEM the K dim is the contiguous dim (within 16B units);
- **MN-major**: the M dim (for A) or the N dim (for B) is contiguous.

Majorness is a **physical layout property**, independent of transA/transB
(a **shape order** convention): an `[M,K]` region can be stored either
K-major or M-major. The dispatch uses `is_transposed` to decide "which axis
is K" and the atom matching / branch rules to decide "which axis is
contiguous"; the two combine into the majorness bit (comment at
lines 574–580, L4).

### 1.2 Physical Layout and 16B Normalization

- dtype bit width `w`; `E := 128/w` (elements per 16B unit; `elem_per_128b`
  at line 629, `elem_per_16b/B` at lines 709/750). All three offset fields of
  the descriptor are in 16B units
  (`matrix-descriptor-encode(x) = (x & 0x3FFFF) >> 4`, Table 43).
- swizzle atom (`mma_atom_shape`, tma_utils.py lines 47–61): mode
  s ∈ {1,2,3} (32B/64B/128B) corresponds to `[8, 2^s·E]` elements,
  `8·2^s·16 = 2^(7+s)` bytes = exactly one swizzle repetition period.
  `mma_atom_layout` (lines 37–44) = `SwizzleLayout(pe=log2 E, s, 3)`; its
  equivalence to PTX `Swizzle<s,4,3>` is shown in cp proof §1.3 (cited
  directly here).
- TMEM: axes `TLane` (0–127) / `TCol` (in element units); the taddr packing
  is bits 16–31 = lane, bits 0–15 = 32-bit column (cp proof §1.2 Def 3).

### 1.3 dtype Semantic Names

`A_sem/B_sem = "tf32" if is_AB_tf32 else the storage dtype` (lines 439–444);
the dense domain is `_DENSE_DTYPES = {float16, bfloat16, float8_e4m3fn,
float8_e5m2, tensor_float32, tf32}` (lines 459–472), with `A_sem == B_sem`
(lines 473–475) and `C_type == float32` (line 436). `MMA_K` is taken from
`A_sem` as {fp4:64, fp8:32, tf32:8, f16/bf16:16} (lines 891–899).

---

## 2. Axioms and Trusted Lemmas

The `T-` prefix denotes layout-algebra facts (proven/trusted in cp proof
§2.1, cited directly). The `AX-` prefix denotes hardware / codegen axioms,
grounded in the PTX ISA 8.8 text plus B200 measurements. Page numbers are
printed page numbers of the PTX ISA 8.8 PDF.

### 2.1 AX-IDESC (Dense Instruction Descriptor Bit Table)

PTX §9.7.16.4.2 **Table 45** ("Instruction descriptor format for
.kind::tf32, .kind::f16, .kind::f8f6f4 and .kind::i8", printed page p639 /
PDF page 651), bit by bit:

| bit | width | field | value |
|---|---|---|---|
| 0–1 | 2 | Sparsity selector | 0–3 (sparse only) |
| 2 | 1 | Sparsity | Dense=0 / Sparse=1 |
| 3 | 1 | Saturate (integer) | 0/1 |
| 4–5 | 2 | dtype (D format) | f16 kind: F16=0; F32=1 for tf32/f16 kinds; i8: S32=2 |
| 6 | 1 | Reserved | 0 |
| 7–9 | 3 | atype | f16 kind: F16=0, BF16=1; tf32: TF32=2; f8f6f4: E4M3=0, E5M2=1, E2M3=3, E3M2=4, E2M1=5 |
| 10–12 | 3 | btype | same as above |
| 13 | 1 | Negate A | 0/1 |
| 14 | 1 | Negate B | 0/1 |
| 15 | 1 | **Transpose A** (majorness) | No Transpose (K-major)=0 / Transpose=1 |
| 16 | 1 | **Transpose B** (majorness) | same as above |
| 17–22 | 6 | N (3 LSBs excluded) | `N >> 3`, 1 (N=8) … 32 (N=256) |
| 23 | 1 | Reserved | 0 |
| 24–28 | 5 | M (4 LSBs excluded) | `M >> 4`, 4 (M=64) / 8 (M=128) / 16 (M=256) |
| 29 | 1 | Reserved | 0 |
| 30–31 | 2 | **Maximum shift while attempting B matrix reuse in .ws** | 0=no shift, 1=8, 2=16, 3=32 |

The majorness semantics of bits 15/16 are given by §9.7.16.10.6 (p678 / PDF
page 690, the quotation in §1.1 Def 2). Note: **K does not enter the idesc**
— dense K is uniquely determined by the kind (f16→16, tf32→8, f8f6f4→32; the
PTX shape constraints are delegated via the Target ISA Note, mirrored on the
runtime side as `_TCGEN05_MMA_K`, `intrinsics/tcgen05.py` lines 435–443);
cta_group does not enter the idesc either (it is the instruction qualifier
`.cta_group::N`).

### 2.2 AX-SMEM-DESC (SMEM Descriptor Bit Table) and AX-SMEM-WALK (Read Walk)

**AX-SMEM-DESC** (§9.7.16.4.1 Table 43, p638): bits 0–13 = start(>>4),
bits 16–29 = LBO(>>4), bits 32–45 = SBO(>>4), bits 46–48 = constant 0b001,
bits 49–51 = matrix base offset, bit 52 = LBO mode (0 = relative offset),
bits 61–63 = swizzle (0 none, 6=32B, 4=64B, 2=128B, 1=128B-32B-atomicity).
The three fields must be 16B-aligned. base offset = 0 if and only if the
start of the swizzle repetition pattern falls on the absolute boundaries of
Table 44 (p638) (128B/64B/32B swizzle → 1024/512/256B boundaries).

**AX-SMEM-WALK** (field semantics; majorness selected by the idesc bits):

- LBO (§9.7.16.3.1.1, p629):
  - K-major: with no swizzle, = "the stride from the first column to the
    second column of the 8x2 tile in the 128-bit element normalized matrix"
    (the spacing between adjacent 16B K-column units); **when swizzled,
    "not used, assumed to be 1"**.
  - MN-major: when swizzled, = "stride from the first
    (swizzle-byte-size/16) rows to the next (swizzle-byte-size/16) rows"
    (the spacing between adjacent swizzle-block rows along the MN direction).
- SBO (§9.7.16.3.2, p630): K-major = "offset from the first 8 rows to the
  next 8 rows" (the spacing between 8-row MN groups); MN-major (swizzled) =
  "offset from the first 8 columns to the next 8 columns" (the spacing
  between 8-column K groups).
- **Canonical layouts** (§9.7.16.3.3 Canonical Layouts table, p630, CuTe
  notation; T = E):

```
K-major,  no swizzle: ((8,m),(T,2k)) : ((1T, SBO), (1, LBO))   ∘ Swizzle<0,4,3>
K-major,  sw s>0:     ((8,m),(T,2k)) : ((2^s·T, SBO), (1, T))  ∘ Swizzle<s,4,3>
MN-major, no swizzle: ((T,1,m),(8,k)) : ((1, T, SBO), (1T, LBO)) ∘ Swizzle<0,4,3>
MN-major, sw s>0:     ((T,2^s,m),(8,k)) : ((1, T, LBO), (2^s·T, SBO)) ∘ Swizzle<s,4,3>
```

  That is, the hardware reads the M×K elements of A / the K×N elements of B
  according to this layout (with start as the origin, LBO/SBO as free
  parameters, and the swizzle XOR phase anchored — when base offset = 0 — to
  the Table 44 absolute boundaries; for the measured argument for absolute
  anchoring see cp proof AX-CP-S: `tcgen05.mma` and `.cp` share the same
  descriptor encoding and Table 44). The swizzle atom shapes cross-check
  against **Table 55** (§9.7.16.10.6, p678 / PDF page 690): the 128B-swizzle
  K-major atom is 8×8 (in 128b element units; smaller elements scale by
  ×E/1 along the leading dim), consistent with `[8, 2^s·E]` in §1.2.

**Majorness legality domain** (Tables 52/54, p667–668): MN-major
(trans bit = 1) is legal only for E4M3/E5M2/INT8/F16/BF16/TF32; moreover,
MN-major for tf32 allows only the 128B-32B-atomicity swizzle, while the
other kinds are exactly the opposite (the dispatch rejects tf32 + MN-major,
lines 867–884; L4e).

### 2.3 AX-MMA (tcgen05.mma / .ws Instruction Semantics)

Per §9.7.16.10 (p665) and §9.7.16.10.9.1/.3 (p712–725):

1. `tcgen05.mma{.ws}.cta_group::G.kind::K [d-tmem], a, b-desc, idesc, …`
   initiates `D = A·B + D` (`D = A·B` when the `enable-input-d` predicate is
   false): A is M×K (TMEM or SMEM desc), B is K×N (SMEM desc), D is M×N
   (TMEM). M/N come from idesc bits 24–28/17–22; K is uniquely determined by
   the kind (§2.1). **Single-thread issue semantics** (Table 49, p644).
2. **`.ws` and non-`.ws` share the Table 45 idesc format** (§9.7.16.10.9.3
   states verbatim "The 32-bit register operand idesc is the instruction
   descriptor as described in Instruction descriptor", p724). The difference
   lies only at the tail of the operand list: non-ws carries the
   `{disable-output-lane}` vector (all zeros ⇒ no lane is masked, p715);
   ws replaces it with an optional `zero-column-mask-desc` (Table 48, p642:
   bit 39 Non-Zero Mask = 0 ⇒ an all-zero mask is generated ⇒ all columns of
   B participate in the computation, p643 Example 1).
3. **Correctness of bits 30–31 (max-shift) = 0**: this field constrains the
   maximum shift amount only when ".ws attempts B-matrix reuse" (Table 45,
   rows 30–31). The emitted ws instructions carry no `.collector_usage`
   qualifier ⇒ default `.collector::b0::discard` (§9.7.16.10.9.3, p725:
   "If no .collector_usage qualifier is specified, then it defaults to
   .collector::b0::discard") ⇒ every MMA re-reads B from SMEM and
   establishes no cross-instruction reuse state ⇒ "no shift = 0" is the
   appropriate encoding for this mode. (Collector reuse itself, even when
   permitted, is opportunistic and does not change value semantics,
   §9.7.16.10, p666.)
4. **D-side datapath footprint** (§9.7.16.10.5, p671–677): with
   cta_group::1, M=128 → Layout D (lane r = row r); M=64 + `.ws` →
   **Layout E** (§9.7.16.10.5.5, p676: the column space is folded in half,
   lane = m + 64·(n ≥ N/2), physical column = n mod (N/2)); M=64 non-ws →
   Layout F (scattered). `tmem_datapath_layout`
   (`python/tvm/tirx/layout.py` lines 606–696) implements this table
   verbatim.

   **Layout E is a batch fold, and this dispatch conflates it with N — a
   known semantic gap (SEM-WS-BATCH).** The hardware `.ws` computes a plain
   `D[M,N] = A·B` (point 1), and for M=64 stores the two halves of N in the
   two 64-lane banks (`n < N/2` → lane m, `n ≥ N/2` → lane m+64). But the
   two banks are an *output* axis of extent 2, and a caller is free to use
   that axis as a **batch** rather than as the upper/lower half of one N
   range. FlashMLA's NoPE gemm does exactly this: it feeds A and B so that
   bank `b ∈ {0,1}` receives the partial contraction over head-dim half `b`
   of the *same* 64 keys, i.e. the true operation is a **batch-2 M=64 bmm**
   `einsum("i k0 k1 k2, j k0 k1 k2 -> k1 i j")` (k1 = 2 = the fold/batch,
   contracted over k0,k2), producing `C[2,64,64]` — two partials that the
   kernel reduces *downstream* by adding the two banks
   (`P[h,key] = tmem_p[h,key] + tmem_p[h,key+64]`,
   `sparse_prefill_head64_phase1.py`). The tell-tale is that `tmem_p` is
   64×128 = twice the 64×64 a complete Q·Kᵀ over one key set could produce,
   and that adding bank 0 to bank 1 is only meaningful if they are the same
   keys' two head-dim halves, never two independent N-halves.

   So dispatching this via `gemm_async(A[M,K], B[N,K], C[M,N])` with the
   packed Layout-E C is **byte-correct but semantically a coincidence**: the
   physical footprint is identical whether the "2" is an N-half or a batch,
   so the emitted `.ws` writes the right bytes and the standalone-gemm tests
   (§2.3 measured anchors, which feed a genuine N=128) pass — but the
   dispatch never checks that the caller's batch semantics match.

   **SEM-WS-BATCH remediation (implemented, lines 563–599):** the dispatch
   now accepts C in the **explicit batched form** `C[2, M, N//2]` with the
   fold layout `(2, M, N//2):(64@TLane, 1@TLane, 1@TCol)`, so a caller spells
   the two `.ws` output partials as a first-class batch dim instead of hiding
   them in the packed N-fold. **This path is correct-by-construction, and
   acceptance implies a correct result:**

   1. *Acceptance is exact.* A batched C is accepted only if its sliced
      layout `assert_structural_equal`s `(2, M, N//2):(64@TLane, 1@TLane,
      1@TCol)` (line 584). The M=64 `.ws` datapath is the *only* producer of
      this fold (Layout E; a non-ws M=64 MMA writes the scattered Layout F,
      which cannot match). So a mis-declared C fails the assert — it is never
      silently accepted.
   2. *It reduces to the proven packed path.* `(2, M, N//2):(64@TLane,
      1@TLane, 1@TCol)` and the packed `(M, 2, N//2):(1@TLane, 64@TLane,
      1@TCol)` are the **same physical tile** — batch `b`, row `m`, col `n`
      both map to lane `m + 64·b`, column `n` (`b ≡ half`). After validation
      the code normalizes `C_slice_layout` to the packed form (lines 585–588),
      so every downstream step (`packed_n2`, the emit, the D-footprint of
      AX-MMA.4) runs **byte-identically** to the packed path proved in §3.
      This is checked in-process by
      `test_gemm_tcgen05_cta1_m64_accepts_batched_c_layout_ws` (batched source
      == packed source up to the entry-point name).
   3. *`.ws` is inferred soundly.* Because the fold layout is uniquely the
      M=64 `.ws` datapath, the batched form implies `weight_stationary`
      (lines 596–598 set it for `cta_group::1`) — the caller need not pass the
      flag. This cannot mis-select: any C that is *not* the `.ws` fold fails
      step 1's assert, so "accepted batched C" ⟺ ".ws is the correct
      datapath". Hence **dispatch-accepts ⟹ generated code is correct**.

   **Packed vs batched — when each is correct.** The packed C[M, 2, N//2]
   path is *not* deleted, because the M=64 fold is also the correct spelling
   for a **genuine N** output whose two banks are distinct output columns
   (reduced by nothing downstream) — e.g. FlashMLA head64's `O = P·V` gemm
   writes `tmem_o_{lo,hi}` = O[:, 0:256] / O[:, 256:512], real N=256 folded
   into 64 lanes. That is a legitimate packed-C use and stays. The **wrong
   path** is precisely the opposite: using packed C to mean a *batch* whose
   two banks are summed downstream (the logits gemm, `tmem_p[h,key] +
   tmem_p[h,key+64]`). That caller — head64's only one — is now migrated to
   the explicit `C[2, M, N//2]` form, so no batch is hidden in an N-fold
   anymore. The dispatch cannot tell genuine-N from batch at the packed
   level (identical bytes), which is exactly why the batch case must declare
   itself via the batched shape; both spellings remain, each correct for its
   intent. (cta_group::2 M=64 uses the 2×2 Layout B — a different, M-across-
   CTA organization — and is unaffected by all of this.)
5. **Ordering** (§9.7.16.6.2, p646): `tcgen05.mma.cta_group::N →
   tcgen05.mma.cta_group::N (same N and accumulator and shape)` forms a
   pipeline, guaranteeing execution in issue order — the correctness basis
   of the K-chain accumulation.
6. The A-in-TMEM form (`[a-tmem]`) has no notion of majorness (no descriptor
   exists); idesc bit 15 must be 0 (the K-major default). The spec text does
   not explicitly address this point; it is anchored by measurements from
   FlashMLA/the unit tests (trusted, see L4d).

Measured anchors: the GPU numerical matrix in
`tests/python/tirx/operator/tile_primitive/cuda/gemm_async/test_gemm_async.py`
(`test_gemm_tcgen05_cta_group_1/2` lines 186/428, `_layout_f_m64` line 302,
`_arbitrary_tiles` line 1834, `_contiguous_kslice_partial_k` line 2158, and
`test_gemm_tcgen05_no_swizzle_col_major_a_ws_local_idesc` lines 2040–2152,
including the datapath-E half-by-half read-back check at lines 2141–2152).

### 2.4 AX-EXEC (Codegen Fidelity, Trusted)

- `T.ptx.tcgen05.mma(...)` goes through `backend/cuda/op.py`
  lines 2450–2543 (defaults `disable_output_lane = [0]*4`,
  `scale_input_d=0`) → `intrinsics/tcgen05.py::_mma_dense_parts`
  (lines 736–875), generating a single asm statement operand by operand:
  non-ws template
  `tcgen05.mma.cta_group::G.kind::K [%0], %1, %2, %3, {m0..m3}, p;`
  (lines 858–875, `p = (enable_input_d != 0)`); ws template
  `tcgen05.mma.ws.cta_group::1.kind::K [%0], %1, %2, %3, p, 0;`
  (lines 837–856, the trailing literal `0` = zero-column-mask-desc,
  AX-MMA.2). The ws side rejects sparse / cta_group≠1 / scale_input_d
  (lines 764–769).
- `T.unroll` fully unrolls, `T.meta_var` inlines, `@T.inline`
  macro-expands, and the pointer builtins have the semantics their names
  suggest (same as cp proof AX-EXEC).
- At warp scope, `elect_pred = T.ptx.elect_sync()` (line 1028) and every MMA
  is wrapped in `if elect_pred` (lines 1266/1313) ⇒ exactly one thread
  issues (the execution-domain premise of AX-MMA.1; at single-thread scope
  the predicate is identically true). The dispatch predicate is
  `single_thread_or_warp` (lines 1430–1444).

### 2.5 T-Lemmas (Layout Algebra, Cited from the cp Proof)

T-SLICE / T-APPLY / T-CANON / T-GROUP / T-ARITH are as in cp proof §2.1.
This document additionally cites:

**T-TILE-INNER** (`src/tirx/ir/layout/tile_tile_ops.cc` lines 270–366;
`swizzle_layout.cc` lines 95–112, `compose_layout.cc` lines 91–99): when
`atom.is_tile_inner(L, shape, atom_shape)` returns a non-None tiler:

1. (swizzle congruence) If the atom contains a `SwizzleLayout`, then `L`
   must be a `ComposeLayout(Swz', T')` or a bare `SwizzleLayout`, and `Swz'`
   is **structurally equal** to the atom's swizzle (`StructuralEqual`:
   per_element, swizzle_len, atom_len, inner all compared,
   swizzle_layout.cc lines 100/105) — a swizzle-mode mismatch between the
   buffer and the descriptor therefore cannot pass matching (the cp planner
   enforces the same congruence);
2. (decomposition) `Φ_L(i_outer ⊗ i_inner) = tiler(i_outer)·span +
   atom(i_inner)`, where per dimension the inner segment equals the atom
   strictly iter by iter (extent/stride/axis, tile_tile_ops.cc
   lines 327–336), and the outer segment's strides are divided by the atom's
   address span via `rescale_by_inner_span` (lines 294–303; divisibility is
   checked, and anything not provable is rejected) — i.e. **the tiler's
   strides are in units of "atom count × span"**, and atom origins
   necessarily fall on integer multiples of span (= atom_size elements =
   one swizzle period).

Trust rationale: this implementation is library-wide shared infrastructure,
indirectly corroborated via numerical results by both the gemm and copy
families of GPU tests; this document takes it as a trusted lemma (the sketch
included is already mechanically checkable).

---

## 3. Step-by-Step Lemmas

### L0 (Scoping and Routing)

Lines 403–529: the dense path requires `C_scope=tmem`,
`A ∈ {shared, tmem}`, `B=shared` (lines 416–427, otherwise ValueError);
`C_type=float32` (line 436); `A_sem=B_sem ∈ _DENSE_DTYPES`
(lines 459–475); `cta_group ∈ {1,2}` (lines 515–516).
`descI = config.get("descI", None)` (line 525): **a dense call passing
non-None is rejected outright** (lines 526–529 raise "descI was removed";
negative tests
`test_gemm_tcgen05_dense_descI_rejected` lines 2504–2527,
`test_gemm_tcgen05_dense_descI_rejected_at_dispatch` lines 2787–2806);
block-scaled calls may pass it (lines 1356–1366, the active-use surface of
hoisted-encode + per-ki sf_id rotation, outside the scope of this theorem);
dense with None takes the **self-encoding path** (lines 1375–1396) — the
subject of this document. `weight_stationary` (lines 441–442) changes only
the instruction form (AX-EXEC / AX-MMA.2–.3) and the datapath pairing check
of L7, not the idesc encoding path. ∎

### L1 (Runtime `InstrDescriptor` Packing ≡ AX-IDESC)

The runtime encoder = C bit-field filling (`intrinsics/tcgen05.py`
lines 494–526; bit-field definitions in `intrinsics/header.py`
lines 726–753). Field-by-field comparison against Table 45:

```
sparse_id2_ : 2  bit [0,2)    ← 0            ✓ (Table 45 rows 0–1)
sparse_flag_: 1  bit [2,3)    ← is_sparse    ✓
saturate_   : 1  bit [3,4)    ← sat_d        ✓
c_format_   : 2  bit [4,6)    ← d_format     ✓ (0=F16, 1=F32, 2=S32)
(pad 1)          bit [6,7)      value-init 0 ✓ (Reserved)
a_format_   : 3  bit [7,10)   ← a_format     ✓
b_format_   : 3  bit [10,13)  ← b_format     ✓
a_negate_   : 1  bit [13,14)  ← neg_a        ✓
b_negate_   : 1  bit [14,15)  ← neg_b        ✓
a_major_    : 1  bit [15,16)  ← trans_a      ✓ (0=K-major, 1=MN-major)
b_major_    : 1  bit [16,17)  ← trans_b      ✓
n_dim_      : 6  bit [17,23)  ← N >> 3       ✓
(pad 1)          bit [23,24)    0            ✓
m_dim_      : 5  bit [24,29)  ← M >> 4       ✓
(pad 1)          bit [29,30)    0            ✓
max_shift_  : 2  bit [30,32)  ← 0            ✓ (the "WS not used" comment =
                                               no B-reuse shift; AX-MMA.3
                                               argues its appropriateness)
```

`InstrDescriptor _desc{}` value-initialization zeroes all pad bits
(self-evidenced by the comment at line 502). The runtime wrapper
(lines 529–603) additionally validates the inputs: kind ∈
{f16,tf32,f8f6f4,i8} (lines 562–566), `_check_tcgen05_mma_matrix_shape`
(the M/N/K shape table, lines 426–490), trans-bit dtype legality
(lines 592–595, against the §2.2 legality domain — for the tf32 exception
see L4e). ∎

### L2 (Compile-Time Mirror ≡ Runtime Packing; Domain: Dense Kinds)

`_encode_instr_descriptor_dense_uint32` (lines 91–156) against the L1 bit
shifts, item by item:

```
line 145: (is_sparse & 0x1) << 2      ≡ sparse_flag_
line 146: (sat_d    & 0x1) << 3      ≡ saturate_
line 147: (d_format & 0x3) << 4      ≡ c_format_ (2 bits)
line 148: (a_format & 0x7) << 7      ≡ a_format_ (3 bits)
line 149: (b_format & 0x7) << 10     ≡ b_format_
lines 150–151: neg << 13 / 14         ≡ a/b_negate_
lines 152–153: trans << 15 / 16       ≡ a/b_major_
line 154: ((N >> 3) & 0x3F) << 17    ≡ n_dim_ (6 bits)
line 155: ((M >> 4) & 0x1F) << 24    ≡ m_dim_ (5 bits)
(unset = 0) bits 0–1, 3 pads, 30–31   ≡ sparse_id2_ / pads / max_shift_ = 0
```

Mask widths match the bit-field widths ⇒ no out-of-range contamination. The
format table `_INSTR_DESC_FORMAT_MAP` (lines 66–81) is **key-for-key equal**
to the runtime `format_map` (intrinsics lines 570–584)
(f16:0, bf16:1, tf32/tensor_float32:2, e4m3fn(=fnuz):0, e5m2:1, f6e2m3:3,
f6e3m2:4, f4e2m1:5, u8:0, i8:1, f32:1, i32:2), and consistent with the value
column of Table 45 (d=float32→1=F32 ✓; a/b per the kind column ✓). The two
tables have no shared source, so this key-for-key equality is a load-bearing
invariant; drift on the covered dtypes is pinned only by the idesc
value-pinning test (the 0x04410490 literal assertion). The call
site (lines 1381–1391) fixes `d_dtype="float32"`,
`a/b_dtype = A_sem/B_sem ∈ _DENSE_DTYPES`, and `neg/sat/sparse` default to
False — falling within the legal domain of the L1 validator
(f16/tf32/f8f6f4 kinds; i8 unreachable).

**The mirror performs the runtime's validation**: it reuses the runtime's
validators — `_get_tcgen05_mma_kind` + the kind domain check
(lines 129–134), `_check_tcgen05_mma_matrix_shape` (line 135, including the
shape-table rule N % 16 for cta1 with M_desc=128), and the trans×dtype
legality of `_TCGEN05_MMA_TRANS_DTYPES` (lines 136–139) — so the fold does
not lose the fail-fast layer. `_TCGEN05_MMA_TRANS_DTYPES` is a single set
shared between this compile-time mirror and the runtime encoder (exported
from the intrinsics side, imported at lines 47–51), so the dispatcher and
the runtime cannot drift apart on trans×dtype legality. Regression:
`test_gemm_tcgen05_instr_desc_fold_mirrors_runtime_shape_rules`
(lines 2718–2733). ∎

### L3 (SMEM Descriptor Constructions Mutually Equal ≡ AX-SMEM-DESC)

The four construction paths produce **the same encoded value**:

1. `_make_desc` (lines 1045–1065) and local_hoist's explicit encode
   (lines 1327–1351): call the runtime encoder
   `ptx_tcgen05_encode_matrix_descriptor` (intrinsics/tcgen05.py
   lines 287–317): swizzle 0/1/2/3 ↦ layout_type 0/6/4/2 (≡ Table 43
   bits 61–63), `start = cvta(addr) >> 4`, `base_offset = 0`,
   `version_ = 1` (bits 46–48 = 0b001 ✓), `lbo_mode_ = 0` (bit 52
   relative-offset mode ✓), ldo/sdo filled directly into the 14-bit fields
   (**the caller passes them in 16B units**).
2. `_uniform_desc` (lines 1067–1078): rebuilds the same value with pure bit
   operations: `hi = (sdo & 0x3FFF) | (1 << 14) | (layout << 29)` ⇒
   bits 32–45 = SBO, bit 46 = 1 (the low bit of 0b001, with 47–48 = 0 ✓),
   bits 61–63 = layout ✓;
   `lo = ((ldo & 0x3FFF) << 16) | (cvta(addr + off16·16) >> 4 & 0x3FFF)`
   ⇒ bits 16–29 = LBO, bits 0–13 = start ✓; base_offset bits 49–51 = 0 ✓.
3. `_encoded_desc_val` (lines 1084–1095): per-MMA calls the runtime encoder
   at `base + off16·16` ⇒ same as (1) with start translated.
4. `smem_desc_add_16B_offset` (common.py lines 58–75): `desc.lo += offset`.
   By the lo layout of (1)(2), the start field occupies bits 0–13 of lo;
   SMEM ≤ 228KB ⇒ `start + offset < 2^14` ⇒ the addition does not carry into
   the empty bits 14–15, much less touch the ldo field ⇒ ≡ replacing the
   start field. `_make_lo_uniform` (lines 1033–1043) only performs an
   intra-warp broadcast; the value is unchanged (it is deliberately skipped
   on the local_hoist path — comment at lines 1336–1339 — where only the
   elected thread consumes the descriptor, AX-EXEC).

Hence under any `smem_desc` mode (hoist/local_hoist/encode/recompute,
lines 1113–1139), the descriptor consumed by the (ni,ki)/(mi,ki)-th MMA ≡
the AX-SMEM-DESC encoding
`(start = base + 16·off16, LBO = ldo, SBO = sdo, sw)`. ∎

### L4 (Technical Core: Majorness–Field Co-Origination)

Claim: for every **returning** branch of
`compute_canonical_params(buf, region, dtype, is_transposed)`
(lines 581–853), the `(sw, ldo, sdo, mn_major)` it yields satisfies: once
assembled as `(mn_major → idesc bit, (ldo, sdo, sw) → descriptor fields)`,
the AX-SMEM-WALK read walk reads out `A'/B'` exactly according to the
caller's physical storage.

Preprocessing (lines 708–742): for a physically swizzled buffer, the
innermost-axis region is rounded up to whole atoms (lines 712–735; a
sub-atom start must be 16B-aligned or is rejected, lines 720–734 — the
per-MMA offsets are still taken from the **un-rounded** sliced region, the
descriptor describes only the "whole-atom grid", and the actual [lo:hi) is
addressed by the 16B start offsets of L8; this mechanism is numerically
verified on GPU, slice by slice for 5 slicings, by
`test_gemm_tcgen05_contiguous_kslice_partial_k` lines 2155–2238).
`shape_2d` is the 2D shape after stripping unit dims (lines 737–742).

#### L4a (Swizzled Dual-Atom Matching Branch, Lines 600–691 + 830–832)

For each mode s ∈ {128B, 64B, 32B} (lines 641–645), two candidate atoms are
constructed:

- `swizzle_atom` (lines 646–647) acts on `base_shape = [8, 2^s·E]`:
  row-major `(r, c) ↦ r·2^s·E + c` then swizzle — **exactly the §2.2
  K-major canonical layout restricted to a single atom**
  (`((8,·),(T,2·)) : ((2^s·T, ·), (1, T))`: r stride `2^s·T` ✓, stride 1
  within a 16B unit of c, stride T between units ⇒ composing to a contiguous
  c ✓);
- `mn_atom = Compose(swizzle_atom, [2^s·E, 8]:(1, 2^s·E))`
  (lines 650–654): `(v, r) ↦ v + r·2^s·E` then swizzle — **exactly the
  MN-major canonical layout restricted to a single atom**
  (`((T,2^s,·),(8,·)) : ((1, T, ·), (2^s·T, ·))`: v contiguous ✓,
  r stride `2^s·T` ✓).

`is_transposed` determines the pairing between the candidates and semantic
majorness (lines 669–678 and the comment at 656–668): for non-transposed
`[MN, K]`, K is the last dim ⇒ the last-dim-contiguous atom (swizzle_atom) =
K-major and the first-dim-contiguous atom (mn_atom) = MN-major; for
transposed `[K, MN]` they swap exactly. **The successfully matched candidate
determines both the majorness bit and the field source at once** —
co-origination is guaranteed by construction.

Field derivation (`_try_atom`, lines 606–639): by T-TILE-INNER, a match ⇒
`Φ_L(outer ⊗ inner) = tiler(outer)·atom_size + atom(inner)`; after grouping
the tiler by `tiler_shape = shape_2d / atom_shape`, **each dim must be
exactly a single iter** (lines 613–628, `seps` verified position by
position; non-uniform atom grids are rejected — the comment states
explicitly that this is in the same hazard class as a majorness desync;
regression
`test_gemm_tcgen05_rejects_non_uniform_atom_grid` lines 2759–2784):

- `_atom_off(dim) = dim.stride · atom_size / E` (lines 631–635) converts
  "atom-count units" into 16B units (`stride·atom_size` elements ÷ E
  elements/16B); extent=1 ⇒ 0 (only one group in that direction; the field
  is never touched by hardware).
- `ldo = _atom_off(shard[-1])` (the atom-group spacing along the last dim)
  and `sdo = _atom_off(shard[-2])` (the atom-group spacing along the first
  dim) (lines 637–638); then `if is_mn_major != is_transposed: swap`
  (lines 684–689) re-homes them to "LBO ↔ MN groups / SBO ↔ K groups
  (MN-major); LBO ↔ K groups / SBO ↔ MN row groups (K-major)".

Against AX-SMEM-WALK:

- **K-major**: the atom's vertical extent is exactly 8 rows ⇒ the first-dim
  (MN) atom-group spacing = "first 8 rows → next 8 rows" = SBO ✓.
  K direction: the LBO of swizzled K-major is ignored by hardware
  ("assumed to be 1"); the K footprint of a single MMA = MMA_K elements =
  256b = 2 sixteen-byte units (this holds for every dense kind:
  16·16b = 8·32b = 32·8b = 256b), which is ≤ the atom width `2^s·E`
  elements and divides it, and is positioned by L8's per-tile start ⇒
  contiguity of the 16B units within a row (intra-atom stride T) already
  suffices ✓ (the ldo value becomes a don't-care under this axiom).
- **MN-major**: the atom's vertical extent = `2^s·E` elements =
  swizzle-byte-size/16 rows of 128b ⇒ the first-dim atom-group spacing =
  the LBO definition ✓; the atom is exactly 8 deep along K ⇒ the last-dim
  atom-group spacing = "first 8 columns → next 8 columns" = SBO ✓.
- **swizzle bits**: the structural equality of T-TILE-INNER guarantees that
  the buffer's actual swizzle (pe/s/atom_len/inner, all of them) ≡ the mode
  declared by the descriptor ⇒ hardware XOR ≡ layout XOR; atom origins
  falling on integer multiples of span = period (T-TILE-INNER.2) + the base
  address being period-aligned (a trusted convention, §5) ⇒ the absolutely
  anchored phases agree.

Mechanical verification (host-side pure Python, faithfully transcribing this
branch's logic and evaluating it on the real layouts; for kernel site line
numbers see L9):

| Input | Result `(sw, ldo, sdo, mn_major)` | Check |
|---|---|---|
| head64 O's B: `k_nope_gemm[buf,:,0:256]` (Compose(Swz(3,3,3), 4D tile), transB=T) | `(128B, 512, 64, True)` | LBO: N atom spacing 64·64 elem = 8192B/16 = 512 ✓; SBO: 8 K columns = 512 elem = 1024B/16 = 64 ✓ |
| head64 P-rope's B: bare `SwizzleLayout(3,2,3)` on (128,32), transB=F | `(64B, 0, 32, False)` | single K group ⇒ ldo=0 (ignored) ✓; SBO: 8 rows × 64B = 512B/16 = 32 ✓ |
| head64 P-nope's B: `mma_shared_layout(bf16, 128B)` sliced (128,128), transB=F | `(128B, 1024, 64, False)` | SBO: 8 rows = 1024B/16 = 64 ✓; ldo=1024 is a don't-care ✓ |
| unit-test B: `mma_shared_layout(bf16, 128B, (64,256))`, transB=T | `(128B, 512, 64, True)` | same as the O site ✓ |

(The 128B→64B→32B trial order is lazy: swizzle structural equality means at
most one mode can match.) ∎

#### L4b (No-Swizzle Packed-16B Branch, Lines 748–777)

The structural check (lines 751–760) pins the layout down **literally** as:

```
(d0, k1, k0) ↦ d0·E + k1·(shape_2d[0]·E) + k0 ,   k = k1·E + k0
```

i.e. "16B row lines packed along the first dim; K moves to a new column
block every E elements". This is an instance of the K-major (when
non-transposed; when transposed the first dim is K and the lines run along
MN ⇒ MN-major — `return …, is_transposed`, line 777, isomorphic to L4a's
pairing rule) no-swizzle canonical layout: the 16B units within a row (k0)
are contiguous ✓, row stride = E elements = 16B (canonical `1T`) ✓,
LBO = the spacing between adjacent 16B K-column units = `shape_2d[0]·E`
elements = `shape_2d[0]·16` bytes ⇒ field `ldo = shape_2d[0]` ✓;
SBO = the 8-row-group spacing = a constant 128 bytes ⇒ the field is the
**literal constant 8** (line 777; §9.7.16.3.2). An `sdo` derived as
`elem_per_16B` equals 8 only when E = 8 (16-bit dtypes), so the code
encodes the literal 8 and **explicitly
rejects non-16-bit dtypes** (lines 770–776 — this encoding has only been
hardware-validated in the bf16 domain; reject rather than silently
extrapolate). Anchor:
`test_gemm_tcgen05_no_swizzle_smem_descriptor_codegen[packed_16b]`
(lines 1985–2037) pins `(ldo, sdo, sw) = (64, 8, 0)` (the `[column_major]`
parametrization simultaneously pins the L4c branch producing the same
encoding); negative test
`test_gemm_tcgen05_no_swizzle_packed_16b_rejects_non_16bit_dtype`
(lines 2698–2715). ∎

#### L4c (No-Swizzle Col-Major-View Branch, Lines 779–828 — the Majorness Fix Site)

Checks (lines 779–800): `dim0.stride == 1 ∧ dim1.stride == shape_2d[0]`
(a pure column-major view, lines 779–785) and `shape_2d[0] == 8·E`
(lines 786–800, domain-boundary check — see below).
Returns `(NONE, ldo = dim1.stride, sdo = 8, is_transposed)`
(lines 822–828).

**Semantics (formalization of the fix comment at lines 801–821)**: this view
is a **stride fiction**. The caller contract (the FlashMLA S-tile ABI, which
is also the write pattern of the unit test at lines 2099–2104) is: the
logical `A'[m, k]` is physically stored at

```
phys(m, k) = m·E' + (k div E)·(shape_2d[0]·E) + (k mod E)   (E' = E, 16B-line packing)
```

(bf16, shape_2d[0]=64 instance: `8m + 512·(k div 8) + k mod 8`) —
**element-for-element identical** to L4b's packed-16B layout; the col-major
view is merely a raw memory addressing window in which "view linear offset =
physical element offset" (the writer side uses it exactly so:
`A_smem[a_phys % M, a_phys // M]`), and its (1, M) strides do **not** claim
that `A'[m,k]` lives at `m + M·k`.

The correct descriptor is therefore exactly L4b's K-major encoding, and:

- `ldo = dim1.stride = shape_2d[0]` (line 822): numerically equal to the
  LBO field of the packed ABI (both = shape_2d[0], which holds for any
  dtype/shape because the col-major check forces
  `dim1.stride == shape_2d[0]`) ✓;
- `sdo = 8` (line 827, literal constant): the true SBO field of the packed
  ABI is always 8. The original implementation `sdo = shape_2d[0] // E`
  equals it only when `shape_2d[0] = 8E` (bf16 ⇒ 64 — the precise content
  of the comment at lines 812–815, "coincide exactly when the contiguous
  dim is 64 elements of a 16-bit dtype"); the domain-boundary check at
  lines 786–800 forces `shape_2d[0] = 8E` (out-of-domain rejected),
  and the literal 8 is encoded directly.
- **majorness = `is_transposed` (the body of the fix, line 828)**: the
  physical 16B lines extend along the **last** axis ⇒ non-transposed
  (last axis = K) ⇒ K-major ⇒ bits 15/16 = 0; transposed ⇒ MN-major ⇒ 1.
  Consistent with the pairing rules of L4a/L4b. Before the fix it returned
  `not is_transposed` (HEAD line 724: `return …, not is_transposed`),
  pairing the MN-major bit with K-major fields — historically masked by the
  hand-written `descI` (trans_a=0), and wrong as soon as the fold happened.

**Empirical anchors (twofold)**: (i) FlashMLA head64's original hand-written
descriptors (descA fields identical to this branch + hand-written idesc
trans_a=0) were once verified bit-for-bit across the whole kernel; (ii) the
unit test `test_gemm_tcgen05_no_swizzle_col_major_a_ws_local_idesc`
(lines 2040–2152): pins `", 64, 8, 0)"` (the descA encoding arguments) +
`tcgen05.mma.ws…kind::f16` + the idesc literal `0x04410490` (bit 15 = 0)
and asserts that `0x04418490` (the mis-encoding with bit 15 = 1) does
**not** appear (lines 2133–2139), and on GPU reads back half by half through
the datapath-E fold against a numpy reference (lines 2141–2152).
Domain-boundary negative test:
`test_gemm_tcgen05_no_swizzle_col_major_rejects_non_128B_contiguous`
(lines 2679–2695). ∎

#### L4d (A-in-TMEM Path, Lines 855–858)

When `a_is_tmem`: `assert not transA` (line 857; the hardware has no
transposed-read form for TMEM, AX-MMA.6); `a_mn_major = False` (line 858) ⇒
bit 15 identically 0 — TMEM A has no descriptor and no majorness, and the
K-major encoding is the only legal value (trusted + measured at the head64 P
sites). A's TMEM layout is pinned to structural equivalence with
`(A_dim2, A_dim1):(1@TLane, 1@TCol)` (lines 999–1011); for `_a_operand`'s
address arithmetic see L8. ∎

#### L4e (tf32 Majorness Legality Gate, Lines 867–884)

Table 52 (p667) restricts MN-major for tf32 to the 128B-32B-atomicity
swizzle only (layout_type=1, `_SWIZZLE_TO_LAYOUT[4]`; the matcher never
produces that mode). The dispatch explicitly rejects operands with
`A_sem/B_sem ∈ {tf32, tensor_float32}` that match as MN-major (A side
lines 873–878, B side lines 879–884), so this dispatch cannot encode the
PTX-illegal combination. Residual surface: the runtime validator
`_TCGEN05_MMA_TRANS_DTYPES` contains TENSOR_FLOAT32 (intrinsics
lines 448–459 + 592–595) and lets it through, so block-scaled paths or
callers that reach the runtime encoder directly are not protected by this
gate; all real-world tf32 calls are K-major. Regression:
`test_gemm_tcgen05_rejects_tf32_mn_major` (lines 2736–2756). ∎

### L5 (Dimension Extraction and Cross-Checks)

`M, N` are taken from C (lines 562–563);
`K = A_dim2 if transA else A_dim1` (line 889); the checks are `A_M == M`
(lines 909–911), `K == B_K` (lines 913–915), `B_N · cta_group == N`
(lines 917–921), `K % MMA_K == 0` (line 907). By the "exactly 2 non-unit
dims" assertion of `_mat_dim_vals` (lines 556–560), all four quantities are
well-defined. This step binds the (M, N, K) of Def 1 to the three regions;
any inconsistency is rejected with an AssertionError.
Caveat (caller contract): `_a_offset`/`_b_offset` use `extent[-1]` as the
row width (lines 1098–1102/1106–1110) and the atom rounding uses
`len(region)-1` as the contiguous axis (line 708); a region with a
**trailing** unit dim (e.g. `[M, K, 1]`) passes the 2-non-unit-dims check
yet makes both pick the unit axis — regions must place unit dims only at
the front (`[1, M, K]`, as all real-world sites do). ∎

### L6 (Tile Selection and idesc Shape-Field Legality)

`_choose_mma_tile` (lines 321–348): `M_desc` = the largest element of
{128,64} (cta1) / {256,128} (cta2) that divides `M·cta_group`,
`M_mma = M_desc / cta_group`; `N_mma = N` (if ≤256 and divisible by
MMA_N_MIN), otherwise the largest legal value ≤256 that divides N. Hence
the idesc's `M = M_mma·cta_group ∈ {64,128,256}` (legal m_dim values ✓) and
`N = N_mma ∈ [8, 256]`, a multiple of 8 (legal n_dim values ✓;
MMA_N_MIN=16 for cta2). `M_tiles = M/M_mma`, `N_tiles = N/N_mma`,
`K_iters = K/MMA_K` are all positive integers (divisibility given by
construction/assertions). The deviation from the runtime shape table
(n_step=16 for cta1 with M_desc=128) is covered by L2's mirrored
validation: an illegal (M, N) combination is rejected by
`_check_tcgen05_mma_matrix_shape` already at fold time. For cta_group=2 the
idesc N is the whole-group N (= N_mma), while the B region supplies the
per-CTA N/2 (lines 917–921, L5); the peer CTA's B is fetched by the
hardware from peer SMEM (within the scope of AX-MMA.1, spec text p714) —
this document reads the encoding as the single-CTA projection. ∎

### L7 (C-Side TMEM Footprint ≡ AX-MMA.4 Datapath)

Lines 938–997: C's sliced layout is structurally asserted to be exactly one
of:

- `is_2x2 ∨ packed_n2`: `(M, 2, N/2) : (1@TLane, 64@TLane, 1@TCol)`
  (packed_n2 recognition at lines 946–957; base at lines 977–981) — the
  folded semantics of Layout B (cta2 M_total=128) or **Layout E**
  (cta1 M=64 ws): lane = m + 64·(n ≥ N/2), physical column = n mod (N/2)
  (AX-MMA.4). **This packed-C match is how the M=64 `.ws` path is
  recovered, and it is the site of the SEM-WS-BATCH gap (AX-MMA.4): the
  recognised "2" is treated as an N-half, but a batched caller uses it as
  the batch of a bmm. The match is byte-correct; it does not verify the
  caller's batch semantics.**;
- Layout F (M=64 non-ws, `_layout_matches_datapath_f`,
  lines 351–367 + 982–988);
- otherwise the identity `(M, N) : (1@TLane, 1@TCol)` (Layout D/A)
  (lines 989–990).

`N_mma_phys_cols = N_mma/2 if is_2x2 or packed_n2 else N_mma`
(lines 1200–1203) makes the ni stepping proceed in **physical columns**;
`tmem_col = tmem_offset_32b + ni·N_mma_phys_cols / C_elem_per_32b`
(lines 1309–1311, f32 ⇒ divide by 1), `tmem_row = mi·M_mma` (when
M_tiles>1, line 1312). `_get_tmem_addr_fast` (lines 1149–1160) folds to
`base + col` when row=0 — the taddr encodes the column field in bits 0–15
and TMEM ≤ 512 columns ⇒ no carry overflow. Thus the D footprint of the
(mi, ni)-th MMA = the physical cell set of the C layout restricted to
`[mi·M_mma, (mi+1)·M_mma) × [ni·N_mma, (ni+1)·N_mma)`, matching AX-MMA.4's
write footprint point by point; footprints of distinct (mi, ni) are pairwise
disjoint (the layout is injective).
**The coupling of packed_n2 with `.ws` is checked**: under cta_group::1, a
packed C (Layout-E-shaped) with
`weight_stationary=False` is explicitly rejected (lines 959–975) — non-ws
M=64 writes Layout F ≠ E, so accepting would mean silently misplaced
accumulators (regressions:
`test_gemm_tcgen05_cta1_m64_packed_c_requires_weight_stationary`
lines 2601–2617; the positive pairing is pinned by
`test_gemm_tcgen05_cta1_m64_accepts_packed_c_layout_ws` lines 2579–2598).
The converse (ws + Layout F/identity layout) is unchecked — the dual error
of "ws writes E while the caller reads per F" — and remains a caller
contract. ∎

### L8 (Tiling + Accumulation Chain = GEMM Summation)

`main_impl` (dense non-encode mode: lines 1300–1321; encode_per_mma mode:
lines 1255–1299) triple `T.unroll(M_tiles × N_tiles × K_iters)`:

1. **Operand positioning**: `_a_offset / _b_offset` (lines 1097–1111) take
   the tile origin `(mi·M_mma, ki·MMA_K)` (converted per transA/transB into
   a row-major linear index within the region, with `extent[-1]` as the row
   width), pass it through the **pre-swizzle** sliced tile layout's
   `apply(·)["m"]` (T-APPLY/T-SLICE: includes the region.min translation) to
   obtain the element offset, and `÷ elem_per_16B` lands exactly on 16B
   units (when swizzled, the atom-alignment/16B-alignment checks were
   already imposed before L4; with no swizzle, the tile-origin offset =
   m + shape_2d[0]·k, with k ≡ 0 (mod MMA_K) and M_tiles=1 ⇒ divisible into
   16B — the domain-boundary checks of L4b/L4c pin this domain down; in
   particular the L4c check `shape_2d[0] = 8·E` forces `M = M_mma`, i.e.
   `M_tiles = 1`, so the col-major view's tile origins sit at `m = 0`,
   where the view offset and the packed-ABI offset coincide). By L3, the
   per-MMA descriptor = the
   base descriptor + a start translation ⇒ AX-SMEM-WALK walks with that
   tile origin as its origin, reading out
   `A'[mi·M_mma + ·, ki·MMA_K + ·]` / `B'[ki·MMA_K + ·, ni·N_mma_per_cta + ·]`
   (L4's single-instruction consistency + absolute period anchoring).
   A-in-TMEM: `a_col = A_tmem_offset_32b + ki·(MMA_K / A_elem_per_32b)`
   (line 1167; bf16 ⇒ 8 columns per K step, consistent with the 16-bit TMEM
   packing of §9.7.16.10.4.2).
2. **Accumulation chain**: `should_accum = (ki != 0) ∨ accum_expr`
   (lines 1308, 1013–1021) ⇒ the ki=0 MMA does `D=A·B` or `D+=A·B` per the
   caller's `accum`, and ki>0 always accumulates. The K_iters MMAs of the
   same (mi, ni) share the same accumulator, shape, and cta_group ⇒
   AX-MMA.5 guarantees execution in issue order ⇒ semantics =
   `C_tile (+)= Σ_{ki} A'_tile(ki) · B'_tile(ki)`.
3. **Exactly once**: the `(mi, ni, ki)` enumeration is a box-product
   bijection; the K blocks `[ki·MMA_K, (ki+1)·MMA_K)` partition `[0, K)`
   (divisibility from L6) ⇒ the summands are neither duplicated nor
   dropped; the (mi, ni) footprints are disjoint (L7) ⇒ the composition is
   exactly Def 1.
4. **Execution domain**: `if elect_pred` + the single-thread/warp predicate
   (AX-EXEC) ⇒ the sequence is issued exactly once. ∎

### L9 (Site Instantiation: head64 Fold ≡ the Deleted Hand-Written Values)

The head64 kernel
(`tirx-kernels/tirx_kernels/flashmla/sparse_prefill_head64_phase1.py`,
constants `B_H=64, B_TOPK=64, D_V=512`, lines 18–20; mma shapes
P=(64,128,16) and O=(64,256,16) at the call sites) has two classes of dense
sites (both `weight_stationary=True, cta_group=1`, `descI` defaulted):

**P sites** (lines 913–922 rope, 942–960 nope):
`C = tmem_p[:, :]` ((64,128) f32, datapath-E alloc, line 387 ⇒
packed_n2 ✓), A in TMEM (q_rope_tmem (64,32) / q_nope_tmem slice (64,128),
lines 365–367), B is K-major SMEM (L4a table rows 2/3; `k_rope_tiled_mma`
lines 388–390, `k_nope_tiled_mma` line 314). ⇒ `M_mma=64, N_mma=128,
a_mn_major=False (L4d), b_mn_major=False`:

```
descI_value = 0x10 (d=F32) | 0x80 (a=BF16) | 0x400 (b=BF16)
            | 0<<15 | 0<<16 | (128>>3)<<17 | (64>>4)<<24
            = 0x04200490
```

**O sites** (lines 969–990): `C = tmem_o_lo/hi[:, :]` ((64,256) f32
datapath E, lines 362–363 ⇒ packed_n2 ✓), `A = s_smem_gemm[:, :]`
((64,64) bf16 col-major view, line 391 ⇒ L4c:
`(NONE, 64, 8, mn_major=False)`), `B = k_nope_gemm[buf,:,·]` with
transB=True (k_nope_gemm lines 392–400; L4a table row 1 ⇒ mn_major=True).
⇒ `M_mma=64, N_mma=256, trans_a=0, trans_b=1`:

```
descI_value = 0x10 | 0x80 | 0x400 | 1<<16 | (256>>3)<<17 | 4<<24 = 0x04410490
```

(Both values are verified by pure-Python recomputation;
`0x04410490 | 1<<15 = 0x04418490` is exactly the mis-encoding excluded by
the unit test.)

**Equality with the replaced hand-written values**: the fold supersedes
the kernel's former explicit `Tx.tcgen05_instr_desc(desc_i_p_rope /
desc_i_p_nope / desc_i_o, …)` encodings and their `descI=` arguments.
The hand-written parameters were
P: `(M=64, N=128, K=16, trans_a=False, trans_b=False, n_cta_groups=1)`,
O: `(M=64, N=256, K=16, trans_a=False, trans_b=True, n_cta_groups=1)` —
through the runtime encoder (L1) these produce **exactly**
`0x04200490 / 0x04410490`. Of these, `K` and `n_cta_groups` only enter the
runtime validation and **enter no bits** (§2.1: K is determined by the
kind, cta_group is an instruction qualifier); on the dispatch side the
equivalent information is carried by the `MMA_K` derivation + the `K_iters`
loop (L8) and the pass-through of the `cta_group` qualifier. The fold is
therefore bit-for-bit value-preserving. ∎

---

## 4. Theorems

### Theorem 1 (Acceptance ⟹ Emitted Sequence ≡ GEMM Semantics)

Suppose a dense `Tx.gemm_async` call satisfies:

1. it takes the self-encoding path (not block-scaled; a dense `descI` can no
   longer occur — passing one is rejected, L0) and
   `gemm_async_tcgen05_impl` returns `impl` without exception;
2. the caller contract holds: the regions are within buffer bounds; the
   physical storage of the SMEM operands is consistent with their layout
   declarations — **the col-major-view case is interpreted per L4c's
   packed-16B ABI** (this is a contract, not the literal meaning of the
   declaration: any bijective view is compatible with any physical bytes,
   so the writer side decides the semantics, see L4c); completion
   synchronization (`tcgen05.commit`/mbarrier) is arranged by the caller;
3. the trust base holds (§5); in particular: swizzled buffer base addresses
   are period-aligned (§5 item 4).

Then the side-effect semantics of `impl` is exactly Def 1: every `C[m,n]` is
written exactly once per `Σ_k A'[m,k]·B'[k,n]` (with correct accum
semantics), and there are no TMEM writes outside the C footprint.

**Note (premise discharge)**: the anchored-domain constraints — the
no-swizzle branches limited to 16-bit dtypes, the col-major case limited to
a contiguous dim of 64 elements, and the M=64 packed C layout having to
pair with `.ws` — are unconditionally guaranteed by the dispatch's runtime
rejections (lines 770–776 / 786–800 / 959–975), and are not caller
contracts or theorem premises.

**Proof**: L0 (scoping) → L5/L6 ((M,N,K) and tile legality) → L1+L2 (idesc
bits exact) → L3 (descriptor values exact) → L4 (each MMA's (idesc bits,
descriptor fields) are co-original ⇒ AX-SMEM-WALK reads the correct
operands; L4e excludes the illegal tf32 pairing) → L7 (D footprints match
and are mutually exclusive, including the datapath×ws pairing check) → L8
(tiling bijection + ordered accumulation chain ⇒ summation without
duplication or omission) → AX-EXEC (issued exactly once). ∎

**Empirical note**: the composite proposition is corroborated end-to-end on
GPU by a test matrix: the cta1/cta2/Layout F/arbitrary-tile/K-slicing tests,
plus the ws + col-major-view + local-idesc numerical test
(lines 2141–2152, host expectation = a numpy GEMM re-arranged through the
datapath-E fold).

### Theorem 2 (Fold = Semantics-Preserving Transformation)

For the 4 dense sites of head64, before and after deleting the hand-written
`descI`:

- before: `impl` takes the explicit path (the original implementation's
  `descI is not None` branch), `call_main(descI_hand)`;
- after: `impl` takes the self-encoding path (lines 1375–1396),
  `call_main(descI_const)`.

By L9, the runtime-encoded value of `descI_hand` = `descI_const`,
bit-for-bit equal (0x04200490 / 0x04410490); `call_main` and everything
downstream treat the two identically (both are inlined as the uint32 4th
operand, AX-EXEC); the descriptor construction path is unaffected by the
`descI` key. Hence the instruction sequences of the two programs are
identical instruction by instruction except that "the encoding instructions
themselves are replaced by constants" ⇒ semantics preserved (while saving
the per-call `alignas(64)` local array + inline-asm encoding block, i.e. the
performance motivation claimed by the docstring at lines 106–112).
(Note: the dense explicit path has since been removed entirely —
lines 526–529; the "before"-side program of this theorem can now only be
constructed at the HEAD version, but the equality itself is unaffected.) ∎

### Theorem 3 (Rejection Safety)

Any precondition failure (a raise/assert of L0/L4/L4e/L5/L6/L7) makes this
variant produce no IR; the dispatcher goes on to try other variants and
aggregates an error if all fail (the same mechanism as cp proof Theorem 2).
The only "accepted but contract-dependent" surface is Theorem 1
conditions 2–3 and the caller contracts of the trust base (§5). ∎

### Completeness Discussion (Conservative, Not Incorrect)

Legal combinations that get rejected (the safe direction): non-power-of-two
/ illegal shapes (L6 assertions + L2 mirrored validation); swizzled layouts
that do not tile into whole atoms (the actionable error at lines 837–853);
non-uniform atom grids (lines 613–628); sub-atom K-slice starts not
16B-aligned (lines 720–734); no-swizzle that is neither packed-16B nor a
col-major view (lines 781–785); no-swizzle with a non-16-bit dtype /
contiguous dim ≠ 128B (lines 770–776 / 786–800, domain-boundary
rejections); packed C without `.ws` (lines 966–975); tf32 MN-major
(lines 873–884); `transA ∧ a_is_tmem` (line 857); dtype out of domain
(lines 436/459–475); dense `descI` (lines 526–529).

---

## 5. Trust Base (Summary)

1. **Layout algebra**: T-SLICE / T-APPLY / T-CANON / T-GROUP / T-ARITH
   (cp proof §2.1/§5); T-TILE-INNER (§2.5, including swizzle structural
   congruence, which the cp planner enforces as well).
2. **PTX axioms**: AX-IDESC (Table 45, p639); AX-SMEM-DESC (Tables 43/44,
   p638); AX-SMEM-WALK (§9.7.16.3.1–.3.3 canonical layouts, p629–630;
   Table 55 and the majorness terminology §9.7.16.10.6, p678 / PDF
   page 690; absolute anchoring carried over from the cp proof's
   measurements); AX-MMA (§9.7.16.10 semantics p665;
   `.ws`/collector/zero-column-mask §9.7.16.10.9.3 + Table 48,
   p642/723–725; datapath §9.7.16.10.5 p671–677; pipeline ordering
   §9.7.16.6.2 p646; majorness legality domain Tables 52/54 p667–668).
3. **codegen**: AX-EXEC (asm templates intrinsics lines 736–875, operand
   defaults op.py lines 2450–2543, unroll/meta_var/inline), spot-checked by
   compile-pin tests (`_no_swizzle_smem_descriptor_codegen`,
   `_weight_stationary_codegen` lines 2489–2501,
   `_smem_desc_modes_codegen` lines 2445–2486, the idesc/descA literal
   assertions of the col-major unit test; the former
   `_tx_instr_desc_codegen` was reworked into
   `test_gemm_tcgen05_dense_descI_rejected` along with the dense descI
   removal).
4. **Conventions**: swizzled-buffer base-address period alignment +
   base_offset=0 — the encoder hard-codes base offset = 0 (intrinsics
   lines 308–309); Table 44 absolute anchoring requires swizzled buffer
   base addresses aligned to 1024/512/256B; the dispatch does not check
   base addresses (`alloc_mma`/SMEMPool align by convention), and the
   sub-atom per-MMA starts of L8 rely on the absolute-anchoring axiom.
   (The no-swizzle branch domain boundaries and the M=64-packed-C ⟺ `.ws`
   pairing are unconditionally guaranteed by dispatch rejections and are
   not conventions; see Theorem 1's premise-discharge note.)
5. **Caller contracts**: region validity, with unit dims only in leading
   positions (L5 caveat); the col-major view as a marker for the
   packed-16B ABI (L4c); completion synchronization
   (commit/mbarrier/fence); the block-scaled explicit `descI` path at the
   caller's own risk (the dispatch does not cross-check an explicit
   block-scaled `descI` against the descriptors it builds).
6. **Empirical anchors**: the B200 GPU numerical matrix (end of §2.3); the
   bit-for-bit verification from FlashMLA head64's hand-written-descriptor
   era (L4c/L9); host-side pure-Python recomputation (the L4a table, the
   L9 encoded values).

---

## Appendix A: Code Checkpoint ↔ Lemma Correspondence Table

Every explicit rejection/validation/construction point on the self-encoding
path of `gemm_async/tcgen05.py` (line numbers refer to the current
working tree):

| Lines | Check / construction | Lemma |
|---|---|---|
| 416–427 | scope envelope (C=tmem, A∈{shared,tmem}, B=shared) | L0 |
| 436 | `C_type == float32` | L0 |
| 459–475 | dense dtype domain + `A_sem == B_sem` | L0/L1 domain |
| 515–516 | `cta_group ∈ {1,2}` | L0 |
| 525–529 | dense `descI` rejection ("descI was removed") | L0 |
| 550–552 | C always 2D, A/B ≥ 2D | L5 |
| 556–560 | `_mat_dim_vals` exactly 2 non-unit dims | L5 (trailing-unit-dim caveat) |
| 607–608 | atom divides shape_2d | L4a |
| 610–611 | `is_tile_inner` match (swizzle structural congruence) | L4a / T-TILE-INNER |
| 613–628 | single iter per tiler dim (non-uniform atom grid rejected) | L4a |
| 631–635 | `_atom_off`: extent 1 ⇒ 0 | L4a |
| 637–638, 684–689 | fields taken from shard[-1]/[-2] + majorness swap | L4a |
| 720–734 | sub-atom slice start 16B-aligned or rejected | L4 preprocessing / L8 |
| 739–742 | exactly 2 non-unit dims (desc_region) | L4 |
| 744–747 | no-swizzle must be a plain TileLayout | L4b/c |
| 751–760 | packed-16B structural determination | L4b |
| 770–776 | packed-16B non-16-bit dtype rejected | L4b |
| 777 | packed-16B return, literal `sdo = 8` | L4b |
| 779–785 | col-major determination, else rejected | L4c |
| 786–800 | contiguous dim ≠ 128B (8·E) rejected | L4c |
| 822–828 | `ldo` / literal `sdo = 8` / `return …, is_transposed` (the fix body) | **L4c** |
| 837–853 | no atom match ⇒ actionable error | L4a rejection |
| 857 | `assert not transA` (A in TMEM) | L4d |
| 867–884 | tf32 + MN-major rejected | L4e |
| 891–907 | `MMA_K` derivation + `_choose_mma_tile` + `K % MMA_K` | L6 |
| 909–921 | A_M / B_K / B_N cross-checks | L5 |
| 938–994 | C layout structural assertion (packed_n2/F/identity) | L7 |
| 959–975 | packed C without `.ws` rejected | L7 |
| 995–996 | `allocated_addr` non-null | L7 |
| 999–1011 | TMEM A layout assertion + column-offset conversion | L4d/L8 |
| 1045–1078, 1084–1095, common.py 58–75 | the four descriptor construction paths | L3 |
| 1097–1111 | per-tile 16B offsets | L8 |
| 1149–1160 | taddr folding (no-carry bound) | L7 |
| 1207–1321 | triple loop + should_accum + elect | L8 |
| 1381–1391 | `descI_value` self-encoding | L2 |
| 1430–1444 | execution-domain predicate | AX-EXEC |
| main file 129–139 | mirror reuses runtime validation (kind/shape/trans) | L2 |
| intrinsics 562–599 | runtime validation (kind/shape/trans/negate/saturate) | L1 (tf32 residual surface, L4e) |
| intrinsics 764–769 | ws × sparse/cta2/scale rejection | AX-EXEC |

Emission side (no rejection, construction only): 1113–1147 (desc modes),
1163–1173 (the A operand), 1325–1396 (call_main / impl selection) → L8/L2.
Block-scaled (lines 1367–1374, and its `descI` pass-through at
lines 1356–1366) is outside the scope of these theorems (the former has its
own runtime encoder + SF validation; the latter is a
compatibility-retained surface — the dense-side explicit `descI` path
has been deleted).
