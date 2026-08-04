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

"""Tests for the generic swizzle-aware iter pattern in
``cuda/copy/_swizzle_iter.py`` (XOR emit form).

Two layers:

* **Recognizer tests** check that ``try_recognize`` returns the expected
  ``SwizzlePattern`` (or rejects) under (C1)+(distinctness).
* **Numeric correctness tests** verify the XOR formula empirically: for
  many ``(M0, k)`` samples,
  ``(apply(M0) + D_high) ^ sigma(D_low)``
  equals ``apply(M0 + ds_k)`` computed by the layout's own Apply formula —
  including bases whose mask-source bits toggle per thread (the case the
  old additive signed-strides needed runtime signs for; the GF(2)-linear
  XOR form absorbs them exactly).

All algorithm-level (no GPU needed). End-to-end emit is tested in
``test_gmem_smem.py::test_swizzled_smem_emit_must_be_swizzle_aware``.
"""

import pytest

import tvm
from tvm import arith
from tvm.tirx import Var as _TirVar
from tvm.tirx.cuda.tile_primitive.copy._swizzle_iter import (
    SwizzlePattern,
    _BitIter,
    _LinearIter,
    emit_xor_offset,
    emit_xor_offset_var,
    get_swizzle,
    try_recognize,
)
from tvm.tirx.expr import IntImm as _IntImm
from tvm.tirx.layout import ComposeLayout, S, TileLayout
from tvm.tirx.stmt_functor import substitute as _substitute

# ----------------------------------------------------------------------------
# Pure-Python reference: the bare-swizzle ComposeLayout's Apply, plus the XOR
# formula. Used as ground truth — both must agree for the math to hold.
# ----------------------------------------------------------------------------


def py_swizzle_apply(M: int, p: int, sw: int, at: int) -> int:
    """Pure-Python reimplementation of the swizzle Apply (swizzle_inner=True):
    phys = swz_q * C + (M mod C)
    q = M / C; swz_q = q XOR ((q & outer_mask) >> at)
    """
    C = 1 << p
    q = M // C
    outer_mask = ((1 << sw) - 1) << at
    swz_q = q ^ ((q & outer_mask) >> at)
    return swz_q * C + (M % C)


def py_xor_offset(base_off: int, delta_elems: int, p: int, sw: int, at: int) -> int:
    """The XOR formula: (base_off + D_high) ^ sigma(D_low), element units."""
    C = 1 << p
    assert delta_elems % C == 0
    d_chunks = delta_elems // C
    thr = 1 << (at + sw)
    low = d_chunks % thr
    high = d_chunks - low
    low = low ^ ((low >> at) & ((1 << sw) - 1))
    return (base_off + high * C) ^ (low * C)


def py_outer_ds(k: int, iter_extents: list[int], iter_strides: list[int]) -> int:
    """Decode a flat outer index k into per-iter coords (matching
    _flat_outer_coords) and sum coord_i * stride_i for the corresponding ds."""
    coords: list[int] = []
    rem = k
    for ext in reversed(iter_extents):
        coords.append(rem % ext)
        rem //= ext
    coords.reverse()
    return sum(c * s for c, s in zip(coords, iter_strides))


# ----------------------------------------------------------------------------
# Recognizer tests — verify try_recognize accepts / rejects under
# (C1)+(distinctness).
# ----------------------------------------------------------------------------


def test_get_swizzle_extracts_from_compose():
    sw = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
    assert get_swizzle(sw) is not None
    assert (
        get_swizzle(
            ComposeLayout(
                sw.per_element,
                sw.swizzle_len,
                sw.atom_len,
                TileLayout(S[(64, 64)]),
                sw.swizzle_inner,
            )
        )
        is not None
    )
    assert get_swizzle(TileLayout(S[(64, 64)])) is None


def test_recognize_nvfp4_case():
    """nvfp4's epilogue: swizzle(3,3,3), iter extents [2,2,2] strides
    [8,16,32], M0 = tid * 64 (each thread starts at col 0 of one row;
    row_stride 64 = 8 chunks, ensures chunk bits of M0/C are zero for all
    iter bit positions)."""
    sw = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
    tid = _TirVar("tid", "int32")
    # M0 = tid * 64 → M0/C = tid * 8 → bits 0,1,2 are 0 (since multiplied by 8).
    M0 = tid * _IntImm("int32", 64)
    pat = try_recognize(sw, [2, 2, 2], [8, 16, 32], M0)
    assert pat is not None
    assert pat.bit_positions == [0, 1, 2]
    assert pat.iter_strides_elems == [8, 16, 32]
    assert pat.n_binary_iters == 3


def test_recognize_binary_split():
    """A single outer iter with extent=4 stride=8 splits into two binary
    iters with strides 16 and 8 (outermost first, matching _flat_outer_coords)."""
    sw = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
    tid = _TirVar("tid", "int32")
    M0 = tid * _IntImm("int32", 64)
    pat = try_recognize(sw, [4], [8], M0)
    assert pat is not None
    # Split: stride 8*2 = 16 (outermost), stride 8 (innermost) → bits [1, 0]
    assert pat.bit_positions == [1, 0]
    assert pat.iter_strides_elems == [16, 8]


def test_recognize_mid_bits():
    """swizzle(p=4, sw=2, at=4): an iter at bj=2 lives in the mid range
    [sw, at) — accepted; its delta is swizzle-invariant (sigma = identity)."""
    sw = ComposeLayout(4, 2, 4, TileLayout(S[(1024,)]))  # C=16, mid covers bits 2..3
    tid = _TirVar("tid", "int32")
    # row_stride = 256 (= 16*C) → M0/C = tid*16 → bits 0..3 all 0.
    M0 = tid * _IntImm("int32", 256)
    pat = try_recognize(sw, [2], [64], M0)  # stride 64 = C * 2^2 → bj=2 (mid)
    assert pat is not None
    assert pat.bit_positions == [2]
    assert pat.iter_strides_elems == [64]


def test_reject_not_chunk_aligned():
    """Stride must be a multiple of C."""
    sw = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))  # C=8
    tid = _TirVar("tid", "int32")
    M0 = tid * _IntImm("int32", 64)
    # stride 4 is not a multiple of C=8 → reject.
    assert try_recognize(sw, [2], [4], M0) is None


def test_reject_carry_into_masked_bit():
    """(C1): a binary iter at bj=3 needs bit 3 of M0/C == 0 universally.
    M0 = tid*64 → M0/C = tid*8 → bit 3 = bit 0 of tid — not provably 0."""
    sw = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
    tid = _TirVar("tid", "int32")
    M0 = tid * _IntImm("int32", 64)
    assert try_recognize(sw, [2, 2, 2, 2], [8, 16, 32, 64], M0) is None


def test_reject_chunk_overlap():
    """(C1): (M0/C) must have 0 bits at all iter-bit positions per thread.
    M0 = tid * 8 → M0/C = tid → bit 0 NOT provably zero → reject."""
    sw = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))  # C=8
    tid = _TirVar("tid", "int32")
    M0 = tid * _IntImm("int32", 8)
    assert try_recognize(sw, [2], [8], M0) is None


def test_recognize_no_outer_iters():
    """Degenerate case: no outer iter at all. Recognizer returns a trivial
    pattern (empty bit_positions). Emit will use base_off alone."""
    sw = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
    tid = _TirVar("tid", "int32")
    M0 = tid * _IntImm("int32", 64)
    pat = try_recognize(sw, [], [], M0)
    assert pat is not None
    assert pat.n_binary_iters == 0


def test_recognize_inner_outer_pair_accepted():
    """Inner-outer iter pair (bj_A in [0,at) and bj_A + at in [at,at+sw)):
    the GF(2)-linear XOR form absorbs the pair's secondary contribution
    exactly, so the recognizer must ACCEPT (the old additive signed-strides
    encoding had to reject these to avoid double-counting)."""
    sw = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
    # bj=0 (stride 8) and bj=3 (stride 64): pair (0, 0+3).
    # (C1): bits {0, 3} of M0/C must be 0 ⇒ M0 multiple of 128.
    pat = try_recognize(sw, [2, 2], [8, 64], _IntImm("int32", 128))
    assert pat is not None
    assert pat.bit_positions == [0, 3]


# ----------------------------------------------------------------------------
# Numeric correctness — the XOR formula must equal apply(M0 + ds_k) for all
# sampled (M0, k), including bases with toggling mask-source bits.
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "p,sw,at,iter_extents,iter_strides,row_stride",
    [
        # nvfp4-like (p=sw=at=3, 3 binary iters covering one swizzle row)
        (3, 3, 3, [2, 2, 2], [8, 16, 32], 64),
        # single binary iter at chunk_bit position
        (3, 3, 3, [2], [8], 64),
        # split-from-extent-4 (one outer becomes two binary)
        (3, 3, 3, [4], [8], 64),
        # mid_bits region
        (4, 2, 4, [2], [64], 256),
        # mix: one chunk_bit + one mid_bit
        (3, 2, 4, [2, 2], [8, 32], 256),
        # inner-outer pair (accepted only by the XOR form)
        (3, 3, 3, [2, 2], [8, 64], 128),
    ],
)
def test_formula_matches_apply_under_conditions(
    p,
    sw,
    at,
    iter_extents,
    iter_strides,
    row_stride,
):
    """For every (M0, k) sample, the XOR formula must equal
    py_swizzle_apply(M0 + ds_k). Sweeps multiple per-thread M0 values so
    mask-source bits toggle — a formula that only worked for zero mask
    bits would fail here."""
    swizzle = ComposeLayout(p, sw, at, TileLayout(S[(1 << (p + sw + at),)]))
    tid = _TirVar("tid", "int32")
    M0_template = tid * _IntImm("int32", row_stride)
    pat = try_recognize(swizzle, iter_extents, iter_strides, M0_template)
    assert pat is not None, (
        f"recognizer rejected supposedly-valid case p={p},sw={sw},at={at} "
        f"iter_extents={iter_extents} iter_strides={iter_strides} row_stride={row_stride}"
    )

    total_iters = 1
    for ext in iter_extents:
        total_iters *= ext

    for tid_val in [0, 1, 3, 5, 7, 13, 21]:
        M0 = tid_val * row_stride
        base_off = py_swizzle_apply(M0, p, sw, at)
        for k in range(total_iters):
            ds_k = py_outer_ds(k, iter_extents, iter_strides)
            ground_truth = py_swizzle_apply(M0 + ds_k, p, sw, at)
            formula = py_xor_offset(base_off, ds_k, p, sw, at)
            assert formula == ground_truth, (
                f"formula mismatch: p={p},sw={sw},at={at} "
                f"iter_extents={iter_extents} iter_strides={iter_strides} "
                f"tid={tid_val} M0={M0} k={k} ds_k={ds_k} "
                f"apply(M0+ds_k)={ground_truth} formula={formula}"
            )


def test_mask_source_bit_toggle_needs_no_sign():
    """The case the old additive form needed a runtime ±1 sign for: an inner
    iter whose mask-source bit (at + bj) toggles with tid. The XOR form uses
    one compile-time sigma for every base and is still exactly right."""
    p, sw, at = 3, 3, 3
    row_stride = 64  # M0/C = tid * 8 → mask-source bit (at + 0) = bit 0 of tid
    swizzle = ComposeLayout(p, sw, at, TileLayout(S[(1 << (p + sw + at),)]))
    iter_extents, iter_strides = [2], [8]
    results = []
    for tid_val in (0, 1):
        M0 = tid_val * row_stride
        base_off = py_swizzle_apply(M0, p, sw, at)
        for k in range(2):
            ds_k = py_outer_ds(k, iter_extents, iter_strides)
            truth = py_swizzle_apply(M0 + ds_k, p, sw, at)
            formula = py_xor_offset(base_off, ds_k, p, sw, at)
            assert formula == truth
            results.append((tid_val, k, formula))
    # The toggle is real: the two tids' offsets actually differ in the low bits.
    assert (results[0][2] ^ results[2][2]) != 0


# ----------------------------------------------------------------------------
# TIR-level emit: emit_base + emit_xor_offset (int k) + emit_xor_offset_var
# (Var k) must evaluate to the ground truth after substitution.
# ----------------------------------------------------------------------------


def _eval(e, env):
    """Tiny recursive evaluator for the emitted offset exprs. The analyzer
    can't fold ``tirx.bitwise_*`` Call forms, so we evaluate directly."""
    t = type(e).__name__
    if t == "IntImm":
        return int(e.value)
    if t == "Var":
        return env[e]
    if t in ("Add", "Sub", "Mul", "FloorDiv", "FloorMod"):
        a, b = _eval(e.a, env), _eval(e.b, env)
        if t == "Add":
            return a + b
        if t == "Sub":
            return a - b
        if t == "Mul":
            return a * b
        if t == "FloorDiv":
            return a // b
        return a % b
    if t == "Cast":
        return _eval(e.value, env)
    if t == "Call":
        args = [_eval(a, env) for a in e.args]
        name = str(e.op.name)
        if name == "tirx.bitwise_xor":
            return args[0] ^ args[1]
        if name == "tirx.bitwise_and":
            return args[0] & args[1]
        if name == "tirx.shift_right":
            return args[0] >> args[1]
        if name == "tirx.shift_left":
            return args[0] << args[1]
        if name in ("tirx.add", "tirx.Add"):
            return args[0] + args[1]
        if name in ("tirx.multiply", "tirx.Mul"):
            return args[0] * args[1]
        raise AssertionError(f"cannot eval Call op {name}")
    raise AssertionError(f"cannot eval node type {t}")


def _make_pattern(p, sw, at):
    swizzle = ComposeLayout(p, sw, at, TileLayout(S[(1 << (p + sw + at + 4),)]))
    period = 1 << (p + at + sw)
    return (
        swizzle,
        SwizzlePattern(
            swizzle=swizzle,
            bit_positions=[2, 1],
            iter_strides_elems=[32, 16],
            outer_iters=[
                _LinearIter(ext=3, stride=period),
                _BitIter(ext=4, n_bits=2, slot_start=0),
            ],
        ),
        period,
    )


@pytest.mark.parametrize("p,sw,at", [(3, 3, 3), (3, 2, 3), (3, 1, 3), (3, 0, 3)])
def test_emit_xor_offset_int_k(p, sw, at):
    """emit_xor_offset with a Python-int k must fold to the ground truth."""
    swizzle, pat, period = _make_pattern(p, sw, at)
    an = arith.Analyzer()
    C = 1 << p
    for base in range(0, 2048, 64):
        if base % 4:
            continue  # LOW delta bits {1,2} must be clear in base chunks
        swz_base = py_swizzle_apply(base, p, sw, at)
        for k in range(12):
            lin, bits = divmod(k, 4)
            delta = lin * period + bits * 16
            truth = py_swizzle_apply(base + delta, p, sw, at)
            got = an.simplify(emit_xor_offset(pat, swz_base, k))
            assert hasattr(got, "value") and int(got.value) == truth, (
                f"p={p},sw={sw},at={at} base={base} k={k} truth={truth} got={got}"
            )


@pytest.mark.parametrize("p,sw,at", [(3, 3, 3), (3, 2, 3), (3, 1, 3)])
def test_emit_xor_offset_var_k(p, sw, at):
    """emit_xor_offset_var with a TIR-Var k must evaluate to the ground
    truth for every concrete k after substitution."""
    swizzle, pat, period = _make_pattern(p, sw, at)
    k_var = _TirVar("k", "int32")
    for base in range(0, 2048, 64):
        if base % 4:
            continue
        swz_base = py_swizzle_apply(base, p, sw, at)
        expr = emit_xor_offset_var(pat, swz_base, k_var)
        for k in range(12):
            got = _eval(_substitute(expr, {k_var: _IntImm("int32", k)}), {})
            lin, bits = divmod(k, 4)
            truth = py_swizzle_apply(base + lin * period + bits * 16, p, sw, at)
            assert got == truth, f"p={p},sw={sw},at={at} base={base} k={k} truth={truth} got={got}"


# ----------------------------------------------------------------------------
# Recognizer structure — LinearIter (pure Case 1.D) and its rejection.
# ----------------------------------------------------------------------------


def test_recognize_linear_iter_pure_case_1d():
    """Outer iter with non-pow2 ext is accepted IF its stride is a multiple
    of the swizzle period 2^(p+at+sw) (pure Case 1.D, swizzle has no XOR
    effect). The iter is stored as a LinearIter (no bit decomposition).
    """
    p, sw, at = 3, 3, 3
    swizzle = ComposeLayout(p, sw, at, TileLayout(S[(1 << (p + sw + at),)]))
    period = 1 << (p + at + sw)  # 512
    # Outer iter (ext=3, stride=period) — non-pow2 but pure Case 1.D.
    # Inner iter (ext=2, stride=8) — pow2, Case 1.A (bj=0).
    pat = try_recognize(swizzle, [3, 2], [period, 8], _IntImm("int32", 0))
    assert pat is not None
    assert len(pat.outer_iters) == 2
    # Outermost (index 0) corresponds to first input iter = the linear one.
    assert isinstance(pat.outer_iters[0], _LinearIter)
    assert pat.outer_iters[0].ext == 3
    assert pat.outer_iters[0].stride == period
    # Innermost (index 1) is the binary-split iter.
    assert isinstance(pat.outer_iters[1], _BitIter)
    assert pat.outer_iters[1].ext == 2
    assert pat.outer_iters[1].n_bits == 1
    assert pat.outer_iters[1].slot_start == 0
    # bit_positions / iter_strides_elems only contain the binary iter's bit.
    assert pat.bit_positions == [0]  # 8/8 = 2^0
    assert pat.iter_strides_elems == [8]


def test_reject_non_pow2_ext_not_case_1d():
    """Non-pow2 ext where stride is NOT in pure Case 1.D regime — reject.
    stride=64 = 2^(p+at) = one atom row, which is in [at, at+sw) territory
    and interacts with the XOR mask, so the linear path is unsafe."""
    swizzle = ComposeLayout(3, 3, 3, TileLayout(S[(512,)]))
    pat = try_recognize(swizzle, [3], [64], _IntImm("int32", 0))
    assert pat is None


def test_emit_mixed_linear_bit_correctness():
    """Brute-force: for a mixed (LinearIter outer, BitIter inner) pattern,
    the XOR formula must equal the actual swizzle output for every (tid, k)
    — including the non-pow2 outer extent's coord 2."""
    p, sw, at = 3, 3, 3
    swizzle = ComposeLayout(p, sw, at, TileLayout(S[(1 << (p + sw + at),)]))
    period = 1 << (p + at + sw)  # 512
    iter_extents, iter_strides = [3, 2], [period, 8]
    tid = _TirVar("tid", "int32")
    # Inner iter bj=0 in [0, sw); (C1) needs bit_0(M0/C) = 0 ⇒ M0/C even
    # ⇒ M0 multiple of 16. So row_stride = 16.
    M0_template = tid * _IntImm("int32", 16)
    pat = try_recognize(swizzle, iter_extents, iter_strides, M0_template)
    assert pat is not None

    total_k = iter_extents[0] * iter_extents[1]
    for tid_val in [0, 1, 5, 7, 13]:
        M0 = tid_val * 16
        base_off = py_swizzle_apply(M0, p, sw, at)
        for k in range(total_k):
            ds_k = py_outer_ds(k, iter_extents, iter_strides)
            ground_truth = py_swizzle_apply(M0 + ds_k, p, sw, at)
            formula = py_xor_offset(base_off, ds_k, p, sw, at)
            assert formula == ground_truth, (
                f"mixed mismatch: tid={tid_val} M0={M0} k={k} ds_k={ds_k} "
                f"truth={ground_truth} formula={formula}"
            )


def test_fallback_path_when_recognizer_rejects():
    """The recognizer should reject when (C1) fails, and the resulting
    fallback emit (swizzle.apply per iter) is the correct path. This test
    proves the rejection and demonstrates that the swizzled offset really
    differs from the linear offset for the rejected case — so a buggy
    `linear-offset-without-XOR` emit would give the wrong answer on at
    least one (tid, k) sample. The fallback emit, by construction,
    delegates to swizzle.apply and is thus correct."""
    p, sw, at = 3, 3, 3
    swizzle = ComposeLayout(p, sw, at, TileLayout(S[(1 << (p + sw + at),)]))
    tid = _TirVar("tid", "int32")
    M0_template = tid * _IntImm("int32", 8)  # (C1) fails: bit 0 of M0/C = bit 0 of tid
    pat = try_recognize(swizzle, [2], [8], M0_template)
    assert pat is None, "recognizer must reject when (C1) fails"

    # Demonstrate the swizzled offset differs from linear for at least one
    # (tid, k) — proves the swizzle is actually non-trivial here.
    iter_extents, iter_strides = [2], [8]
    diverging_samples = 0
    for tid_val in range(16):
        M0 = tid_val * 8
        for k in range(2):
            ds_k = py_outer_ds(k, iter_extents, iter_strides)
            linear = M0 + ds_k
            swizzled = py_swizzle_apply(linear, p, sw, at)
            if swizzled != linear:
                diverging_samples += 1
    assert diverging_samples > 0, (
        "no (tid, k) sample shows swizzled != linear — the swizzle is a "
        "no-op for this layout, so the test isn't catching anything"
    )


if __name__ == "__main__":
    tvm.testing.main()
