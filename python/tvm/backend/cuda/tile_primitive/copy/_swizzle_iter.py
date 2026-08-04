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

"""Generic swizzle-aware iter pattern for CUDA copy dispatches.

The swizzle map sigma(q) = q ^ ((q>>at) & (2^sw - 1)) of a
``ComposeLayout(per_element=p, swizzle_len=sw, atom_len=at,
swizzle_inner=True)`` is GF(2)-linear and additive over high bits
(>= 2^(at+sw) chunks), so the swizzled physical address at iter ``k``
reduces to

    addr(k) = (base_off + D_high) ^ sigma(D_low)     [element units]

where ``base_off = swizzle.apply(s_off)`` is computed once per thread and
sigma(D_low) is a compile-time constant — one XOR per iter instead of a full
``swizzle.apply(...)``. Verified bitwise for the whole swizzle family
(128B/64B/32B/NONE; sw = 3/2/1/0).

Notation. Each binary outer iter has element-stride ``2^(bj + p)`` for
some chunk bit position ``bj >= 0`` (so ``stride / C = 2^bj`` where
``C = 1 << p``). Conditions for the fast path:

  (C1) bit-clear no-carry: ``bit_bj(q(M0)) = 0`` for every binary iter —
       the enumeration from the per-thread base must be carry-free (XOR
       and ADD coincide) at every bit any iter flips.

  (distinctness) The ``bj`` values across all binary iters must be
       distinct — two iters at the same ``bj`` collapse into bit
       ``bj + 1`` whose behavior may differ.

GF(2)-linearity absorbs inner-outer iter pairs exactly, so no
support-disjointness condition is needed.

The ``swizzle_inner=False`` mode swaps the inner/outer roles and is not
yet covered; ``try_recognize`` gates on this.
"""

from dataclasses import dataclass

import tvm
from tvm import arith
from tvm.tirx.expr import IntImm as _IntImm
from tvm.tirx.layout import ComposeLayout, S, TileLayout


@dataclass
class _BitIter:
    """Pow2-extent outer iter, binary-split into ``n_bits`` chunk-bit flips.

    ``slot_start..slot_start + n_bits`` is this iter's range in the global
    ``bit_positions`` / ``iter_strides_elems`` arrays. Slot
    ``slot_start + b`` corresponds to bit position ``n_bits - 1 - b`` of
    this iter's per-iter coord (outermost binary bit first).
    """

    ext: int
    n_bits: int
    slot_start: int


@dataclass
class _LinearIter:
    """Outer iter contributing ``c * stride`` to the offset (no bit decomp).

    Used when ``stride`` is a multiple of ``2^(p + at + sw)`` (pure Case 1.D
    regime: swizzle XOR has no effect on bits the iter flips). ``ext`` does
    not need to be a power of two.
    """

    ext: int
    stride: int


@dataclass
class SwizzlePattern:
    """A recognized swizzle iter pattern.

    ``bit_positions[j]`` and ``iter_strides_elems[j]`` collect the binary
    sub-iters from every BitIter in outer-iter order (outermost first).
    ``outer_iters`` lists every outer iter (BitIter or LinearIter) in
    outermost-first order; the emit functions walk this list to decompose
    the per-iter coord. Empty lists = trivially recognized degenerate case
    (no outer iter, just base_off).
    """

    swizzle: ComposeLayout
    bit_positions: list[int]
    iter_strides_elems: list[int]
    outer_iters: "list[_BitIter | _LinearIter]"

    @property
    def n_binary_iters(self) -> int:
        return len(self.bit_positions)


def get_swizzle(layout) -> ComposeLayout | None:
    """Return a bare-swizzle view of ``layout`` if it is swizzled, else ``None``.

    The result carries the swizzle params and an identity tile over the swizzle
    period, so ``.apply()`` reproduces the bare swizzle XOR and
    ``.per_element`` / ``.swizzle_len`` / ``.atom_len`` / ``.swizzle_inner``
    read the params directly.
    """
    if isinstance(layout, ComposeLayout):
        p = int(layout.per_element)
        sw = int(layout.swizzle_len)
        at = int(layout.atom_len)
        period = 1 << (p + sw + at)
        return ComposeLayout(p, sw, at, TileLayout(S[(period,)]), bool(layout.swizzle_inner))
    return None


def _is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _recognize_common(
    swizzle: ComposeLayout,
    iter_extents: list[int],
    iter_strides: list[int],
    s_off_template,
    var_bounds: dict | None = None,
):
    """Recognition core of ``try_recognize``.

    Checks: swizzle_inner, per-iter stride validity, pow2 binary split,
    distinctness of chunk-bit positions, and (C1) bit-clear no-carry on the
    per-thread base. Returns ``(bit_positions, iter_strides_elems,
    outer_iters)`` or ``None``.
    """
    # swizzle_inner=False swaps the inner/outer xor direction — Cases 1.A
    # and 1.C roles flip. Not derived/tested yet; reject for safety.
    if not swizzle.swizzle_inner:
        return None

    p = swizzle.per_element
    C = 1 << p
    # Pure Case 1.D threshold: stride a multiple of this means every chunk-bit
    # the iter flips is at position >= at + sw (above the swizzle XOR region),
    # so swizzle has no effect and the contribution is purely linear in the
    # iter coord — no power-of-2 ext requirement.
    pure_1d = 1 << (p + swizzle.atom_len + swizzle.swizzle_len)

    bit_positions: list[int] = []
    iter_strides_elems: list[int] = []
    outer_iters: list = []

    for ext, stride in zip(iter_extents, iter_strides):
        if ext == 1:
            # Trivial iter contributes nothing (coord always 0); skip without
            # any stride requirement — its (placeholder) stride may be 0.
            continue
        # Zero-stride iters degrade dq=0 → log2 undefined. Explicit guard.
        if stride == 0 or stride % C != 0:
            return None
        if ext <= 0:
            return None
        if not _is_pow2(ext):
            # Non-pow2 ext can only be handled by the linear path. That in turn
            # requires the iter to be in pure Case 1.D (stride a multiple of
            # the swizzle period) so the swizzle does not interact with the
            # per-coord contribution.
            if stride % pure_1d != 0:
                return None
            outer_iters.append(_LinearIter(ext=ext, stride=stride))
            continue
        # pow2 ext: binary split (existing path).
        k = ext.bit_length() - 1  # log2(ext)
        slot_start = len(bit_positions)
        # Split into k binary iters; the outermost (within this split) carries
        # the largest stride so that flat-index bit decomp matches our
        # outer-iter list ordering.
        for j in range(k - 1, -1, -1):
            substride = stride * (1 << j)
            dq = substride // C
            # dq must be a single bit set (so this binary iter flips exactly
            # one bit of the chunk index). _is_pow2 also rejects dq=0.
            if not _is_pow2(dq):
                return None
            bj = dq.bit_length() - 1
            # All bj >= 0 accepted; case branching happens in xor_delta.
            bit_positions.append(bj)
            iter_strides_elems.append(substride)
        outer_iters.append(_BitIter(ext=ext, n_bits=k, slot_start=slot_start))

    # Distinctness: two binary iters at the same bj collapse to bj+1, whose
    # case behavior may differ from bj. See module docstring NB.
    if len(set(bit_positions)) != len(bit_positions):
        return None

    # (C1) per-iter no-carry on q(M0). Must hold *symbolically over all*
    # free lane / warp placeholders in s_off_template — ``can_prove_equal``
    # returns False if the analyzer can't discharge the equality
    # universally, conservatively forcing a fallback.
    #
    # These are compile-time SMEM-offset proofs (values stay far below
    # 2**32), so unsigned index terms are analyzed in the no-overflow
    # domain; every uint expression admitted is logged once by the analyzer.
    analyzer = arith.Analyzer()
    if var_bounds:
        for var, rng in var_bounds.items():
            analyzer.bind(var, rng)
    with arith.allow_uint_as_index():
        for bj in set(bit_positions):
            divisor = C * (1 << bj)
            check = tvm.tirx.floormod(
                tvm.tirx.floordiv(s_off_template, _IntImm(s_off_template.expr_ty().dtype, divisor)),
                _IntImm(s_off_template.expr_ty().dtype, 2),
            )
            if not analyzer.can_prove_equal(check, _IntImm(s_off_template.expr_ty().dtype, 0)):
                return None

    return bit_positions, iter_strides_elems, outer_iters


def _mk_pattern(swizzle, core) -> SwizzlePattern:
    bit_positions, iter_strides_elems, outer_iters = core
    return SwizzlePattern(
        swizzle=swizzle,
        bit_positions=bit_positions,
        iter_strides_elems=iter_strides_elems,
        outer_iters=outer_iters,
    )


def try_recognize(
    swizzle: ComposeLayout,
    iter_extents: list[int],
    iter_strides: list[int],
    s_off_template,
    var_bounds: dict | None = None,
) -> SwizzlePattern | None:
    """Return a ``SwizzlePattern`` if (C1)+(distinctness) hold, else ``None``.

    ``iter_extents`` / ``iter_strides``: the outer-iter list on the S side
    (excluding T iter and vec iter), in outermost-first order matching
    ``s_p.shard[:-2]`` (or the atom-derived analog in ``vec_auto_reg.py``).
    Strides are in element units.

    Each outer iter with ``extent=2^k`` and ``stride=s`` is conceptually
    split into ``k`` binary iters of strides ``2^(k-1)*s, ..., 2*s, s``
    (outermost first within the split — this matches ``_flat_outer_coords``
    semantics, since the highest-stride iter must change slowest in the
    flat-index decomposition).

    ``s_off_template`` is the per-thread linear base offset expression
    (with a placeholder var for the thread-id contribution). It is used
    only to check condition (C1) symbolically via ``arith.Analyzer``;
    ``emit_base`` takes the resolved form separately.

    ``var_bounds`` is an optional ``{Var: tvm.ir.Range}`` map of placeholder
    bounds to ``analyzer.bind`` before the (C1) check. Without bounds,
    structurally-OK forms like ``(lane // 8) * 8 + (lane % 8) * Q`` where
    ``lane < 32`` make ``(... // (C·2^bj)) % 2 == 0`` unprovable — the
    bit is in fact always 0 but the analyzer can't conclude it universally.
    Pass ``{lane_ph: Range(0, 32), warp_ph: Range(0, n_warps)}`` (or the
    scope's equivalents) to let the (C1) check fire on these templates.

    No support-disjointness condition is needed: sigma is GF(2)-linear, so
    inner-outer iter pairs (``bj_A`` + ``bj_A + at``) cancel exactly inside
    sigma(delta).
    """
    core = _recognize_common(swizzle, iter_extents, iter_strides, s_off_template, var_bounds)
    if core is None:
        return None
    return _mk_pattern(swizzle, core)


def emit_fallback_offset(swizzle: ComposeLayout, s_off_resolved, ds_k):
    """Slow but always-correct path: full ``swizzle.apply(s_off + ds_k)``
    per iter. Use when ``try_recognize`` returns ``None``.

    ``ds_k`` is the outer-iter delta for unrolled iter k — typically a
    Expr (a function of the unroll var that simplifies to a constant
    after unrolling) or a Python int. ``s_off_resolved`` is the per-thread
    base linear offset with the real tid Var substituted.
    """
    return swizzle.apply(s_off_resolved + ds_k)["m"]


def xor_delta(swizzle: ComposeLayout, delta_chunks: int) -> tuple[int, int]:
    """Split a compile-time chunk-domain delta into (low_xor_chunks, high_add_chunks).

    Physical address = ``(base + high * 2^p) ^ (low * 2^p)`` in element units.
    ``low = sigma(delta mod 2^(at+sw))``; the high part is swizzle-invariant and
    stays additive.
    """
    at = int(swizzle.atom_len)
    sw = int(swizzle.swizzle_len)
    thr = 1 << (at + sw)
    low = delta_chunks % thr
    high = delta_chunks - low
    low = low ^ ((low >> at) & ((1 << sw) - 1))
    return low, high


def emit_base(swizzle: ComposeLayout, s_off_resolved):
    """``base_off = swizzle.apply(s_off_resolved)`` — runtime, per-thread,
    computed once."""
    return swizzle.apply(s_off_resolved)["m"]


def emit_xor_offset(pattern: SwizzlePattern, base_off, k: int):
    """Per-iter physical S offset = ``(base_off + D_high) ^ xor_const``.

    ``k`` must be a Python int (parse-time unrolled iter); for a TIR-expr
    iter use ``emit_xor_offset_var`` instead. Returns a PrimExpr; folds to
    ``base_off`` unchanged when the iter contributes no delta.
    """
    assert isinstance(k, int), "emit_xor_offset requires a compile-time iter index"
    if not pattern.outer_iters:
        return base_off

    swizzle = pattern.swizzle
    p = int(swizzle.per_element)

    # Decompose k innermost-first across outer_iters and accumulate the
    # element-domain delta.
    delta_elems = 0
    remaining = k
    for it in reversed(pattern.outer_iters):
        ext = it.ext
        c = remaining % ext
        remaining //= ext
        if isinstance(it, _LinearIter):
            delta_elems += c * it.stride
            continue
        for b in range(it.n_bits):
            bit_pos = it.n_bits - 1 - b
            slot = it.slot_start + b
            if (c >> bit_pos) & 1:
                delta_elems += pattern.iter_strides_elems[slot]
    assert remaining == 0
    assert delta_elems % (1 << p) == 0

    low, high = xor_delta(swizzle, delta_elems >> p)
    off = base_off
    if high:
        off = off + _IntImm("int32", high << p)
    if low:
        off = off ^ _IntImm("int32", low << p)
    return off


def emit_xor_offset_var(pattern: SwizzlePattern, base_off, k):
    """Var-k counterpart of ``emit_xor_offset`` (TIR-expr iter index).

    Decomposes ``k`` across ``pattern.outer_iters`` with floordiv/floormod
    and emits, for each binary iter j with chunk stride ``stride_j``:

        off ^= bit_j(k) * sigma(stride_j)    (XOR part, sigma compile-time)

    GF(2)-linearity makes ``sigma(sum_j bit_j(k)*stride_j) = XOR_j sigma(stride_j)``
    exact, so each set bit selects its compile-time sigma constant — ~2
    instructions per bit.
    ``_LinearIter`` contributes ``c * stride`` additively as usual.
    """
    if not pattern.outer_iters:
        return base_off

    swizzle = pattern.swizzle
    p = int(swizzle.per_element)

    off = base_off
    remaining = k
    for it in reversed(pattern.outer_iters):  # innermost first
        ext = it.ext
        c = tvm.tirx.floormod(remaining, _IntImm("int32", ext))
        remaining = tvm.tirx.floordiv(remaining, _IntImm("int32", ext))
        if isinstance(it, _LinearIter):
            off = off + c * _IntImm("int32", it.stride)
            continue
        for b in range(it.n_bits):
            bit_pos = it.n_bits - 1 - b
            slot = it.slot_start + b
            low, high = xor_delta(swizzle, pattern.iter_strides_elems[slot] >> p)
            # bit = bit_pos-th bit of c, via floordiv/floormod (real tirx
            # nodes that the analyzer and downstream constant-folding can
            # fold after unrolling — the shift/and Call forms stay opaque).
            bit = tvm.tirx.floormod(
                tvm.tirx.floordiv(c, _IntImm("int32", 1 << bit_pos)),
                _IntImm("int32", 2),
            )
            if high:
                off = off + bit * _IntImm("int32", high << p)
            if low:
                off = off ^ (bit * _IntImm("int32", low << p))
    return off
