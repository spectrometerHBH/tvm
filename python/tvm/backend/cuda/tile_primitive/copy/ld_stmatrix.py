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

"""copy dispatch variant: ldmatrix / stmatrix (TBD algorithm).

Handles register ↔ shared copies on CUDA via PTX ``ldmatrix`` / ``stmatrix``.
Direction (ld vs st) and exec scope (warp / warpgroup) are decided inside
``_emit`` from the src/dst scopes and ``sctx.scope_kind``.
"""

from math import prod

import tvm
from tvm import arith
from tvm.script import tirx as T
from tvm.tirx import PrimFunc
from tvm.tirx import Var as _TirVar
from tvm.tirx.expr import IntImm as _IntImm
from tvm.tirx.layout import ComposeLayout, S, TileLayout
from tvm.tirx.operator.tile_primitive.dispatcher import fail, predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.tile_primitive import TilePrimitiveCall

from ._common import (  # noqa: F401  (_carve_tail reserved for future variants)
    _carve_tail,
    _extract_tile,
)
from ._swizzle_iter import (
    emit_base,
    emit_init,
    emit_iter_offset,
    emit_xor_offset,
    get_swizzle,
    try_recognize_xor,
    xor_delta,
)
from .utils import _is_valid_copy, _scope_allowed
from .vec_auto_reg import _all_threads_active, _ptr_off

_REG_SMEM_PAIRS = [
    ("local", "shared*"),
    ("shared*", "local"),
]

_VALID_R_LANE_AXES = {"laneid", "tid_in_wg", "tx"}


def _compute_r_perm(r):
    """Permutation: thread iters first (stride-desc), then memory iters (stride-desc)."""

    def key(p):
        it = p[1]
        return (0 if it.axis.is_thread() else 1, -int(it.stride))

    return [i for i, _ in sorted(enumerate(r.shard), key=key)]


def _is_ldstmatrix(op_call: TilePrimitiveCall, sctx: DispatchContext) -> tuple[bool, str | None]:
    if not sctx.is_target("cuda"):
        return False, "non-cuda target"
    if sctx.scope_kind not in ("warp", "warpgroup", "cta"):
        return False, f"unsupported exec_scope {sctx.scope_kind} (need warp, warpgroup, or cta)"
    for check in (
        lambda: _all_threads_active(sctx),
        lambda: _is_valid_copy(op_call, sctx),
        lambda: _scope_allowed(op_call, sctx, allowed_pairs=_REG_SMEM_PAIRS),
    ):
        ok, msg = check()
        if not ok:
            return False, msg
    return True, None


def _emit(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    op_call = TilePrimitiveCall.downcast(op_call)

    # Step 1: identify reg / smem sides and pull their tensor shape + layout.
    src_br = op_call.src
    dst_br = op_call.dst
    if src_br.buffer.scope() == "local":
        r_br, s_br = src_br, dst_br
        direction = "st"  # reg -> smem (stmatrix)
    else:
        r_br, s_br = dst_br, src_br
        direction = "ld"  # smem -> reg (ldmatrix)
    r_buf = r_br.buffer
    s_buf = s_br.buffer
    r_shape = list(r_buf.shape)
    r_layout = r_buf.layout
    s_shape = list(s_buf.shape)
    s_layout = s_buf.layout

    r_region = [(r.min, r.min + r.extent) for r in r_br.region]
    s_region = [(r.min, r.min + r.extent) for r in s_br.region]
    with sctx.target:
        r_sliced = r_layout.slice(r_shape, r_region)
        s_sliced = s_layout.slice(s_shape, s_region)
        r = r_sliced.canonicalize()
        s = s_sliced.canonicalize()

    # Step 2.5: peel any S-side swizzle wrapper to expose the underlying
    # TileLayout. The ComposeLayout doesn't have ``.replica`` / ``.shard``,
    # so we must peel *before* the structural checks below. Capture the
    # swizzle separately for use at emit time.
    # NB: read swizzle from the *buffer* layout, not the post-canon ``s`` --
    # the buffer layout is the reliable source for the swizzle params.
    s_swizzle = get_swizzle(s_buf.layout)
    if s_swizzle is not None and s_swizzle.per_element < 3:
        # ldmatrix/stmatrix .b16 reads/writes 8 fp16 = 128b per lane in one
        # contiguous chunk. The swizzle preserves the lowest ``per_element``
        # bits of the address (in-chunk offset). For the per-lane 128b unit
        # to stay contiguous post-swizzle, ``2^per_element >= 8`` ⇒ p >= 3.
        fail(
            f"swizzle per_element={s_swizzle.per_element} < 3 incompatible "
            f"with .b16 ldmatrix/stmatrix (need 8-fp16 chunk integrity)"
        )
    s = _extract_tile(s, s_region)

    # Step 3: ldstmatrix doesn't broadcast — require zero replica on both sides.
    if len(r.replica) != 0:
        fail(f"R layout has replica {list(r.replica)}; ldstmatrix requires no replica")
    if len(s.replica) != 0:
        fail(f"S layout has replica {list(s.replica)}; ldstmatrix requires no replica")

    # Step 4: R must have exactly one kind of lane axis from the valid set.
    r_thread_axes = {it.axis.name for it in r.shard if it.axis.is_thread()}
    if len(r_thread_axes) != 1:
        fail(f"R must have exactly one thread axis name; got {sorted(r_thread_axes)}")
    r_lane_axis = next(iter(r_thread_axes))
    if r_lane_axis not in _VALID_R_LANE_AXES:
        fail(f"R thread axis {r_lane_axis!r} not in {sorted(_VALID_R_LANE_AXES)}")

    # Step 5: group S by R's iter extents (one S group per R iter, outer→inner).
    r_group_shape = [int(it.extent) for it in r.shard]
    s_grp, s_seps = s.group(r_group_shape)

    # Step 6: permute R so thread iters come first (stride-desc), then memory
    # iters (stride-desc).
    r_perm = _compute_r_perm(r)
    r = r.permute_dims(r_perm)

    # Step 7: apply R's perm to S in group units (1-to-1 with R's iters), and
    # rebuild s_seps to track group boundaries in the new order.
    s = s_grp.permute_by_groups(list(s_seps), r_perm)
    old_sizes = [s_seps[i + 1] - s_seps[i] for i in range(len(s_seps) - 1)]
    s_seps = [0]
    for pi in r_perm:
        s_seps.append(s_seps[-1] + old_sizes[pi])

    # Step 7.5: canonicalize both R and S after permute. Fuses adjacent
    # contig iters — keeps step 8's group input clean. Push target so
    # scope-aware fusers run (laneid+wid_in_wg → tid_in_wg, etc.).
    with sctx.target:
        r = r.canonicalize()
        s = s.canonicalize()

    t_total = prod(int(it.extent) for it in r.shard if it.axis.is_thread())
    m_total = prod(int(it.extent) for it in r.shard if not it.axis.is_thread())
    if t_total % 32 != 0:
        fail(f"R thread section total {t_total} not divisible by 32")

    def _strs(lay, seps):
        # Atoms 8 / 4 / 2 (segs 1, 2, 5) must be single iters — their strides
        # feed downstream stride checks (lane partition + fragment 2-fp16
        # contig). The num atom (seg 4) may be MULTI-ITER: we return its iter
        # list and let layout.apply handle the decomposition at emit time.
        fixed_segs = [list(lay.shard[seps[i] : seps[i + 1]]) for i in (1, 2, 5)]
        if not all(len(g) == 1 for g in fixed_segs):
            return None
        num_iters = list(lay.shard[seps[4] : seps[5]])
        return (
            int(fixed_segs[0][0].stride),  # 8 atom stride
            int(fixed_segs[1][0].stride),  # 4 atom stride
            num_iters,  # num atom iter list (multi-iter OK)
            int(fixed_segs[2][0].stride),  # 2 atom stride
        )

    def _try_num(r_in, s_in, num):
        """Try grouping (r_in, s_in) with [T/32, 8, 4, M/(2num), num, 2].

        Returns (rg, rsep, sg, ssep, trans, p, num) if structural checks pass,
        else None. ``trans`` is the ldmatrix .trans flag; ``p`` is the
        per-tile-row S stride used at emit.
        """
        gs = [t_total // 32, 8, 4, m_total // (num * 2), num, 2]
        try:
            rg, rsep = r_in.group(gs)
            sg, ssep = s_in.group(gs)
        except Exception:
            return None
        # R seg 0 (T/32 outer): require single iter with stride 32. When
        # T/32 == 1 the segment is trivial — skip.
        if t_total > 32:
            seg0 = list(rg.shard[rsep[0] : rsep[1]])
            if len(seg0) != 1 or int(seg0[0].stride) != 32:
                return None
        rs, ss = _strs(rg, rsep), _strs(sg, ssep)
        if rs is None or ss is None:
            return None
        r8, r4, _r_num_iters, r2 = rs
        s8, s4, s_num_iters, s2 = ss
        if (r8, r4, r2) != (4, 1, 1):
            return None
        # S num atom: every iter must have stride > 0 and multiple of 8 (the
        # per-tile spacing geometry of ldmatrix m8n8; 8 fp16 = 16 bytes = one
        # tile column dimension).
        if num > 1 and not all(
            int(it.stride) > 0 and int(it.stride) % 8 == 0 for it in s_num_iters
        ):
            return None
        # m_outer (seg 3) iters: each per-mm advance must keep the per-lane
        # SMEM address 16-byte aligned (ldmatrix .b16 reads 8 fp16 = 16 bytes
        # per lane), so the m_outer S-stride must also be a multiple of 8.
        # Without this, mm > 0 iterations land at unaligned addresses and
        # silently read garbage even though the layout group succeeds.
        # Skip extent-1 trivial iters — they contribute no per-mm advance,
        # so their (placeholder) stride is irrelevant.
        m_outer_iters = list(sg.shard[ssep[3] : ssep[4]])
        if not all(int(it.extent) == 1 or int(it.stride) % 8 == 0 for it in m_outer_iters):
            return None
        if (s4, s2) == (2, 1) and s8 > 0 and s8 % 8 == 0:
            return (rg, rsep, sg, ssep, False, s8, num)
        if s8 == 1 and s2 > 0 and s2 % 8 == 0 and s4 == 2 * s2:
            return (rg, rsep, sg, ssep, True, s2, num)
        return None

    # Try the **sorted** variant: 5D-group, sub-group R's M/2 by S's M/2
    # extents, sort the sub-groups by descending S-stride, rebuild. This
    # makes the m_outer iter list carry the largest S-strides on top, which
    # maximizes the §2 swizzle fast-path applicability later. If anything
    # in the rebuild raises (e.g. M/2 can't be sub-grouped by S's extents),
    # we silently fall back to the no-sort path below.
    r_sort = s_sort = None
    try:
        gs5 = [t_total // 32, 8, 4, m_total // 2, 2]
        rg5, rsep5 = r.group(gs5)
        sg5, ssep5 = s.group(gs5)
        r_m_iters = list(rg5.shard[rsep5[3] : rsep5[4]])
        s_m_iters = list(sg5.shard[ssep5[3] : ssep5[4]])
        s_m_extents = [int(it.extent) for it in s_m_iters]
        # Sub-group R's M/2 iters by S's M/2 iter extents. This 1-to-1's
        # the R sub-groups with the S iters so we can permute them together.
        r_m_sub = TileLayout.from_iters(r_m_iters)
        r_m_grouped, r_m_seps = r_m_sub.group(s_m_extents)
        # Sort S iters by S-stride descending; permute R sub-groups in lockstep.
        perm = sorted(range(len(s_m_iters)), key=lambda i: -int(s_m_iters[i].stride))
        if perm != list(range(len(perm))):
            r_m_permuted = r_m_grouped.permute_by_groups(list(r_m_seps), perm)
            s_m_permuted = [s_m_iters[i] for i in perm]
            r_sort = TileLayout.from_iters(
                list(rg5.shard[: rsep5[3]])
                + list(r_m_permuted.shard)
                + list(rg5.shard[rsep5[4] :]),
                offset=dict(rg5.offset),
            )
            s_sort = TileLayout.from_iters(
                list(sg5.shard[: ssep5[3]]) + list(s_m_permuted) + list(sg5.shard[ssep5[4] :]),
                offset=dict(sg5.offset),
            )
        # If perm is identity, sorted == unsorted; no need to build duplicate layouts.
    except Exception:
        r_sort = s_sort = None

    # Enumerate num largest-first; for each num try sorted then unsorted.
    chosen = None
    for num in (4, 2, 1):
        if m_total % (num * 2):
            continue
        if r_sort is not None:
            res = _try_num(r_sort, s_sort, num)
            if res is not None:
                chosen = res
                break
        res = _try_num(r, s, num)
        if res is not None:
            chosen = res
            break

    if chosen is None:
        fail("ldstmatrix layout doesn't fit any num ∈ {4,2,1}")
    r, r_seps, s, s_seps, trans, p, num = chosen

    # Order the num atom (seg 4) iters by R m-stride descending. The dst of
    # each ldmatrix/stmatrix must be 4 consecutive registers; when the seg-4
    # decomposition order matches R's fragment word order (largest m-stride
    # slowest), the frag words land in consecutive dst registers and ptxas
    # needs no register-shuffle MOVs. R and S are permuted in lockstep, so
    # the R↔S element pairing (and thus numerics) is unchanged.
    r_num_iters = list(r.shard[r_seps[4] : r_seps[5]])
    if len(r_num_iters) > 1:
        num_order = sorted(range(len(r_num_iters)), key=lambda i: -int(r_num_iters[i].stride))
        if num_order != list(range(len(r_num_iters))):
            r_iters = list(r.shard)
            s_iters = list(s.shard)
            r_seg = [r_iters[r_seps[4] : r_seps[5]][i] for i in num_order]
            s_seg = [s_iters[s_seps[4] : s_seps[5]][i] for i in num_order]
            r = TileLayout.from_iters(
                r_iters[: r_seps[4]] + r_seg + r_iters[r_seps[5] :],
                offset=dict(r.offset),
            )
            s = TileLayout.from_iters(
                s_iters[: s_seps[4]] + s_seg + s_iters[s_seps[5] :],
                offset=dict(s.offset),
            )

    # Step 10: emit one ldmatrix/stmatrix per mm, per warp.

    def _get_warp_idx_in_T():
        # T.warp_id_in_wg() / T.warp_id() must be called from inside a
        # @T.prim_func body — wrap so the prim_func parser calls us at parse
        # time (Python `if` here is plain control flow, not TIR-intercepted).
        if r_lane_axis == "laneid":
            return 0
        if r_lane_axis == "tid_in_wg":
            return T.warp_id_in_wg()
        return T.warp_id()  # "tx"

    def _seg4_coord(laneid_expr):
        # num=1: seg 4 trivially extent-1, pass 0. num>1: use lane//8 (tile
        # index in ldmatrix lane convention); layout.apply decomposes through
        # the seg's iter structure (single or multi-iter).
        if num > 1:
            return laneid_expr // 8
        return 0

    apply_shape = [t_total // 32, 8, 4, m_total // (num * 2), num, 2]
    r_mem_axis = r.shard[r_seps[5]].axis.name
    s_mem_axis = s.shard[s_seps[5]].axis.name
    m_outer = m_total // (num * 2)
    s_zero = [0] * len(s_buf.shape)

    # Swizzle fast-path setup. When S is swizzled, the per-mm `tile_off +
    # row_off` is a logical offset; the physical SMEM address is
    # `swizzle.apply(logical)`. The slow path computes that per iter; the
    # fast path (§2.E of the swizzle-iter plan) reduces it to
    # `base_off + sum_j bit_j(mm) · signed_strides[j]` where base_off and
    # signed_strides are per-thread constants set once. We try to
    # recognize the m_outer iter list as such a pattern; if it fails (e.g.
    # the analyzer can't discharge condition C1 over the lane/warp
    # placeholders) we silently fall through to the slow path.
    swizzle_pattern = None
    s_off_template = None
    lane_ph = warp_ph = None
    if s_swizzle is not None:
        m_outer_iters = list(s.shard[s_seps[3] : s_seps[4]])
        iter_extents = [int(it.extent) for it in m_outer_iters]
        iter_strides = [int(it.stride) for it in m_outer_iters]
        # Build s_off at mm=0 with placeholder vars for lane and warp_idx.
        lane_ph = _TirVar("lane_ph", "int32")
        seg4_ph = (lane_ph // 8) if num > 1 else _IntImm("int32", 0)
        if r_lane_axis == "laneid":
            warp_ph_expr = _IntImm("int32", 0)
        else:
            warp_ph = _TirVar("warp_ph", "int32")
            warp_ph_expr = warp_ph
        s_off_template = s.apply(
            warp_ph_expr,
            _IntImm("int32", 0),
            _IntImm("int32", 0),
            _IntImm("int32", 0),
            seg4_ph,
            _IntImm("int32", 0),
            shape=apply_shape,
        )[s_mem_axis] + (lane_ph % 8) * _IntImm("int32", p)
        # Bind lane / warp placeholder bounds for the (C1) analyzer. ``lane_ph``
        # is the per-warp lane id ∈ [0, 32); ``warp_ph`` (when present) is the
        # warp index inside the scope: warpgroup ⇒ [0, 4), cta ⇒ [0, t_total/32).
        var_bounds = {lane_ph: tvm.ir.Range.from_min_extent(0, 32)}
        if warp_ph is not None:
            var_bounds[warp_ph] = tvm.ir.Range.from_min_extent(0, t_total // 32)
        swizzle_pattern = try_recognize_xor(
            s_swizzle,
            iter_extents,
            iter_strides,
            s_off_template,
            var_bounds=var_bounds,
        )

    class _SwizzleState:
        def __init__(self):
            self.signed_strides = None
            self.base_off = None
            self.s_off_resolved = None

    def _split_carry_free(split_tmpl, low_pre, p_):
        """Numeric carry-free check for the constant split.

        Every chunk-bit set in ``low_pre`` must be 0 in the split base's
        chunk index, for every (lane, warp) in scope. Kernel-scalar buffer
        reads (``prep_stage`` etc.) are sealed to 0 — they only contribute
        bits above the checked range (stage * 20992 and pair_base * 64 start
        at bit 6 while checks are at bits <= 8).
        """
        from tvm.tirx.buffer import BufferType as _BufferType
        from tvm.tirx.expr_functor import ExprMutator as _ExprMutator
        from tvm.tirx.expr_functor import ExprVisitor as _ExprVisitor
        from tvm.tirx.stmt_functor import substitute as _subst_expr

        free: dict[str, object] = {}
        scalar_bufs: dict[str, object] = {}

        class _VarGrab(_ExprVisitor):
            def visit_var_(self, op):
                if isinstance(op.ty, _BufferType):
                    scalar_bufs.setdefault(str(op), op)
                else:
                    free.setdefault(str(op), op)

            def visit_buffer_load_(self, op):
                if isinstance(op.buffer.ty, _BufferType):
                    scalar_bufs.setdefault(str(op.buffer), op.buffer)
                super().visit_buffer_load_(op)

        class _SealReads(_ExprMutator):
            def __init__(self, val_of):
                super().__init__()
                self._val_of = val_of

            def visit_buffer_load_(self, op):
                if op.buffer in self._val_of:
                    return _IntImm(str(op.expr_ty()), self._val_of[op.buffer])
                return super().visit_buffer_load_(op)

        _VarGrab().visit_expr(split_tmpl)
        scalar_bufs = list(scalar_bufs.values())
        seal = _SealReads({b: 0 for b in scalar_bufs})
        an = arith.Analyzer()
        n_warps = max(1, t_total // 32)
        for b in range(32):
            if not (low_pre >> b) & 1:
                continue
            for lane_i in range(32):
                for warp_i in range(n_warps):
                    subs = {lane_ph: _IntImm("int32", lane_i)}
                    if warp_ph is not None:
                        subs[warp_ph] = _IntImm("int32", warp_i)
                    v = an.simplify(seal.visit_expr(_subst_expr(split_tmpl, subs)))
                    if not isinstance(v, _IntImm):
                        return False
                    if (int(v.value) >> (p_ + b)) & 1:
                        return False
        return True

    state = _SwizzleState()

    # ------------------------------------------------------------------
    # Region-min constant split: keep compile-time constants out of the
    # swizzle apply input. When the sliced offset carries a compile-time
    # constant Δ (e.g. the k0 region min of a chain of copies), emit
    #
    #     addr = (apply(X) + D_high) ^ σ(D_low)      (element units)
    #
    # with X = s_off − Δ. Every copy in the chain then emits a *textually
    # identical* apply(X) plus a per-copy XOR constant, and nvcc's plain
    # CSE merges them into one base computation + one XOR per copy — the
    # same shape the handwritten .cu relies on. Valid iff X is carry-free
    # at Δ's low chunk-bits (checked numerically below); otherwise the
    # behavior is unchanged (correct by construction regardless — the
    # split only changes codegen shape, never addresses).
    # ------------------------------------------------------------------
    def _region_min_delta():
        """Compile-time constant in the smem offset attributable to IntImm
        region mins, from the S layout's per-dim shard strides (positionally
        grouped to tensor dims)."""
        base_lay = s_layout.tile_layout if isinstance(s_layout, ComposeLayout) else s_layout
        iters = list(base_lay.shard)
        groups = []
        i = 0
        for t_ext in s_shape:
            acc = []
            prod_e = 1
            while prod_e < int(t_ext) and i < len(iters):
                it = iters[i]
                acc.append((int(it.extent), int(it.stride)))
                prod_e *= int(it.extent)
                i += 1
            if prod_e != int(t_ext):
                return 0
            groups.append(acc)
        if i != len(iters):
            return 0
        delta = 0
        for (r_min, _), acc in zip(s_region, groups):
            if not isinstance(r_min, _IntImm):
                continue
            m = int(r_min.value)
            for sub_ext, stride in reversed(acc):
                delta += (m % sub_ext) * stride
                m //= sub_ext
            if m:
                return 0
        return delta

    split_low_elems = 0
    split_high_elems = 0
    s_off_template_base = s_off_template
    if (
        s_swizzle is not None
        and s_off_template is not None
        and swizzle_pattern is not None
        and not swizzle_pattern.outer_iters
    ):
        d_elems = _region_min_delta()
        p_ = int(s_swizzle.per_element)
        if d_elems > 0 and d_elems % (1 << p_) == 0:
            d_chunks = d_elems >> p_
            low_c, high_c = xor_delta(s_swizzle, d_chunks)
            low_pre = d_chunks % (1 << (int(s_swizzle.atom_len) + int(s_swizzle.swizzle_len)))
            cand_template = s_off_template - _IntImm("int32", d_elems)
            if _split_carry_free(cand_template, low_pre, p_):
                split_low_elems = low_c << p_
                split_high_elems = high_c << p_
                s_off_template_base = cand_template

    def _resolve_s_off(laneid_var, warp_var):
        # Build the placeholder→runtime-var map and substitute. Keep this in a
        # regular Python helper — the @T.prim_func parser intercepts dict
        # literals when written directly in the body.
        vmap = {lane_ph: laneid_var}
        if warp_ph is not None:
            vmap[warp_ph] = warp_var
        return tvm.tirx.stmt_functor.substitute(s_off_template_base, vmap)

    def _setup_swizzle(s_off_resolved):
        if swizzle_pattern is None:
            return
        state.s_off_resolved = s_off_resolved
        state.base_off = emit_base(s_swizzle, s_off_resolved)

    def _apply_split(off):
        # Append the region-min constant back as one XOR (the split form).
        if split_high_elems:
            off = off + _IntImm("int32", split_high_elems)
        if split_low_elems:
            off = off ^ _IntImm("int32", split_low_elems)
        return off

    def _smem_off(mm_idx, logical_off):
        # Three paths:
        #   * pattern matched + compile-time mm: physical off =
        #     (base_off + D_high) ^ σ(D_low) — one XOR, no signed_strides.
        #   * pattern matched + runtime mm: additive signed-strides (lazy).
        #   * swizzle present, pattern missed: per-iter swizzle.apply(logical).
        #   * no swizzle: identity.
        if swizzle_pattern is not None and isinstance(mm_idx, int):
            return _apply_split(emit_xor_offset(swizzle_pattern, state.base_off, mm_idx))
        if swizzle_pattern is not None:
            if state.signed_strides is None:
                state.signed_strides, _ = emit_init(swizzle_pattern, state.s_off_resolved)
            return _apply_split(
                emit_iter_offset(
                    swizzle_pattern,
                    state.signed_strides,
                    state.base_off,
                    mm_idx,
                )
            )
        if s_swizzle is not None:
            return s_swizzle.apply(logical_off)["m"]
        return logical_off

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        r_local = r_buf.local(m_total, layout=TileLayout(S[(m_total,)]))
        laneid = T.lane_id()
        warp_idx_in_T = _get_warp_idx_in_T()
        # Resolve s_off_template by substituting placeholders → actual
        # scope-id vars (via _resolve_s_off helper to keep the dict literal
        # out of the parser's view). Only the swizzle fast path needs this;
        # without swizzle we keep using the per-iter s.apply directly.
        if swizzle_pattern is not None:
            _setup_swizzle(_resolve_s_off(laneid, warp_idx_in_T))
        for mm in T.unroll(m_outer):
            tile_off = s.apply(
                warp_idx_in_T, 0, 0, mm, _seg4_coord(laneid), 0, shape=apply_shape,
            )[s_mem_axis]
            row_off = (laneid % 8) * p
            logical_off = tile_off + row_off
            smem_ptr = _ptr_off(s_buf.ptr_to(s_zero), _smem_off(mm, logical_off))
            handles = [
                r_local.ptr_to([
                    r.apply(0, 0, 0, mm, i, 0, shape=apply_shape)[r_mem_axis]
                ])
                for i in range(num)
            ]
            if direction == "ld":
                T.ptx.ldmatrix(trans, num, ".b16", smem_ptr, *handles)
            else:
                T.ptx.stmatrix(
                    trans, num, ".b16", smem_ptr, *handles,
                    shape="m8n8", space="shared",
                )
    # fmt: on
    return impl


@register_dispatch(
    "copy",
    "cuda",
    variant="ldstmatrix",
    priority=10,
    when=[predicate("ldstmatrix_applicable", _is_ldstmatrix)],
)
def copy_schedule_ldstmatrix(op_call: TilePrimitiveCall, sctx: DispatchContext) -> PrimFunc:
    return _emit(op_call, sctx)


__all__ = ["copy_schedule_ldstmatrix"]
