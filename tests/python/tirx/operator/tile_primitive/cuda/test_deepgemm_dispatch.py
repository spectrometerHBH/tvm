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
# pylint: disable=missing-function-docstring
"""Unit tests for the tile-primitive dispatch changes the fp4/fp8/tf32 deepgemm
kernels depend on (see docs/deepgemm/dispatch_changes.md in tirx-kernels):

* dense fp8 + tf32 (``is_AB_tf32``) tcgen05 ``gemm_async``               (§1)
* TFLOAT32 TMA descriptor (``tma_dtype="tf32"``)                        (§2)
* ``maximum`` elementwise CUDA dispatch                                 (§4)
* uint32 TMEM-column divisibility proof (``_can_prove_divisible``)      (§6)

§3 (split-laneid ``.16x*b`` reg↔smem deposit) and §5 (B00011 split-laneid
canonicalize fix) are layout/canonicalize internals exercised end-to-end by the
``tf32_hc_prenorm_gemm`` cast warp; they are covered by that kernel's 26/26
numerical-correctness integration check (tirx-kernels), not a standalone unit
test here.
"""

import functools
import operator

import numpy as np
import pytest

try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None

import tvm
import tvm.testing
from tvm.backend.cuda.operator.tile_primitive.tma_utils import mma_shared_layout
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import S, TCol, TileLayout, TLane
from tvm.tirx.layout import tid_in_wg as axis_tid_in_wg


def next_power_of_2(n):
    p = 1
    while p < n:
        p *= 2
    return p


# ===========================================================================
# §6 — uint32 TMEM-column divisibility proof
# ===========================================================================
def test_can_prove_divisible_uint32_offset():
    """``_can_prove_divisible`` must prove a symbolic uint32 index divisible by
    1 (where the bare ``floormod`` proof fails) and a known-even uint32 index
    divisible by 2 — the fp8 epilogue reads TMEM at a runtime uint32 column."""
    from tvm.backend.cuda.operator.tile_primitive.copy_async.tcgen05_ldst import (
        _can_prove_divisible,
    )

    analyzer = tvm.arith.Analyzer()
    u = T.Var("u", "uint32")

    # divisor == 1 is always divisible (the elem_per_32b==1 fp32-TMEM case).
    assert _can_prove_divisible(analyzer, u, 1) is True

    # A known-even uint32 index is provably divisible by 2. _can_prove_divisible
    # falls back to a signed view here: the bare unsigned proof of (u*2) % 2 == 0
    # is not available, since the multiply-cancellation simplifier rule is unsound
    # under unsigned wraparound and is (correctly) absent.
    assert _can_prove_divisible(analyzer, u * tvm.tirx.const(2, "uint32"), 2) is True
    # A symbolic uint32 is NOT provably divisible by 2.
    assert _can_prove_divisible(analyzer, u, 2) is False


def test_can_prove_divisible_signed_unaffected():
    from tvm.backend.cuda.operator.tile_primitive.copy_async.tcgen05_ldst import (
        _can_prove_divisible,
    )

    analyzer = tvm.arith.Analyzer()
    i = T.Var("i", "int32")
    assert _can_prove_divisible(analyzer, i, 1) is True
    assert _can_prove_divisible(analyzer, i * tvm.tirx.const(4, "int32"), 4) is True
    assert _can_prove_divisible(analyzer, i, 4) is False


# ===========================================================================
# §4 — maximum elementwise CUDA dispatch (ReLU building block)
# ===========================================================================
def test_maximum_elementwise():
    N = 128

    # fmt: off
    @T.prim_func
    def relu_max(a_ptr: T.handle, b_ptr: T.handle, c_ptr: T.handle) -> None:
        A = T.match_buffer(a_ptr, (N,), "float32")
        B = T.match_buffer(b_ptr, (N,), "float32")
        C = T.match_buffer(c_ptr, (N,), "float32")
        T.device_entry()
        T.warp_id([4])
        T.cta_id([1])
        T.warpgroup_id([1])
        tid = T.thread_id_in_wg([N])
        a_reg = T.alloc_local((1,), "float32")
        b_reg = T.alloc_local((1,), "float32")
        Tx.copy(a_reg[:], A[tid:tid + 1])
        Tx.copy(b_reg[:], B[tid:tid + 1])
        Tx.maximum(a_reg[:], a_reg[:], b_reg[:])
        Tx.copy(C[tid:tid + 1], a_reg[:])
    # fmt: on

    dev = tvm.cuda(0)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": relu_max}), target=target, tir_pipeline="tirx")
    np.random.seed(0)
    a = np.random.randn(N).astype("float32")
    b = np.random.randn(N).astype("float32")
    c = np.zeros(N, dtype="float32")
    a_t, b_t, c_t = (tvm.runtime.tensor(x, dev) for x in (a, b, c))
    mod["main"](a_t, b_t, c_t)
    np.testing.assert_allclose(c_t.numpy(), np.maximum(a, b), atol=0, rtol=0)


# ===========================================================================
# §1 / §2 — dense fp8 and tf32 (+ TFLOAT32 TMA) tcgen05 gemm_async
# ===========================================================================
def _run_dense_gemm(
    A_dtype, B_dtype, C_dtype, K, *, is_AB_tf32=False, tma_dtype_B=None, atol=1e-3, rtol=1e-3
):
    """Build + run a single-CTA M=128,N=128 dense tcgen05 GEMM (C = A @ Bᵀ) and
    compare to a numpy reference. Used to exercise the dense fp8 / tf32 MMA
    schedule and the TFLOAT32 TMA descriptor."""
    M, N = 128, 128
    A_shape = (M, K)
    B_shape = (N, K)
    C_shape = (M, N)
    A_swizzle, B_swizzle = 3, 3
    A_layout = mma_shared_layout(A_dtype, A_swizzle, A_shape)
    B_layout = mma_shared_layout(B_dtype, B_swizzle, B_shape)
    C_elem_32b = 4 // (tvm.runtime.DataType(C_dtype).bits // 8)
    cols_alloc = max(32, next_power_of_2(N // C_elem_32b))
    total_bytes = functools.reduce(operator.mul, A_shape, 1) * (
        tvm.runtime.DataType(A_dtype).bits // 8
    ) + functools.reduce(operator.mul, B_shape, 1) * (tvm.runtime.DataType(B_dtype).bits // 8)
    gemm_kw = {"dispatch": "tcgen05"}
    if is_AB_tf32:
        gemm_kw["is_AB_tf32"] = True
    b_tma_kw = {"dispatch": "tma"}
    if tma_dtype_B is not None:
        b_tma_kw["tma_dtype"] = tma_dtype_B

    # fmt: off
    @T.prim_func
    def gemm_async(A_ptr: T.handle, B_ptr: T.handle, C_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, A_shape, A_dtype)
        B = T.match_buffer(B_ptr, B_shape, B_dtype)
        C = T.match_buffer(C_ptr, C_shape, C_dtype)
        T.device_entry()
        warp_id = T.warp_id([4])
        T.cta_id([1])
        wg_id = T.warpgroup_id([1])
        tid_in_wg = T.thread_id_in_wg([128])

        A_smem = T.alloc_buffer(A_shape, A_dtype, scope="shared", layout=A_layout)
        B_smem = T.alloc_buffer(B_shape, B_dtype, scope="shared", layout=B_layout)
        tmem_addr = T.alloc_shared([1], "uint32")
        tma_mbar = T.alloc_shared([1], "uint64")
        mma_mbar = T.alloc_shared([1], "uint64")

        if tid_in_wg == 0:
            T.ptx.mbarrier.init(tma_mbar.ptr_to([0]), 1)
            T.ptx.mbarrier.init(mma_mbar.ptr_to([0]), 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if warp_id == 0:
            T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=cols_alloc, cta_group=1)
        T.cuda.cta_sync()
        tmem = T.decl_buffer((128, N), C_dtype, scope="tmem", allocated_addr=tmem_addr[0], layout=TileLayout(S[(128, N) : (1 @ TLane, 1 @ TCol)]))  # noqa: E501

        if tid_in_wg == 0:
            Tx.copy_async(A_smem[:, :], A[:, :], dispatch="tma", mbar=tma_mbar.ptr_to([0]))
            Tx.copy_async(B_smem[:, :], B[:, :], mbar=tma_mbar.ptr_to([0]), **b_tma_kw)
            T.ptx.mbarrier.arrive.expect_tx(tma_mbar.ptr_to([0]), total_bytes)
        T.ptx.mbarrier.try_wait(tma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        if tid_in_wg == 0:
            Tx.gemm_async(tmem[:, :], A_smem[:, :], B_smem[:, :], **gemm_kw)
            T.ptx.tcgen05.commit(mma_mbar.ptr_to([0]), cta_group=1)
        T.ptx.mbarrier.try_wait(mma_mbar.ptr_to([0]), 0)
        T.cuda.cta_sync()

        T.ptx.tcgen05.fence.after_thread_sync()
        C_reg = T.alloc_local(N, dtype=C_dtype)
        C_view = C_reg.view(128, N, layout=TileLayout(S[(128, N) : (1 @ axis_tid_in_wg, 1)]))
        if wg_id == 0:
            Tx.wg.copy_async(C_view[:, :], tmem[:, :])
            T.ptx.tcgen05.wait.ld()
        T.cuda.cta_sync()
        Tx.copy(C[tid_in_wg, 0:N], C_reg[:])

        if warp_id == 0:
            T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
            T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=cols_alloc, cta_group=1)
    # fmt: on

    dev = tvm.cuda(0)
    np.random.seed(0)
    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": gemm_async}), target=target, tir_pipeline="tirx")

    def _rand(shape, dtype):
        f = np.random.randn(*shape).astype("float32")
        return f.astype(dtype) if ml_dtypes is not None or "float8" not in dtype else f

    A_np = _rand(A_shape, A_dtype)
    B_np = _rand(B_shape, B_dtype)
    C_np = np.zeros(C_shape, dtype=C_dtype)
    A_t, B_t, C_t = (tvm.runtime.tensor(x, dev) for x in (A_np, B_np, C_np))
    mod["main"](A_t, B_t, C_t)

    C_ref = A_np.astype("float32") @ B_np.astype("float32").T
    np.testing.assert_allclose(C_t.numpy().astype("float32"), C_ref, atol=atol, rtol=rtol)


@pytest.mark.skipif(ml_dtypes is None, reason="Requires ml_dtypes for fp8")
def test_gemm_dense_fp8():
    # Dense (non-block-scaled) fp8 e4m3 MMA, MMA_K=32. K=128 so the 128B-swizzle
    # atom (128 fp8 elems / row) tiles evenly. Loose tolerance (fp8 precision).
    _run_dense_gemm("float8_e4m3fn", "float8_e4m3fn", "float32", 128, atol=2.0, rtol=0.15)


def test_gemm_tf32_with_tfloat32_tma():
    # tf32 dense MMA (is_AB_tf32=True, MMA_K=8); B loaded through the TFLOAT32
    # TMA descriptor so the RN-truncation happens on load (matching the MMA).
    _run_dense_gemm(
        "float32",
        "float32",
        "float32",
        64,
        is_AB_tf32=True,
        tma_dtype_B="tf32",
        atol=2e-2,
        rtol=2e-2,
    )


# ===========================================================================
# Unsigned floormod/floordiv simplification + TMA uint32-shape grouping
# ===========================================================================
def test_unsigned_floormod_floordiv_simplify():
    """The RewriteSimplifier must prove the OVERFLOW-FREE floormod/floordiv
    identities for uint32/uint64 (the signed IsIndexType block skips unsigned),
    so a uint shape extent can be grouped without an int32 cast. The
    multiply-cancellation identity stays UNPROVABLE for unsigned (unsound under
    wraparound) — and is verified to remain so."""
    a = tvm.arith.Analyzer()
    for dt in ("uint32", "uint64"):
        n = T.Var("n", dt)
        one = tvm.tirx.const(1, dt)
        assert a.can_prove_equal(tvm.tirx.floormod(n, n), 0)  # n % n -> 0
        assert a.can_prove_equal(tvm.tirx.floordiv(n, n), 1)  # n / n -> 1
        assert a.can_prove_equal(tvm.tirx.floormod(n, one), 0)  # n % 1 -> 0
        assert a.can_prove_equal(tvm.tirx.floordiv(n, one), n)  # n / 1 -> n
    # signed is unchanged; unsigned multiply-cancellation stays unprovable.
    ni, nu = T.Var("ni", "int32"), T.Var("nu", "uint32")
    assert a.can_prove_equal(tvm.tirx.floormod(ni * tvm.tirx.const(64, "int32"), ni), 0)
    assert not a.can_prove_equal(tvm.tirx.floormod(nu * tvm.tirx.const(64, "uint32"), nu), 0)


def test_tma_uint32_shape_no_cast():
    """A TMA-source gmem buffer whose shape extent is a runtime uint32 (no int32
    cast) must compile: the gmem-layout grouping falls back to the raw per-dim
    layout, which needs only the now-simplifiable ``dim % dim`` proof."""
    from tvm.backend.cuda.operator.tile_primitive.tma_utils import mma_shared_layout

    BK = 64
    A_layout = mma_shared_layout("float16", 3, (128, BK))

    # fmt: off
    @T.prim_func
    def tma_load(n: T.uint32, a_ptr: T.handle, o_ptr: T.handle) -> None:
        A = T.match_buffer(a_ptr, (n, BK), "float16")  # uint32 extent, NO int32 cast
        Out = T.match_buffer(o_ptr, (128, BK), "float16")
        T.device_entry()
        T.warp_id([4])
        T.cta_id([1])
        T.warpgroup_id([1])
        tid = T.thread_id_in_wg([128])
        sm = T.alloc_buffer((128, BK), "float16", scope="shared", layout=A_layout)
        mb = T.alloc_shared([1], "uint64")
        if tid == 0:
            T.ptx.mbarrier.init(mb.ptr_to([0]), 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        if tid == 0:
            Tx.copy_async(sm[:, :], A[0:128, 0:BK], dispatch="tma", mbar=mb.ptr_to([0]))
            T.ptx.mbarrier.arrive.expect_tx(mb.ptr_to([0]), 128 * BK * 2)
        T.ptx.mbarrier.try_wait(mb.ptr_to([0]), 0)
        T.cuda.cta_sync()
        reg = T.alloc_local(BK, "float16")
        Tx.copy(reg[:], sm[tid, 0:BK])
        Tx.copy(Out[tid, 0:BK], reg[:])
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        tvm.compile(tvm.IRModule({"main": tma_load}), target=target, tir_pipeline="tirx")


def test_tma_uint32_slice_base_no_cast():
    """A TMA copy whose gmem slice base is a runtime uint32 (so the copy *extent*
    carries the uint32 dtype) must compile: the smem regroup re-views the
    unsigned copy extent as signed (extents are signedness-irrelevant), so the
    slice base can stay uint32 with no int32 cast."""
    from tvm.backend.cuda.operator.tile_primitive.tma_utils import mma_shared_layout

    BK = 64
    A_layout = mma_shared_layout("float16", 3, (128, BK))

    # fmt: off
    @T.prim_func
    def tma_off(off: T.uint32, a_ptr: T.handle, o_ptr: T.handle) -> None:
        A = T.match_buffer(a_ptr, (4096, BK), "float16")
        Out = T.match_buffer(o_ptr, (128, BK), "float16")
        T.device_entry()
        T.warp_id([4])
        T.cta_id([1])
        T.warpgroup_id([1])
        tid = T.thread_id_in_wg([128])
        sm = T.alloc_buffer((128, BK), "float16", scope="shared", layout=A_layout)
        mb = T.alloc_shared([1], "uint64")
        if tid == 0:
            T.ptx.mbarrier.init(mb.ptr_to([0]), 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        if tid == 0:
            # uint32 slice base 'off' (no int32 cast)
            Tx.copy_async(sm[:, :], A[off:off + 128, 0:BK], dispatch="tma", mbar=mb.ptr_to([0]))
            T.ptx.mbarrier.arrive.expect_tx(mb.ptr_to([0]), 128 * BK * 2)
        T.ptx.mbarrier.try_wait(mb.ptr_to([0]), 0)
        T.cuda.cta_sync()
        reg = T.alloc_local(BK, "float16")
        Tx.copy(reg[:], sm[tid, 0:BK])
        Tx.copy(Out[tid, 0:BK], reg[:])
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        tvm.compile(tvm.IRModule({"main": tma_off}), target=target, tir_pipeline="tirx")


if __name__ == "__main__":
    tvm.testing.main()
