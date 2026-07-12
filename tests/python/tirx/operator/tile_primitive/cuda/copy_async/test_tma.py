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
# pylint: disable=invalid-name, missing-function-docstring
import functools

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm.ir import PointerType, PrimType
from tvm.ir.type import TensorMapType
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.testing import env
from tvm.tirx import IntImm, StringImm, Var
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import (
    mma_atom_layout,
    mma_atom_shape,
    mma_shared_layout,
)
from tvm.tirx.exec_scope import ExecScope
from tvm.tirx.layout import ComposeLayout, Iter, S, SwizzleLayout, TileLayout
from tvm.tirx.operator.tile_primitive.ops import CopyAsync
from tvm.tirx.stmt import DeclBuffer
from tvm.tirx.stmt_functor import StmtExprVisitor
from tvm.tirx.tile_primitive import DispatchContext

# ===========================================================================
# Helpers
# ===========================================================================


class TMACounter(StmtExprVisitor):
    """Visitor to count total TMA operations including loop iterations.

    This verifies that TMA copy operations are optimized correctly,
    resulting in minimal TMA instructions instead of multiple iterations.
    """

    def __init__(self):
        super().__init__()
        self.loop_extents = []  # Stack of loop extents
        self.total_tma_ops = 0

    def visit_for_(self, op):
        extent = op.extent
        self.loop_extents.append(extent)
        self.visit_stmt(op.body)
        self.loop_extents.pop()

    def visit_evaluate_(self, op):
        if isinstance(op.value, tvm.tirx.Call):
            if op.value.op.name in (
                "tirx.ptx.cp_async_bulk_tensor_g2s_cluster",
                "tirx.ptx.cp_async_bulk_tensor_shared_to_global",
                "tirx.ptx.cp_async_bulk_tensor_shared_to_global_reduce",
            ):
                # Multiply all enclosing loop extents
                iters = 1
                for ext in self.loop_extents:
                    iters *= ext
                self.total_tma_ops += iters


class TMABarAddrCounter(StmtExprVisitor):
    """Count TMA calls whose mbarrier operand is already a shared address."""

    def __init__(self):
        super().__init__()
        self.total_bar_addr_ops = 0

    def visit_evaluate_(self, op):
        if (
            isinstance(op.value, tvm.tirx.Call)
            and op.value.op.name == "tirx.ptx.cp_async_bulk_tensor_g2s_cluster"
            and len(op.value.args) >= 10
        ):
            bar_is_addr = op.value.args[9]
            if isinstance(bar_is_addr, tvm.tirx.IntImm) and int(bar_is_addr) == 1:
                self.total_bar_addr_ops += 1
        super().visit_evaluate_(op)


class AddressOfVarCollector(StmtExprVisitor):
    """Collect variable names passed directly to address_of()."""

    def __init__(self):
        super().__init__()
        self.var_names = set()

    def visit_call_(self, op):
        if (
            isinstance(op.op, tvm.ir.Op)
            and op.op.name == "tirx.address_of"
            and len(op.args) == 1
            and isinstance(op.args[0], tvm.tirx.Var)
        ):
            self.var_names.add(op.args[0].name)
        super().visit_call_(op)


class TensorMapEncodeCollector(StmtExprVisitor):
    """Collect integer args passed to runtime.cuTensorMapEncodeTiled."""

    def __init__(self):
        super().__init__()
        self.int_args = []

    def visit_call_(self, op):
        if (
            isinstance(op.op, tvm.ir.Op)
            and op.op.name == "tirx.tvm_call_packed"
            and len(op.args) >= 5
            and isinstance(op.args[0], StringImm)
            and op.args[0].value == "runtime.cuTensorMapEncodeTiled"
        ):
            self.int_args.append([int(arg) for arg in op.args[5:] if isinstance(arg, IntImm)])
        super().visit_call_(op)


class Gather4CallCollector(StmtExprVisitor):
    """Collect gather4 TMA calls in visitation order."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def visit_evaluate_(self, op):
        if (
            isinstance(op.value, tvm.tirx.Call)
            and op.value.op.name == "tirx.ptx.cp_async_bulk_tensor_g2s_cluster"
            and len(op.value.args) >= 9
            and isinstance(op.value.args[8], StringImm)
            and op.value.args[8].value == "tile_gather4"
        ):
            self.calls.append(op.value)
        super().visit_evaluate_(op)


class CallOpCounter(StmtExprVisitor):
    """Count TIR call op occurrences by op name."""

    def __init__(self):
        super().__init__()
        self.counts = {}

    def visit_call_(self, op):
        if isinstance(op.op, tvm.ir.Op):
            self.counts[op.op.name] = self.counts.get(op.op.name, 0) + 1
        super().visit_call_(op)


def _make_tma_call(
    g_shape,
    g_region,
    s_shape,
    s_region,
    gmem_layout,
    smem_layout,
    dtype="float16",
    direction="g2s",
    config=None,
    g_data=None,
    sctx=None,
):
    """Construct TilePrimitiveCall + DispatchContext and call copy_tma_impl.

    Returns (impl, host_init_stmts) on success, raises DispatchFail on failure.
    impl is the device-side PrimFunc, host_init_stmts is a list of Stmt
    for host-side tensor map creation.

    ``g_data`` optionally pins the global buffer's data var (so two calls can
    share one underlying tensor pointer); ``sctx`` optionally reuses a
    DispatchContext across calls (so the tensormap cache is shared).
    """
    from tvm.ir import Range
    from tvm.tirx import Var
    from tvm.tirx.cuda.operator.tile_primitive.copy_async.tma import copy_tma_impl
    from tvm.tirx.stmt import BufferRegion

    g_buf = tvm.tirx.decl_buffer(g_shape, dtype, "A", layout=gmem_layout, data=g_data)
    s_buf = tvm.tirx.decl_buffer(s_shape, dtype, "A_smem", scope="shared.dyn", layout=smem_layout)

    g_ranges = [Range.from_min_extent(r[0], r[1] - r[0]) for r in g_region]
    s_ranges = [Range.from_min_extent(r[0], r[1] - r[0]) for r in s_region]

    config = dict(config or {})
    if direction == "g2s":
        mbar_ptr = Var("mbar_ptr", "handle")
        config.setdefault("mbar", mbar_ptr)
        config.setdefault("cta_group", 1)
        dst_br = BufferRegion(s_buf, s_ranges)
        src_br = BufferRegion(g_buf, g_ranges)
    else:  # s2g
        config.setdefault("cta_group", 1)
        dst_br = BufferRegion(g_buf, g_ranges)
        src_br = BufferRegion(s_buf, s_ranges)

    op_call = CopyAsync(dst_br, src_br, config=config)

    if sctx is None:
        target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
        sctx = DispatchContext(target, ExecScope("thread"), {}, {})

    impl = copy_tma_impl(op_call, sctx)
    host_init_stmts = list(sctx.callbacks.get("host_init_stmt", []))
    return impl, host_init_stmts


def _count_tma_ops(impl):
    """Count total TMA ops in a PrimFunc (including loop multiplier)."""
    counter = TMACounter()
    counter.visit_stmt(impl.body)
    return counter.total_tma_ops


def _build_expected_host_init(dtype, encode_args):
    """Build expected host_init Bind+SeqStmt for cuTensorMapEncodeTiled.

    encode_args is a list of ints: the numeric arguments to cuTensorMapEncodeTiled
    after (tensormap, dtype_str, ndim, A_ptr). The full call is:
        runtime.cuTensorMapEncodeTiled(tensormap, dtype_str, ndim, A_ptr, *encode_args)
    where ndim = encode_args[0] and the rest are the tensor map parameters.
    """
    A_tensormap = Var("A_tensormap", PointerType(TensorMapType(), "global"))
    stack_alloca = tvm.tirx.Call(
        "handle",
        tvm.ir.Op.get("tirx.tvm_stack_alloca"),
        [StringImm("tensormap"), IntImm("int32", 1)],
    )
    A_var = Var("A", PointerType(PrimType(dtype), "global"))
    call_args = (
        [
            StringImm("runtime.cuTensorMapEncodeTiled"),
            A_tensormap,
            StringImm(dtype),
            IntImm("int32", encode_args[0]),  # ndim
            A_var,
        ]
        + [IntImm("int32", v) for v in encode_args[1:]]
    )
    encode_call = tvm.tirx.Call("int32", tvm.ir.Op.get("tirx.tvm_call_packed"), call_args)
    replace_point = tvm.tirx.Evaluate(tvm.tirx.op.tvm_kernel_replace_point())
    return tvm.tirx.SeqStmt(
        [tvm.tirx.Bind(A_tensormap, stack_alloca), tvm.tirx.Evaluate(encode_call), replace_point]
    )


def _build_expected_impl(direction, dtype, s_shape, s_layout, impl_spec):
    """Build expected impl PrimFunc.

    impl_spec is a dict with:
        loop_extents: list[int]  — e.g. [1], [2, 2], [8]
        dim: int  — TMA rank (number of coordinates, also the dim arg to PTX call)
        elem_offset_fn: callable(loop_vars) -> PrimExpr  (or None for 0)
        coord_fn: callable(loop_vars) -> list[PrimExpr]  (dim coordinate args)
        s_start: optional list[int]  — starting index for address_of (default all zeros)
    """
    from tvm.tirx.layout import ComposeLayout, SwizzleLayout

    loop_extents = impl_spec["loop_extents"]
    dim = impl_spec["dim"]
    elem_offset_fn = impl_spec.get("elem_offset_fn")
    coord_fn = impl_spec["coord_fn"]

    # Mirror _to_tile_layout() in copy_async/tma.py:
    #   ComposeLayout → tile_layout
    #   SwizzleLayout → identity TileLayout(S[shape])
    #   TileLayout    → as-is
    if isinstance(s_layout, ComposeLayout):
        buf_layout = s_layout.tile_layout
    elif isinstance(s_layout, SwizzleLayout):
        buf_layout = TileLayout(S[tuple(s_shape)])
    else:
        buf_layout = s_layout

    # Create loop vars
    n_loops = len(loop_extents)
    if n_loops == 1:
        loop_vars = [Var("loop_vars", "int32")]
    else:
        loop_vars = [Var(f"loop_vars_{i}", "int32") for i in range(n_loops)]

    # Buffer
    s_buf_ptr = Var("s_buf_w_offset_ptr", PointerType(PrimType(dtype), "shared.dyn"))
    elem_offset = elem_offset_fn(loop_vars) if elem_offset_fn else IntImm("int32", 0)
    s_buf = tvm.tirx.decl_buffer(
        s_shape,
        dtype,
        "s_buf_w_offset",
        data=s_buf_ptr,
        elem_offset=elem_offset,
        scope="shared.dyn",
        layout=buf_layout,
    )

    # Free variables
    mbar_ptr = Var("mbar_ptr", "handle")
    A_tensormap = Var("A_tensormap", PointerType(TensorMapType(), "global"))

    # address_of(s_buf[s_start...])
    s_start = impl_spec.get("s_start")
    if s_start:
        buf_indices = [IntImm("int32", v) for v in s_start]
    else:
        buf_indices = [IntImm("int32", 0)] * len(s_shape)
    addr_of = tvm.tirx.Call(
        "handle", tvm.ir.Op.get("tirx.address_of"), [tvm.tirx.BufferLoad(s_buf, buf_indices)]
    )

    # Coordinate args (must have exactly `dim` entries)
    coords = coord_fn(loop_vars)
    tensormap_addr = tvm.tirx.Call("uint64", tvm.ir.Op.get("tirx.address_of"), [A_tensormap])

    # Build PTX call based on direction
    if direction == "g2s":
        # g2s_cluster(dim, addr, mbar, tensormap, cta_mask, cta_group,
        #             cache_policy, has_cache_policy, load_mode,
        #             mbar_is_shared_addr, multicast, *coords)
        ptx_op = tvm.ir.Op.get("tirx.ptx.cp_async_bulk_tensor_g2s_cluster")
        ptx_args = [
            IntImm("int32", dim),
            addr_of,
            mbar_ptr,
            tensormap_addr,
            IntImm("int32", 0),
            IntImm("int32", 1),
            IntImm("uint64", 0),
            IntImm("int32", 0),
            StringImm("tile"),
            IntImm("int32", 0),
            IntImm("int32", 0),
            *coords,
        ]
    else:  # s2g
        # s2g(dim, addr, tensormap, cache_policy, has_cache_policy, *coords)
        ptx_op = tvm.ir.Op.get("tirx.ptx.cp_async_bulk_tensor_shared_to_global")
        ptx_args = [
            IntImm("int32", dim),
            addr_of,
            tensormap_addr,
            IntImm("uint64", 0),
            IntImm("int32", 0),
            *coords,
        ]

    eval_stmt = tvm.tirx.Evaluate(tvm.tirx.Call("", ptx_op, ptx_args))

    # Wrap: DeclBuffer -> nested For loops (skipped when total extent is 1,
    # matching the implementation's always-unroll single-loop emission).
    body = DeclBuffer(s_buf, eval_stmt)
    for i in range(n_loops - 1, -1, -1):
        body = tvm.tirx.For(
            loop_vars[i],
            IntImm("int32", 0),
            IntImm("int32", loop_extents[i]),
            tvm.tirx.ForKind.UNROLLED,
            body,
        )

    func = tvm.tirx.PrimFunc([], body, ret_type=None, buffer_map={})
    func = func.with_attr("global_symbol", "impl")
    # default s_tir=False is implicit; nothing to set here
    return func


def _zeros(n):
    """Return n zero IntImm coords."""
    return [IntImm("int32", 0)] * n


def _atom_rank5_elem_offset(lvs):
    """elem_offset for the structural 5D atom plan: lv * 8192."""
    return lvs[0] * 8192


def _atom_rank5_coords(lvs):
    """coord_fn for the structural 5D atom plan: [0, 0, 0, lv*2, 0]."""
    return [
        IntImm("int32", 0),
        IntImm("int32", 0),
        IntImm("int32", 0),
        lvs[0] * 2,
        IntImm("int32", 0),
    ]


def _stride_gap_elem_offset(lvs):
    """elem_offset for stride-gap-outer: lv * 4096."""
    return lvs[0] * 4096


def _stride_gap_3d_coords(lvs):
    """coord_fn for stride-gap-outer (rank=3): [0, 0, lv]."""
    return [IntImm("int32", 0), IntImm("int32", 0), lvs[0]]


def _atom_multiphase_rank5_elem_offset(lvs):
    """elem_offset for the multiphase 5D atom plan: lv * 4096."""
    return lvs[0] * 4096


def _atom_multiphase_rank5_coords(lvs):
    """coord_fn for multiphase rank-5 atom: [0, 0, lv%2*4, lv//2*2, 0]."""
    return [
        IntImm("int32", 0),
        IntImm("int32", 0),
        (lvs[0] % 2) * 4,
        (lvs[0] // 2) * 2,
        IntImm("int32", 0),
    ]


# fmt: off
# Expected parameters for each TMA test case.
# Each entry maps case_id -> (impl_spec_dict, encode_args_list).
#
# impl_spec keys:
#   loop_extents: list[int] — iteration counts for nested loops
#   dim: int — TMA rank = number of coordinates = dim arg to PTX call
#   coord_fn: callable(loop_vars) -> list[PrimExpr] — coordinate arguments (len == dim)
#   elem_offset_fn: optional callable(loop_vars) -> PrimExpr — buffer offset
#
# encode_args: list[int] — all numeric args to cuTensorMapEncodeTiled
#   [ndim, global_strides..., global_dims..., box_dims..., elem_strides...,
#    interleave, swizzle_mode, l2_promotion, oob_fill]


# ===========================================================================
# Section 2: TMA unit tests — single parametrized structural-golden driver
# ===========================================================================


def _tma_case(
    *,
    id,
    g_shape,
    g_region,
    s_shape,
    s_region,
    gmem_layout,
    smem_layout,
    dtype="float16",
    direction="g2s",
    config=None,
    impl_spec=None,
    encode_args=None,
    raises=None,
):
    """Build a pytest.param carrying a dict-form case for ``test_copy_tma_codegen``.

    Required: ``g_shape``, ``g_region``, ``s_shape``, ``s_region``, ``gmem_layout``,
    ``smem_layout``, ``id``.

    Optional:
        ``dtype``: element dtype (default ``"float16"``).
        ``direction``: ``"g2s"`` or ``"s2g"`` (default ``"g2s"``).
        ``config``: op config dict forwarded to ``copy_tma_impl`` (e.g.
            ``{"oob": "nan"}``).
        ``impl_spec``: kwargs for ``_build_expected_impl``. ``None`` skips the
            device-impl structural check.
        ``encode_args``: list for ``_build_expected_host_init``. ``None`` skips
            the host-init structural check.
        ``raises``: ``(ExceptionClass, regex_str)`` to expect instead of a
            successful dispatch.
    """
    return pytest.param(
        dict(
            g_shape=g_shape, g_region=g_region,
            s_shape=s_shape, s_region=s_region,
            gmem_layout=gmem_layout, smem_layout=smem_layout,
            dtype=dtype, direction=direction, config=config,
            impl_spec=impl_spec, encode_args=encode_args, raises=raises,
        ),
        id=id,
    )


# fmt: off
TMA_CASES = [
    # ======================================================================
    # G2S — 2D baseline (swizzle + dtype variants sharing (8, 256) shape)
    # ======================================================================
    _tma_case(
        id="g2s-2d-8x256",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float16", 3, (8, 256)),
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 64, 8, 4, 512, 128, 64, 8, 4, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-swizzle2",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float16", 2, (8, 256)),
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 32, 8, 8, 512, 64, 32, 8, 8, 1, 1, 1, 0, 2, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-swizzle1",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float16", 1, (8, 256)),
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 16, 8, 16, 512, 32, 16, 8, 16, 1, 1, 1, 0, 1, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-swizzle0",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float16", 0, (8, 256)),
        # SWIZZLE_NONE now yields the 8x128b packed-16B atom (3-axis), so the
        # TMA descriptor is rank 3 like the swizzled variants (62f57feda6).
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 8, 8, 32, 512, 16, 8, 8, 32, 1, 1, 1, 0, 0, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-int8",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("int8", 3, (8, 256)),
        dtype="int8",
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 128, 8, 2, 256, 128, 128, 8, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-bf16",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("bfloat16", 3, (8, 256)),
        dtype="bfloat16",
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 64, 8, 4, 512, 128, 64, 8, 4, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-fp32",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float32", 3, (8, 256)),
        dtype="float32",
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 32, 8, 8, 1024, 128, 32, 8, 8, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-uint8",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("uint8", 3, (8, 256)),
        dtype="uint8",
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 128, 8, 2, 256, 128, 128, 8, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-fp8e4m3",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float8_e4m3fn", 3, (8, 256)),
        dtype="float8_e4m3fn",
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 128, 8, 2, 256, 128, 128, 8, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-8x256-fp8e5m2",
        g_shape=(8, 256), g_region=((0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float8_e5m2", 3, (8, 256)),
        dtype="float8_e5m2",
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 128, 8, 2, 256, 128, 128, 8, 2, 1, 1, 1, 0, 3, 2, 0],
    ),
    # ======================================================================
    # G2S — 3D / partial / edge / multidim layouts
    # ======================================================================
    _tma_case(
        id="g2s-3d-shared-64x256",
        g_shape=(64, 256), g_region=((0, 64), (0, 256)),
        s_shape=(3, 64, 256), s_region=((1, 2), (0, 64), (0, 256)),
        gmem_layout=TileLayout(S[64, 256]),
        smem_layout=mma_shared_layout("float16", 3, (3, 64, 256)),
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3), s_start=[1, 0, 0]),
        encode_args=[3, 64, 64, 4, 512, 128, 64, 64, 4, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-2d-32x512-atom",
        g_shape=(32, 512), g_region=((0, 32), (0, 512)),
        s_shape=(32, 512), s_region=((0, 32), (0, 512)),
        gmem_layout=TileLayout(S[32, 512]),
        smem_layout=(
            mma_atom_layout("float16", 3)
            .tile_to((16, 256), mma_atom_shape("float16", 3))
            .tile_to((32, 512), (16, 256))
        ),
        impl_spec=dict(
            loop_extents=[2], dim=5,
            coord_fn=_atom_rank5_coords, elem_offset_fn=_atom_rank5_elem_offset,
        ),
        encode_args=[5, 64, 8, 4, 4, 2, 1024, 128, 8192, 512, 64, 8, 4, 2, 2, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    _tma_case(
        id="g2s-2d-partial-8192",
        g_shape=(8192, 8192), g_region=((0, 128), (0, 64)),
        s_shape=(128, 64), s_region=((0, 128), (0, 64)),
        gmem_layout=TileLayout(S[8192, 8192]),
        smem_layout=mma_shared_layout("float16", 3, (128, 64)),
        impl_spec=dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2)),
        encode_args=[2, 8192, 8192, 16384, 64, 128, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-edge-4d-shared-128x64",
        g_shape=(128, 64), g_region=((0, 128), (0, 64)),
        s_shape=(2, 2, 128, 64), s_region=((0, 1), (0, 1), (0, 128), (0, 64)),
        gmem_layout=TileLayout(S[128, 64]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (2, 2, 128, 64)).canonicalize(),
        impl_spec=dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2)),
        encode_args=[2, 64, 128, 128, 64, 128, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-edge-partial-offset",
        g_shape=(128, 64), g_region=((64, 64 + 24), (0, 64)),
        s_shape=(2, 2, 24, 64), s_region=((0, 1), (0, 1), (0, 24), (0, 64)),
        gmem_layout=TileLayout(S[128, 64]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (2, 2, 24, 64)).canonicalize(),
        impl_spec=dict(
            loop_extents=[1], dim=2,
            coord_fn=lambda lv: [IntImm("int32", 0), IntImm("int32", 64)],
        ),
        encode_args=[2, 64, 128, 128, 64, 24, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-edge-large-region",
        g_shape=(256, 64), g_region=((128, 256), (0, 64)),
        s_shape=(256, 64), s_region=((0, 128), (0, 64)),
        gmem_layout=TileLayout(S[256, 64]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (256, 64)).canonicalize(),
        impl_spec=dict(
            loop_extents=[1], dim=2,
            coord_fn=lambda lv: [IntImm("int32", 0), IntImm("int32", 128)],
        ),
        encode_args=[2, 64, 256, 128, 64, 128, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-partial-3d-shared-a",
        g_shape=(128, 256), g_region=((0, 32), (0, 64)),
        s_shape=(6, 128, 64), s_region=((0, 1), (0, 32), (0, 64)),
        gmem_layout=TileLayout(S[128, 256]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (6, 128, 64)).canonicalize(),
        impl_spec=dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2)),
        encode_args=[2, 256, 128, 512, 64, 32, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-partial-3d-shared-b",
        g_shape=(256, 512), g_region=((0, 64), (0, 64)),
        s_shape=(4, 256, 64), s_region=((1, 2), (0, 64), (0, 64)),
        gmem_layout=TileLayout(S[256, 512]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (4, 256, 64)).canonicalize(),
        impl_spec=dict(loop_extents=[1], dim=2, coord_fn=lambda lv: _zeros(2), s_start=[1, 0, 0]),
        encode_args=[2, 512, 256, 1024, 64, 64, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-3d-full-contiguous",
        g_shape=(4, 32, 64), g_region=((0, 4), (0, 32), (0, 64)),
        s_shape=(4, 32, 64), s_region=((0, 4), (0, 32), (0, 64)),
        gmem_layout=TileLayout(S[4, 32, 64]),
        smem_layout=TileLayout(S[4, 32, 64]),
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 64, 32, 4, 128, 4096, 64, 32, 4, 1, 1, 1, 0, 0, 2, 0],
    ),
    _tma_case(
        id="g2s-3d-partial-contiguous",
        g_shape=(8, 16, 128), g_region=((0, 4), (0, 16), (0, 128)),
        s_shape=(4, 16, 128), s_region=((0, 4), (0, 16), (0, 128)),
        gmem_layout=TileLayout(S[8, 16, 128]),
        smem_layout=TileLayout(S[4, 16, 128]),
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 128, 16, 8, 256, 4096, 128, 16, 4, 1, 1, 1, 0, 0, 2, 0],
    ),
    _tma_case(
        id="g2s-3d-stride-gap-outer",
        g_shape=(8, 32, 64), g_region=((0, 8), (0, 32), (0, 64)),
        s_shape=(8, 32, 64), s_region=((0, 8), (0, 32), (0, 64)),
        gmem_layout=TileLayout(S[8, 32, 64]),
        smem_layout=TileLayout(S[(8, 32, 64):(4096, 64, 1)]),
        impl_spec=dict(
            loop_extents=[8], dim=3,
            coord_fn=_stride_gap_3d_coords, elem_offset_fn=_stride_gap_elem_offset,
            s_start=[0, 0, 0],
        ),
        encode_args=[3, 64, 32, 8, 128, 4096, 64, 32, 1, 1, 1, 1, 0, 0, 2, 0],
    ),
    _tma_case(
        id="g2s-4d-reorder-a",
        g_shape=(2, 128, 8, 64), g_region=((0, 1), (0, 128), (0, 1), (0, 64)),
        s_shape=(1, 1, 128, 64), s_region=((0, 1), (0, 1), (0, 128), (0, 64)),
        gmem_layout=TileLayout(S[2, 128, 8, 64]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (1, 1, 128, 64)).canonicalize(),
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4), s_start=[0, 0, 0, 0]),  # noqa: E501
        encode_args=[4, 64, 128, 8, 2, 1024, 128, 131072, 64, 128, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-4d-reorder-b",
        g_shape=(4, 64, 4, 128), g_region=((0, 1), (0, 64), (0, 1), (0, 128)),
        s_shape=(1, 1, 64, 128), s_region=((0, 1), (0, 1), (0, 64), (0, 128)),
        gmem_layout=TileLayout(S[4, 64, 4, 128]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (1, 1, 64, 128)).canonicalize(),
        impl_spec=dict(loop_extents=[1], dim=5, coord_fn=lambda lv: _zeros(5), s_start=[0, 0, 0, 0]),  # noqa: E501
        encode_args=[5, 64, 64, 2, 4, 4, 1024, 128, 256, 65536, 64, 64, 2, 1, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    _tma_case(
        id="g2s-multidim-4d-a",
        g_shape=(2, 2, 128, 64), g_region=((0, 1), (0, 1), (0, 128), (0, 64)),
        s_shape=(128, 64), s_region=((0, 128), (0, 64)),
        gmem_layout=TileLayout(S[2, 2, 128, 64]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (128, 64)),
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 64, 128, 2, 2, 128, 16384, 32768, 64, 128, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-multidim-4d-b",
        g_shape=(4, 64, 4, 128), g_region=((0, 1), (0, 64), (0, 1), (0, 128)),
        s_shape=(64, 128), s_region=((0, 64), (0, 128)),
        gmem_layout=TileLayout(S[4, 64, 4, 128]).canonicalize(),
        smem_layout=mma_shared_layout("float16", 3, (64, 128)),
        impl_spec=dict(loop_extents=[1], dim=5, coord_fn=lambda lv: _zeros(5)),
        encode_args=[5, 64, 64, 2, 4, 4, 1024, 128, 256, 65536, 64, 64, 2, 1, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    # ======================================================================
    # G2S — per-phase slices (multiphase)
    # ======================================================================
    _tma_case(
        id="g2s-multiphase-3x8x256",
        g_shape=(3, 8, 256), g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[3, 8, 256]),
        smem_layout=mma_shared_layout("float16", 3, (8, 256)),
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 64, 8, 4, 3, 512, 128, 4096, 64, 8, 4, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-multiphase-5x64x256",
        g_shape=(5, 64, 256), g_region=((0, 1), (0, 64), (0, 256)),
        s_shape=(64, 256), s_region=((0, 64), (0, 256)),
        gmem_layout=TileLayout(S[5, 64, 256]),
        smem_layout=mma_shared_layout("float16", 3, (64, 256)),
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 64, 64, 4, 5, 512, 128, 32768, 64, 64, 4, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-multiphase-7x32x512-atom",
        g_shape=(7, 32, 512), g_region=((0, 1), (0, 32), (0, 512)),
        s_shape=(32, 512), s_region=((0, 32), (0, 512)),
        gmem_layout=TileLayout(S[7, 32, 512]),
        smem_layout=(
            mma_atom_layout("float16", 3)
            .tile_to((16, 256), mma_atom_shape("float16", 3))
            .tile_to((32, 512), (16, 256))
        ),
        impl_spec=dict(
            loop_extents=[4], dim=5,
            coord_fn=_atom_multiphase_rank5_coords, elem_offset_fn=_atom_multiphase_rank5_elem_offset,  # noqa: E501
        ),
        encode_args=[5, 64, 8, 8, 4, 7, 1024, 128, 8192, 32768, 64, 8, 4, 2, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    # ======================================================================
    # G2S — transpose-like permuted layouts: DECLINED.
    # A column-major smem tile makes the gmem-contiguous dim strided in smem,
    # so the fastest TMA box collapses to a single element (boxDim[0]=1). For
    # fp16 that is 2 B, which fails cuTensorMap's "boxDim[0]*elementSize must be
    # a multiple of 16 B" rule — so these are declined at dispatch rather than
    # emitting a descriptor the host wrapper would reject (they had no GPU
    # round-trip and no production user; real transposes use a legal box).
    # ======================================================================
    _tma_case(
        id="g2s-transpose-32x64",
        g_shape=(32, 64), g_region=((0, 32), (0, 64)),
        s_shape=(32, 64), s_region=((0, 32), (0, 64)),
        gmem_layout=TileLayout(S[32, 64]),
        smem_layout=TileLayout(S[(32, 64):(1, 32)]),
        raises=(Exception, "not a multiple of 16 B"),
    ),
    _tma_case(
        id="g2s-transpose-64x32",
        g_shape=(64, 32), g_region=((0, 64), (0, 32)),
        s_shape=(64, 32), s_region=((0, 64), (0, 32)),
        gmem_layout=TileLayout(S[64, 32]),
        smem_layout=TileLayout(S[(64, 32):(1, 64)]),
        raises=(Exception, "not a multiple of 16 B"),
    ),
    _tma_case(
        id="g2s-transpose-partial-region",
        g_shape=(128, 64), g_region=((0, 64), (0, 64)),
        s_shape=(64, 64), s_region=((0, 64), (0, 64)),
        gmem_layout=TileLayout(S[128, 64]),
        smem_layout=TileLayout(S[(64, 64):(1, 64)]),
        raises=(Exception, "not a multiple of 16 B"),
    ),
    _tma_case(
        id="g2s-transpose-partial-offset",
        g_shape=(128, 64), g_region=((64, 128), (0, 32)),
        s_shape=(64, 32), s_region=((0, 64), (0, 32)),
        gmem_layout=TileLayout(S[128, 64]),
        smem_layout=TileLayout(S[(64, 32):(1, 64)]),
        raises=(Exception, "not a multiple of 16 B"),
    ),
    # ======================================================================
    # G2S — non-prefix compact (4D gmem collapses to one TMA tile)
    # ======================================================================
    _tma_case(
        id="g2s-non-prefix-compact-elides",
        g_shape=(16, 16, 128, 128), g_region=((3, 4), (4, 5), (0, 128), (0, 128)),
        s_shape=(128, 128), s_region=((0, 128), (0, 128)),
        gmem_layout=TileLayout(S[(16, 16, 128, 128):(1024 * 128, 128, 1024, 1)]),
        smem_layout=TileLayout(S[128, 128]),
        impl_spec=dict(
            loop_extents=[1], dim=4,
            coord_fn=lambda lv: [
                IntImm("int32", 0), IntImm("int32", 0),
                IntImm("int32", 4), IntImm("int32", 3),
            ],
        ),
        encode_args=[4, 128, 128, 16, 16, 2048, 256, 262144, 128, 128, 1, 1, 1, 1, 1, 1, 0, 0, 2, 0],  # noqa: E501
    ),
    # ======================================================================
    # G2S — oob contract (config={"oob": ...}); fill kind is encoded in
    # encode_args[-1]. ``None`` and ``"zero"`` both map to fill_kind=0.
    # ======================================================================
    _tma_case(
        id="g2s-oob-zero",
        g_shape=(128, 64), g_region=((120, 136), (0, 64)),
        s_shape=(16, 64), s_region=((0, 16), (0, 64)),
        gmem_layout=TileLayout(S[128, 64]),
        smem_layout=mma_shared_layout("float16", 3, (16, 64)),
        config={"oob": "zero"},
        impl_spec=dict(
            loop_extents=[1], dim=2,
            coord_fn=lambda lv: [IntImm("int32", 0), IntImm("int32", 120)],
        ),
        encode_args=[2, 64, 128, 128, 64, 16, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="g2s-oob-nan",
        g_shape=(128, 64), g_region=((120, 136), (0, 64)),
        s_shape=(16, 64), s_region=((0, 16), (0, 64)),
        gmem_layout=TileLayout(S[128, 64]),
        smem_layout=mma_shared_layout("float16", 3, (16, 64)),
        config={"oob": "nan"},
        impl_spec=dict(
            loop_extents=[1], dim=2,
            coord_fn=lambda lv: [IntImm("int32", 0), IntImm("int32", 120)],
        ),
        encode_args=[2, 64, 128, 128, 64, 16, 1, 1, 0, 3, 2, 1],
    ),
    # ======================================================================
    # G2S — flash_attention4 Q/K/V regression baselines
    # Representative config: batch=1, seq_len=2048, num_qo_heads=32,
    # num_kv_heads=8, head_dim=128 → GQA_RATIO=4, SEQ_Q_PER_TILE=32,
    # BLK_M=BLK_N=128, SMEM_PIPE_DEPTH_Q=2, SMEM_PIPE_DEPTH_KV=3. Each case
    # lowers to exactly one cp_async_bulk_tensor; structural golden locks
    # rank / shape / coord / box.
    # ======================================================================
    _tma_case(
        id="g2s-fa4-q",
        g_shape=(1, 2048, 32, 128), g_region=((0, 1), (0, 32), (0, 4), (0, 128)),
        s_shape=(2, 128, 128), s_region=((0, 1), (0, 128), (0, 128)),
        gmem_layout=TileLayout(S[1, 2048, 32, 128]),
        smem_layout=mma_shared_layout("float16", 3, (2, 128, 128)),
        impl_spec=dict(loop_extents=[1], dim=5, coord_fn=lambda lv: _zeros(5)),
        encode_args=[5, 64, 32, 2048, 2, 1, 256, 8192, 128, 0, 64, 4, 32, 2, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    _tma_case(
        id="g2s-fa4-k",
        g_shape=(1, 2048, 8, 128), g_region=((0, 1), (0, 128), (0, 1), (0, 128)),
        s_shape=(3, 128, 128), s_region=((0, 1), (0, 128), (0, 128)),
        gmem_layout=TileLayout(S[1, 2048, 8, 128]),
        smem_layout=mma_shared_layout("float16", 3, (3, 128, 128)),
        impl_spec=dict(loop_extents=[1], dim=5, coord_fn=lambda lv: _zeros(5)),
        encode_args=[5, 64, 2048, 2, 8, 1, 2048, 128, 256, 0, 64, 128, 2, 1, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    _tma_case(
        id="g2s-fa4-v",
        g_shape=(1, 2048, 8, 128), g_region=((0, 1), (0, 128), (0, 1), (0, 128)),
        s_shape=(3, 128, 128), s_region=((0, 1), (0, 128), (0, 128)),
        gmem_layout=TileLayout(S[1, 2048, 8, 128]),
        smem_layout=mma_shared_layout("float16", 3, (3, 128, 128)),
        impl_spec=dict(loop_extents=[1], dim=5, coord_fn=lambda lv: _zeros(5)),
        encode_args=[5, 64, 2048, 2, 8, 1, 2048, 128, 256, 0, 64, 128, 2, 1, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    # ======================================================================
    # S2G — per-phase slices (swizzle + dtype variants)
    # ======================================================================
    _tma_case(
        id="s2g-multiphase-3x8x256",
        direction="s2g",
        g_shape=(3, 8, 256), g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[3, 8, 256]),
        smem_layout=mma_shared_layout("float16", 3, (8, 256)),
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 64, 8, 4, 3, 512, 128, 4096, 64, 8, 4, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="s2g-multiphase-5x64x256",
        direction="s2g",
        g_shape=(5, 64, 256), g_region=((0, 1), (0, 64), (0, 256)),
        s_shape=(64, 256), s_region=((0, 64), (0, 256)),
        gmem_layout=TileLayout(S[5, 64, 256]),
        smem_layout=mma_shared_layout("float16", 3, (64, 256)),
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 64, 64, 4, 5, 512, 128, 32768, 64, 64, 4, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="s2g-multiphase-7x32x512-atom",
        direction="s2g",
        g_shape=(7, 32, 512), g_region=((0, 1), (0, 32), (0, 512)),
        s_shape=(32, 512), s_region=((0, 32), (0, 512)),
        gmem_layout=TileLayout(S[7, 32, 512]),
        smem_layout=(
            mma_atom_layout("float16", 3)
            .tile_to((16, 256), mma_atom_shape("float16", 3))
            .tile_to((32, 512), (16, 256))
        ),
        impl_spec=dict(
            loop_extents=[4], dim=5,
            coord_fn=_atom_multiphase_rank5_coords, elem_offset_fn=_atom_multiphase_rank5_elem_offset,  # noqa: E501
        ),
        encode_args=[5, 64, 8, 8, 4, 7, 1024, 128, 8192, 32768, 64, 8, 4, 2, 1, 1, 1, 1, 1, 1, 0, 3, 2, 0],  # noqa: E501
    ),
    _tma_case(
        id="s2g-multiphase-3x8x256-swizzle2",
        direction="s2g",
        g_shape=(3, 8, 256), g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[3, 8, 256]),
        smem_layout=mma_shared_layout("float16", 2, (8, 256)),
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 32, 8, 8, 3, 512, 64, 4096, 32, 8, 8, 1, 1, 1, 1, 1, 0, 2, 2, 0],
    ),
    _tma_case(
        id="s2g-multiphase-3x8x256-swizzle0",
        direction="s2g",
        g_shape=(3, 8, 256), g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[3, 8, 256]),
        smem_layout=mma_shared_layout("float16", 0, (8, 256)),
        # SWIZZLE_NONE now yields the 8x128b packed-16B atom (3-axis), so the
        # descriptor gains one box dim (rank 3 -> 4) like the swizzled variants
        # (62f57feda6).
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 8, 8, 32, 3, 512, 16, 4096, 8, 8, 32, 1, 1, 1, 1, 1, 0, 0, 2, 0],
    ),
    _tma_case(
        id="s2g-multiphase-3x8x256-int8",
        direction="s2g",
        g_shape=(3, 8, 256), g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[3, 8, 256]),
        smem_layout=mma_shared_layout("int8", 3, (8, 256)),
        dtype="int8",
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 128, 8, 2, 3, 256, 128, 2048, 128, 8, 2, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="s2g-multiphase-3x8x256-fp32",
        direction="s2g",
        g_shape=(3, 8, 256), g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[3, 8, 256]),
        smem_layout=mma_shared_layout("float32", 3, (8, 256)),
        dtype="float32",
        impl_spec=dict(loop_extents=[1], dim=4, coord_fn=lambda lv: _zeros(4)),
        encode_args=[4, 32, 8, 8, 3, 1024, 128, 8192, 32, 8, 8, 1, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    # ======================================================================
    # S2G — retain multi-dim coords without linear-carry (bf16, custom layout)
    # ======================================================================
    _tma_case(
        id="s2g-keeps-multidim-coords",
        direction="s2g",
        g_shape=(1024, 4, 1024), g_region=((128, 128 + 128), (1, 1 + 1), (32, 32 + 32)),
        s_shape=(128, 32), s_region=((0, 128), (0, 32)),
        gmem_layout=TileLayout(S[(1024, 4, 1024):(4 * 1024, 1024, 1)]),
        smem_layout=TileLayout(S[(128, 32):(32, 1)]),
        dtype="bfloat16",
        impl_spec=dict(
            loop_extents=[1], dim=3,
            coord_fn=lambda lv: [
                IntImm("int32", 32),
                IntImm("int32", 128),
                IntImm("int32", 1),
            ],
        ),
    ),
    # ======================================================================
    # S2G — oob contract variants over the same (2, 128, 64) shape. ``None``
    # and ``"zero"`` map to fill_kind=0; ``"nan"`` maps to fill_kind=1. The
    # descriptor geometry is identical across the three variants.
    # ======================================================================
    _tma_case(
        id="s2g-oob-none",
        direction="s2g",
        g_shape=(2, 128, 64), g_region=((0, 1), (0, 128), (0, 64)),
        s_shape=(128, 64), s_region=((0, 128), (0, 64)),
        gmem_layout=TileLayout(S[(2, 128, 64)]),
        smem_layout=mma_shared_layout("float16", 3, (128, 64)),
        config=None,
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 64, 128, 2, 128, 16384, 64, 128, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="s2g-oob-zero",
        direction="s2g",
        g_shape=(2, 128, 64), g_region=((0, 1), (0, 128), (0, 64)),
        s_shape=(128, 64), s_region=((0, 128), (0, 64)),
        gmem_layout=TileLayout(S[(2, 128, 64)]),
        smem_layout=mma_shared_layout("float16", 3, (128, 64)),
        config={"oob": "zero"},
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 64, 128, 2, 128, 16384, 64, 128, 1, 1, 1, 1, 0, 3, 2, 0],
    ),
    _tma_case(
        id="s2g-oob-nan",
        direction="s2g",
        g_shape=(2, 128, 64), g_region=((0, 1), (0, 128), (0, 64)),
        s_shape=(128, 64), s_region=((0, 128), (0, 64)),
        gmem_layout=TileLayout(S[(2, 128, 64)]),
        smem_layout=mma_shared_layout("float16", 3, (128, 64)),
        config={"oob": "nan"},
        impl_spec=dict(loop_extents=[1], dim=3, coord_fn=lambda lv: _zeros(3)),
        encode_args=[3, 64, 128, 2, 128, 16384, 64, 128, 1, 1, 1, 1, 0, 3, 2, 1],
    ),
    # ======================================================================
    # Rejection cases — oob contract validation
    # ======================================================================
    _tma_case(
        id="reject-unknown-oob",
        direction="s2g",
        g_shape=(3, 8, 256), g_region=((0, 1), (0, 8), (0, 256)),
        s_shape=(8, 256), s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[3, 8, 256]),
        smem_layout=mma_shared_layout("float16", 3, (8, 256)),
        config={"oob": "bogus"},
        raises=(Exception, "Unsupported TMA oob mode"),
    ),
    _tma_case(
        id="reject-g2s-nan-on-non-float",
        g_shape=(128, 64), g_region=((120, 136), (0, 64)),
        s_shape=(16, 64), s_region=((0, 16), (0, 64)),
        gmem_layout=TileLayout(S[128, 64]),
        smem_layout=TileLayout(S[16, 64]),
        dtype="int8",
        config={"oob": "nan"},
        raises=(Exception, "requires a floating-point dtype"),
    ),
    _tma_case(
        id="reject-s2g-nan-on-non-float",
        direction="s2g",
        g_shape=(2, 128, 64), g_region=((0, 1), (0, 128), (0, 64)),
        s_shape=(128, 64), s_region=((0, 128), (0, 64)),
        gmem_layout=TileLayout(S[2, 128, 64]),
        smem_layout=TileLayout(S[128, 64]),
        dtype="int8",
        config={"oob": "nan"},
        raises=(Exception, "requires a floating-point dtype"),
    ),
]
# fmt: on


@pytest.mark.parametrize("case", TMA_CASES)
def test_copy_tma_codegen(case):
    """Unified structural-golden driver for every TMA unit test case.

    See ``_tma_case`` for the dict-form input. When ``raises`` is set, the
    test expects ``_make_tma_call`` to raise; otherwise it compares the
    emitted device impl and host tensormap-init against the inlined
    ``impl_spec`` / ``encode_args`` goldens.
    """
    call_kwargs = dict(
        g_shape=case["g_shape"],
        g_region=case["g_region"],
        s_shape=case["s_shape"],
        s_region=case["s_region"],
        gmem_layout=case["gmem_layout"],
        smem_layout=case["smem_layout"],
        dtype=case["dtype"],
        direction=case["direction"],
        config=case["config"],
    )
    if case["raises"] is not None:
        exc, match = case["raises"]
        with pytest.raises(exc, match=match):
            _make_tma_call(**call_kwargs)
        return

    impl, host_init_stmts = _make_tma_call(**call_kwargs)
    if case["impl_spec"] is not None:
        expected_impl = _build_expected_impl(
            case["direction"],
            case["dtype"],
            case["s_shape"],
            case["smem_layout"],
            case["impl_spec"],
        )
        tvm.ir.assert_structural_equal(impl, expected_impl, map_free_vars=True)
    if case["encode_args"] is not None:
        expected_host = _build_expected_host_init(case["dtype"], case["encode_args"])
        assert len(host_init_stmts) == 1
        tvm.ir.assert_structural_equal(host_init_stmts[0], expected_host, map_free_vars=True)


def test_copy_tma_external_tensor_map_skips_host_init():
    external_tensor_map = Var("external_tensormap", PointerType(TensorMapType(), "global"))
    impl, host_init_stmts = _make_tma_call(
        g_shape=(8, 256),
        g_region=((0, 8), (0, 256)),
        s_shape=(8, 256),
        s_region=((0, 8), (0, 256)),
        gmem_layout=TileLayout(S[8, 256]),
        smem_layout=mma_shared_layout("float16", 3, (8, 256)),
        config={"tensor_map": external_tensor_map},
    )

    assert host_init_stmts == []
    assert _count_tma_ops(impl) == 1
    collector = AddressOfVarCollector()
    collector.visit_stmt(impl.body)
    assert "external_tensormap" in collector.var_names


def test_copy_tma_tensormap_l2_promotion_config():
    def l2_promotion_from_host(config):
        _, host_init_stmts = _make_tma_call(
            g_shape=(8, 256),
            g_region=((0, 8), (0, 256)),
            s_shape=(8, 256),
            s_region=((0, 8), (0, 256)),
            gmem_layout=TileLayout(S[8, 256]),
            smem_layout=mma_shared_layout("float16", 3, (8, 256)),
            config=config,
        )
        assert len(host_init_stmts) == 1
        collector = TensorMapEncodeCollector()
        collector.visit_stmt(host_init_stmts[0])
        assert len(collector.int_args) == 1
        return collector.int_args[0][-2]

    assert l2_promotion_from_host({}) == 2
    assert l2_promotion_from_host({"tensormap_l2_promotion": "L2::256B"}) == 3
    assert l2_promotion_from_host({"tensormap_l2_promotion": 0}) == 0


def _build_tma_gather4_indexer_kernel(
    dtype="float16", cta_group=2, mbarrier_addr=False, prefetch_tensormap=False
):
    rows = 256
    copy_rows = 16
    cols = 64
    thread_cnt = 128
    shared_layout = mma_shared_layout(dtype, 3, (copy_rows, cols))
    smem_bytes = copy_rows * cols * tvm.DataType(dtype).bits // 8

    # fmt: off
    @T.prim_func
    def tma_gather4_indexer(A_ptr: T.handle, I_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (rows, cols), dtype)
        Idx = T.match_buffer(I_ptr, (copy_rows,), "int32")
        B = T.match_buffer(B_ptr, (copy_rows, cols), dtype)

        T.device_entry()
        T.cta_id([1])
        tid = T.thread_id([thread_cnt])
        dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
        A_smem = T.decl_buffer(
            (copy_rows, cols), dtype, dyn.data, elem_offset=0, layout=shared_layout
        )
        mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
        mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

        if tid == 0:
            T.ptx.mbarrier.init(mbar_ptr, 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if tid == 0:
            Tx.copy_async(
                A_smem[:, :],
                A[:, :],
                dispatch="tma",
                mbar=mbar_ptr,
                cta_group=cta_group,
                cache_hint=T.uint64(0x14F0000000000000),
                mbarrier_addr=mbarrier_addr,
                gather_axis=0,
                dst_gather_axis=0,
                indexer=[Idx[i] for i in range(copy_rows)],
                prefetch_tensormap=prefetch_tensormap,
            )
            T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, smem_bytes)
        T.ptx.mbarrier.try_wait(mbar_ptr, 0)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        Tx.cta.copy(B[:, :], A_smem[:, :])
        # fmt: on

    return tma_gather4_indexer


def _build_tma_gather4_rank3_dst_kernel(dtype="float16"):
    rows = 256
    copy_rows = 16
    cols = 64
    stages = 2
    thread_cnt = 128
    shared_layout = mma_shared_layout(dtype, 3, (stages, copy_rows, cols))
    smem_bytes = stages * copy_rows * cols * tvm.DataType(dtype).bits // 8

    # fmt: off
    @T.prim_func
    def tma_gather4_rank3_dst(A_ptr: T.handle, I_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (rows, cols), dtype)
        Idx = T.match_buffer(I_ptr, (copy_rows,), "int32")

        T.device_entry()
        T.cta_id([1])
        tid = T.thread_id([thread_cnt])
        dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
        A_smem = T.decl_buffer(
            (stages, copy_rows, cols), dtype, dyn.data, elem_offset=0, layout=shared_layout
        )
        mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
        mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

        if tid == 0:
            Tx.copy_async(
                A_smem[1:2, :, :],
                A[:, :],
                dispatch="tma",
                mbar=mbar_ptr,
                cta_group=2,
                cache_hint=T.uint64(0x14F0000000000000),
                gather_axis=0,
                dst_gather_axis=1,
                indexer=[Idx[i] for i in range(copy_rows)],
            )
        # fmt: on

    return tma_gather4_rank3_dst


def test_copy_tma_gather4_indexer_lowers_to_four_ptx_calls():
    mod = tvm.IRModule({"main": _build_tma_gather4_indexer_kernel()})
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        lowered = tvm.tirx.transform.LowerTIRx()(mod)

    counter = TMACounter()
    counter.visit_stmt(lowered["main"].body)
    assert counter.total_tma_ops == 4


def test_copy_tma_gather4_indexer_rank3_dst_lowers_to_four_ptx_calls():
    mod = tvm.IRModule({"main": _build_tma_gather4_rank3_dst_kernel()})
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        lowered = tvm.tirx.transform.LowerTIRx()(mod)

    counter = TMACounter()
    counter.visit_stmt(lowered["main"].body)
    assert counter.total_tma_ops == 4


def test_copy_tma_gather4_indexer_issue_axes_emit_chunks_outermost():
    impl, _ = _make_tma_call(
        g_shape=(8192, 512),
        g_region=((0, 16), (0, 256)),
        s_shape=(16, 256),
        s_region=((0, 16), (0, 256)),
        gmem_layout=TileLayout(S[(8192, 512) : (512, 1)]),
        smem_layout=ComposeLayout(
            SwizzleLayout(3, 3, 3, swizzle_inner=True),
            TileLayout.from_iters(
                [
                    Iter(4, 16 * 64, "m"),
                    Iter(4, 64, "m"),
                    Iter(4, 64 * 64, "m"),
                    Iter(64, 1, "m"),
                ]
            ),
        ),
        config={
            "gather_axis": 0,
            "dst_gather_axis": 0,
            "indexer": [IntImm("int32", i) for i in range(16)],
            "cache_hint": IntImm("uint64", 0),
        },
    )

    assert _count_tma_ops(impl) == 16
    assert isinstance(impl.body, tvm.tirx.SeqStmt)
    assert len(impl.body.seq) == 4
    for chunk_idx, stmt in enumerate(impl.body.seq):
        assert isinstance(stmt, tvm.tirx.For)
        assert isinstance(stmt.extent, IntImm)
        assert int(stmt.extent) == 4
        collector = Gather4CallCollector()
        collector.visit_stmt(stmt)
        assert len(collector.calls) == 1
        gather_coords = collector.calls[0].args[-4:]
        assert [int(coord) for coord in gather_coords] == list(
            range(chunk_idx * 4, chunk_idx * 4 + 4)
        )


def test_copy_tma_gather4_indexer_mbarrier_addr_lowers_to_bar_addr():
    mod = tvm.IRModule({"main": _build_tma_gather4_indexer_kernel(cta_group=1, mbarrier_addr=True)})
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        lowered = tvm.tirx.transform.LowerTIRx()(mod)

    counter = TMABarAddrCounter()
    counter.visit_stmt(lowered["main"].body)
    assert counter.total_bar_addr_ops == 4


def test_copy_tma_prefetch_tensormap_uses_elected_lane():
    mod = tvm.IRModule(
        {"main": _build_tma_gather4_indexer_kernel(cta_group=1, prefetch_tensormap=True)}
    )
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        lowered = tvm.tirx.transform.LowerTIRx()(mod)

    counter = CallOpCounter()
    counter.visit_stmt(lowered["main"].body)
    assert counter.counts.get("tirx.ptx.prefetch_tensormap", 0) == 1
    assert counter.counts.get("tirx.ptx.elect_sync", 0) >= 1


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_copy_tma_gather4_indexer_gpu_smoke():
    dtype = "float16"
    rows = 256
    copy_rows = 16
    cols = 64
    dev = tvm.cuda(0)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": _build_tma_gather4_indexer_kernel(dtype, cta_group=1)}),
            target=target,
            tir_pipeline="tirx",
        )

    np.random.seed(0)
    A_np = tvm.testing.generate_random_array(dtype, (rows, cols))
    I_np = np.array([7, 3, 19, 5, 31, 11, 2, 23, 43, 13, 47, 17, 59, 29, 61, 37], dtype="int32")
    B_np = np.zeros((copy_rows, cols), dtype=tvm.testing.np_dtype_from_str(dtype))

    A = tvm.runtime.tensor(A_np, dev)
    Idx = tvm.runtime.tensor(I_np, dev)
    B = tvm.runtime.tensor(B_np, dev)
    mod(A, Idx, B)

    np.testing.assert_allclose(A_np[I_np], B.numpy())


def _build_tma_gather4_multi_iter_kernel(dtype="float16"):
    """Gather4 copies whose plan needs an issue loop (flat_total_extent > 1).

    The smem destination splits its 128 columns into two 64-column blocks at
    stride rows*64 (the chunk-major mma placement), so each gather copy plans
    box (4, 64) plus a 2-iteration issue loop; with a 4-row indexer this is
    exactly one gather4 chunk. Regression for the single-chunk multi-iter
    emission (a SeqStmt-of-1 crash) and for the duplicated CSE binds that
    unrolled shared subtrees used to leave behind (non-SSA at
    SplitHostDevice), which is re-established by ConvertSSA in the pipeline.
    """
    rows = 256
    gather_rows = 8
    cols = 128
    thread_cnt = 128
    shared_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(T.S[(gather_rows, cols // 64, 64) : (64, gather_rows * 64, 1)]),
    )
    smem_bytes = gather_rows * cols * tvm.DataType(dtype).bits // 8

    # fmt: off
    @T.prim_func
    def tma_gather4_multi_iter(A_ptr: T.handle, I_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (rows, cols), dtype)
        Idx = T.match_buffer(I_ptr, (gather_rows,), "int32")
        B = T.match_buffer(B_ptr, (gather_rows, cols), dtype)

        T.device_entry()
        T.cta_id([1])
        tid = T.thread_id([thread_cnt])
        dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
        A_smem = T.decl_buffer(
            (gather_rows, cols), dtype, dyn.data, elem_offset=0, layout=shared_layout
        )
        mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
        mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

        if tid == 0:
            T.ptx.mbarrier.init(mbar_ptr, 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if tid == 0:
            for g in T.unroll(gather_rows // 4):
                row_st: T.let = g * 4
                Tx.copy_async(
                    A_smem[row_st : row_st + 4, :],
                    A[:, :],
                    dispatch="tma",
                    mbar=mbar_ptr,
                    cta_group=1,
                    cache_hint=T.uint64(0x14F0000000000000),
                    gather_axis=0,
                    dst_gather_axis=0,
                    indexer=[Idx[g * 4 + i] for i in range(4)],
                )
            T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, smem_bytes)
        T.ptx.mbarrier.try_wait(mbar_ptr, 0)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        Tx.cta.copy(B[:, :], A_smem[:, :])
        # fmt: on

    return tma_gather4_multi_iter


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_copy_tma_gather4_multi_iter_gpu_smoke():
    dtype = "float16"
    rows = 256
    gather_rows = 8
    cols = 128
    dev = tvm.cuda(0)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": _build_tma_gather4_multi_iter_kernel(dtype)}),
            target=target,
            tir_pipeline="tirx",
        )

    np.random.seed(0)
    A_np = tvm.testing.generate_random_array(dtype, (rows, cols))
    I_np = np.array([7, 3, 19, 5, 31, 11, 2, 23], dtype="int32")
    B_np = np.zeros((gather_rows, cols), dtype=tvm.testing.np_dtype_from_str(dtype))

    A = tvm.runtime.tensor(A_np, dev)
    Idx = tvm.runtime.tensor(I_np, dev)
    B = tvm.runtime.tensor(B_np, dev)
    mod(A, Idx, B)

    np.testing.assert_allclose(A_np[I_np], B.numpy())


def test_copy_tma_gather4_dst_row_stride_soundness():
    """gather4 writes 4 rows box-linearly at the box payload width. A row-major
    destination (row stride == payload) is legal; a padded-row destination
    corrupts rows 1..3 and must be rejected at dispatch."""
    common = dict(
        g_shape=(256, 64),
        g_region=((0, 256), (0, 64)),
        s_shape=(8, 64),
        s_region=((0, 8), (0, 64)),
        gmem_layout=TileLayout(S[(256, 64) : (64, 1)]),
        config={
            "gather_axis": 0,
            "dst_gather_axis": 0,
            "indexer": [IntImm("int32", i) for i in range(8)],
            "cache_hint": IntImm("uint64", 0),
        },
    )
    _make_tma_call(smem_layout=TileLayout(S[(8, 64) : (64, 1)]), **common)
    with pytest.raises(Exception, match="payload width|corrupt"):
        _make_tma_call(smem_layout=TileLayout(S[(8, 64) : (128, 1)]), **common)


def test_copy_tma_gather4_dst_per_chunk_soundness():
    """Each 4-row chunk is written box-linearly from its OWN base (s_st+4*chunk),
    so a split layout that is uniform for rows 0..3 but discontinuous at a later
    chunk boundary must still be rejected. Non-zero s_st shifts the bases too."""
    g = dict(
        g_shape=(256, 64),
        g_region=((0, 256), (0, 64)),
        gmem_layout=TileLayout(S[(256, 64) : (64, 1)]),
    )
    split = TileLayout(S[(2, 6, 64) : (1024, 64, 1)])

    def cfg(n):
        return {
            "gather_axis": 0,
            "dst_gather_axis": 0,
            "indexer": [IntImm("int32", i) for i in range(n)],
            "cache_hint": IntImm("uint64", 0),
        }

    # Uniform (12,64) row-major: every chunk box-linear at 64. Legal.
    _make_tma_call(
        s_shape=(12, 64),
        s_region=((0, 12), (0, 64)),
        smem_layout=TileLayout(S[(12, 64) : (64, 1)]),
        config=cfg(12),
        **g,
    )
    # Valid split (4,4,64):(1024,64,1): each 4-row chunk stays within one inner-4
    # block (stride 64 == payload), so it regroups to Iter(4, 64). Legal.
    _make_tma_call(
        s_shape=(16, 64),
        s_region=((0, 16), (0, 64)),
        smem_layout=TileLayout(S[(4, 4, 64) : (1024, 64, 1)]),
        config=cfg(16),
        **g,
    )
    # Chunk bases need not form one affine region: rows 0..7 and 8..15 are
    # separated, but every emitted 4-row chunk is independently box-linear.
    _make_tma_call(
        s_shape=(16, 64),
        s_region=((0, 12), (0, 64)),
        smem_layout=TileLayout(S[(2, 8, 64) : (1024, 64, 1)]),
        config=cfg(12),
        **g,
    )
    # Split (2,6,64):(1024,64,1): rows 0..3 look box-linear, but chunk 1
    # (rows 4..7) is declared at 256,320,1024,1088 vs hw 256,320,384,448.
    with pytest.raises(Exception, match="split|4-row|straddle"):
        _make_tma_call(
            s_shape=(12, 64),
            s_region=((0, 12), (0, 64)),
            smem_layout=split,
            config=cfg(12),
            **g,
        )
    # Non-zero s_st (rows 4..11): the sliced region straddles the discontinuity.
    with pytest.raises(Exception, match="discontinuity|crosses|box-linear|split"):
        _make_tma_call(
            s_shape=(12, 64),
            s_region=((4, 12), (0, 64)),
            smem_layout=split,
            config=cfg(8),
            **g,
        )


def _build_tma_gather4_padded_src_kernel(dtype="float16"):
    """Source rows are physically ``cols_total`` apart but only ``cols`` wide.
    The descriptor row dim must carry the padded source stride, else the
    indexer would select rows at the wrong byte offset."""
    rows = 256
    copy_rows = 16
    cols = 64
    cols_total = 96
    thread_cnt = 128
    shared_layout = mma_shared_layout(dtype, 3, (copy_rows, cols))
    smem_bytes = copy_rows * cols * tvm.DataType(dtype).bits // 8

    # fmt: off
    @T.prim_func
    def tma_gather4_padded_src(A_ptr: T.handle, I_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (rows, cols_total), dtype)
        Idx = T.match_buffer(I_ptr, (copy_rows,), "int32")
        B = T.match_buffer(B_ptr, (copy_rows, cols), dtype)

        T.device_entry()
        T.cta_id([1])
        tid = T.thread_id([thread_cnt])
        dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
        A_smem = T.decl_buffer(
            (copy_rows, cols), dtype, dyn.data, elem_offset=0, layout=shared_layout
        )
        mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
        mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

        if tid == 0:
            T.ptx.mbarrier.init(mbar_ptr, 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if tid == 0:
            Tx.copy_async(
                A_smem[:, :],
                A[:, :cols],
                dispatch="tma",
                mbar=mbar_ptr,
                cta_group=1,
                cache_hint=T.uint64(0x14F0000000000000),
                gather_axis=0,
                dst_gather_axis=0,
                indexer=[Idx[i] for i in range(copy_rows)],
            )
            T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, smem_bytes)
        T.ptx.mbarrier.try_wait(mbar_ptr, 0)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        Tx.cta.copy(B[:, :], A_smem[:, :])
        # fmt: on

    return tma_gather4_padded_src


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_copy_tma_gather4_padded_src_gpu_roundtrip():
    dtype = "float16"
    rows, copy_rows, cols, cols_total = 256, 16, 64, 96
    dev = tvm.cuda(0)
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": _build_tma_gather4_padded_src_kernel(dtype)}),
            target=target,
            tir_pipeline="tirx",
        )

    np.random.seed(0)
    A_np = tvm.testing.generate_random_array(dtype, (rows, cols_total))
    I_np = np.array([7, 3, 19, 5, 31, 11, 2, 23, 43, 13, 47, 17, 59, 29, 61, 37], dtype="int32")
    B_np = np.zeros((copy_rows, cols), dtype=tvm.testing.np_dtype_from_str(dtype))

    A = tvm.runtime.tensor(A_np, dev)
    Idx = tvm.runtime.tensor(I_np, dev)
    B = tvm.runtime.tensor(B_np, dev)
    mod(A, Idx, B)

    np.testing.assert_allclose(A_np[I_np][:, :cols], B.numpy())


# Section 3: TMA special cases (symbolic dimension, buffer view)
# ===========================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize("swizzle_len", [3])
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_symbolic_dimension(dtype, swizzle_len):
    """Test TMA copy with symbolic dimension in global buffer (like hgemm pattern).

    This tests the pattern:
        Tx.copy_async(A_smem[ks, :, :], A[m_st : m_st + BLK_M, k_start : k_start + BLK_K], **tma_copy)  # noqa: E501

    Where M is a symbolic dimension in the global buffer.
    """  # noqa: E501
    # Fixed dimensions
    K = 256
    BLK_M = 64
    BLK_K = 64
    SMEM_PIPE_DEPTH = 2
    M_CONCRETE = 128  # Concrete value for testing
    thread_cnt = 128

    dev = tvm.cuda(0)

    # Shared memory layout with swizzle
    shared_layout = T.ComposeLayout(
        T.SwizzleLayout(3, swizzle_len, 3, swizzle_inner=True),
        T.TileLayout(T.S[(SMEM_PIPE_DEPTH, BLK_M, BLK_K) : (BLK_M * BLK_K, BLK_K, 1)]),
    )

    # Compute bytes for mbarrier
    smem_bytes = SMEM_PIPE_DEPTH * BLK_M * BLK_K * tvm.DataType(dtype).bits // 8
    copy_bytes = BLK_M * BLK_K * tvm.DataType(dtype).bits // 8

    # fmt: off
    @T.prim_func
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        M = T.int32()
        A = T.match_buffer(A_ptr, [M, K], dtype)
        B = T.match_buffer(B_ptr, [SMEM_PIPE_DEPTH, BLK_M, BLK_K], dtype)

        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([thread_cnt])
        dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
        A_smem = T.decl_buffer(
            [SMEM_PIPE_DEPTH, BLK_M, BLK_K], dtype, dyn.data, elem_offset=0, layout=shared_layout
        )
        mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
        mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

        if tid == 0:
            T.ptx.mbarrier.init(mbar_ptr, 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

                # Copy with pipeline index (like hgemm pattern)
        for ks in range(SMEM_PIPE_DEPTH):
            if tid == 0:
                Tx.copy_async(
                    A_smem[ks, :, :],
                    A[0:BLK_M, ks * BLK_K:(ks + 1) * BLK_K],
                    dispatch="tma",
                    mbar=mbar_ptr
                )
                T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, copy_bytes)

            T.ptx.mbarrier.try_wait(mbar_ptr, ks % 2)

        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        for ks in range(SMEM_PIPE_DEPTH):
            Tx.cta.copy(
                B[ks, :, :],
                A_smem[ks, :, :]
            )
        # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, (M_CONCRETE, K))
        B_np = np.zeros((SMEM_PIPE_DEPTH, BLK_M, BLK_K), dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        # Verify: B[ks, :, :] should equal A[0:BLK_M, ks*BLK_K:(ks+1)*BLK_K]
        B_ref = np.zeros((SMEM_PIPE_DEPTH, BLK_M, BLK_K), dtype=np_dtype)
        for ks in range(SMEM_PIPE_DEPTH):
            B_ref[ks, :, :] = A_np[0:BLK_M, ks * BLK_K : (ks + 1) * BLK_K]
        np.testing.assert_allclose(B_ref, B.numpy())


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize("swizzle_len", [3])
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_3d_with_view(dtype, swizzle_len):
    """Test 3D TMA copy using buffer view and swizzle layout (like flash attention pattern).

    This tests the pattern from FA4:
        Q_smem allocated as 4D: (SMEM_PIPE_DEPTH, NUM_BLK_K, BLK_M, BLK_K)
        Q_smem_3d = Q_smem.view(SMEM_PIPE_DEPTH, NUM_BLK_K, SEQ_TILE, GQA_RATIO, BLK_K)
        Tx.copy_async(Q_smem_3d[pipe_idx, blk_k_idx, :, :, :],
                      Q[batch, seq_start:seq_end, head_start:head_end, k_start:k_end], ...)
    """
    dev = tvm.cuda(0)
    smem_bytes = 2 * 2 * 128 * 64 * tvm.DataType(dtype).bits // 8
    copy_bytes_per_blk = 32 * 4 * 64 * tvm.DataType(dtype).bits // 8

    # Shared memory layout with swizzle
    shared_layout = T.ComposeLayout(
        T.SwizzleLayout(3, swizzle_len, 3, swizzle_inner=True),
        T.TileLayout(T.S[(2, 128, 128) : (128 * 128, 128, 1)]),
    )

    # fmt: off
    @T.prim_func
    def copy_async(Q_ptr: T.handle, B_ptr: T.handle) -> None:
        Q = T.match_buffer(Q_ptr, (2, 128, 8, 128), dtype)
        B = T.match_buffer(B_ptr, (32, 4, 64), dtype)

        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([128])
        dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
                # Allocate as 4D like FA4: (SMEM_PIPE_DEPTH, NUM_BLK_K, BLK_M, BLK_K)
        Q_smem = T.decl_buffer(
            (2, 2, 128, 64),
            dtype, dyn.data, elem_offset=0, layout=shared_layout
        )
        mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
        mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

                # Create 5D view for 3D copy pattern
        Q_smem_5d = Q_smem.view(2, 2, 32, 4, 64)

        if tid == 0:
            T.ptx.mbarrier.init(mbar_ptr, 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if tid == 0:
                    # 3D copy: [SEQ_Q_PER_TILE, GQA_RATIO, BLK_K]
            Tx.copy_async(
                Q_smem_5d[0, 0, :, :, :],
                Q[0, 0:32, 0:4, 0:64],
                dispatch="tma",
                mbar=mbar_ptr
            )
            T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, copy_bytes_per_blk)

        T.ptx.mbarrier.try_wait(mbar_ptr, 0)

        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        Tx.cta.copy(
            B[:, :, :],
            Q_smem_5d[0, 0, :, :, :]
        )
        # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")

    with target:
        mod = tvm.IRModule({"main": copy_async})

        # Verify that LowerTIRx generates exactly 1 TMA instruction
        lowered = tvm.tirx.transform.LowerTIRx()(mod)
        counter = TMACounter()
        counter.visit_stmt(lowered["main"].body)

        assert counter.total_tma_ops == 1, (
            f"Expected exactly 1 TMA operation, got {counter.total_tma_ops}. "
            "This indicates the 3D TMA copy with view is not generating optimal code."
        )

        # Now compile and verify correctness
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        Q_np = tvm.testing.generate_random_array(dtype, (2, 128, 8, 128))
        B_np = np.zeros((32, 4, 64), dtype=np_dtype)

        Q = tvm.runtime.tensor(Q_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(Q, B)

        B_ref = np.zeros((32, 4, 64), dtype=np_dtype)
        B_ref[:, :, :] = Q_np[0, 0:32, 0:4, 0:64]
        np.testing.assert_allclose(B_ref, B.numpy())


# ===========================================================================
# Section 4: TMA GPU smoke tests (end-to-end compilation + correctness)
# ===========================================================================


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize(
    "task",
    [
        # (a) Basic 2D G2S: (8,256) full region
        pytest.param(
            (
                (8, 256),  # g_shape
                ((0, 8), (0, 256)),  # g_region
                (8, 256),  # s_shape
                ((0, 8), (0, 256)),  # s_region
                8,  # thread count per CTA
                TileLayout(S[8, 256]),  # A_layout
                TileLayout(S[8, 256]),  # B_layout
                lambda dtype: mma_shared_layout(dtype, 3, (8, 256)),
            ),
            id="g2s-2d-basic",
        ),
        # (b) 3D pipeline G2S: (3,8,256) → (8,256) per-phase
        pytest.param(
            (
                (3, 8, 256),
                None,  # multi-phase: region computed per-phase
                (8, 256),
                None,  # multi-phase
                8,
                TileLayout(S[3, 8, 256]),
                TileLayout(S[3, 8, 256]),
                lambda dtype: mma_shared_layout(dtype, 3, (8, 256)),
            ),
            id="g2s-3d-pipeline",
        ),
        # (c) 4D with unit dims: (2,2,128,64), copy (1,1,128,64) → 2D shared (128,64)
        pytest.param(
            (
                (2, 2, 128, 64),
                ((0, 1), (0, 1), (0, 128), (0, 64)),
                (128, 64),
                ((0, 128), (0, 64)),
                128,
                TileLayout(S[2, 2, 128, 64]).canonicalize(),
                TileLayout(S[2, 2, 128, 64]).canonicalize(),
                lambda dtype: mma_shared_layout(dtype, 3, (128, 64)),
            ),
            id="g2s-4d-unit-dims",
        ),
    ],
)
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_gpu_smoke_g2s(task, dtype):
    """Smoke test: compile and run TMA G2S copy on GPU to verify end-to-end correctness."""
    g_shape, g_region, s_shape, s_region, thread_cnt, layoutA, layoutB, layoutS_fn = task
    dev = tvm.cuda(0)

    shared_layout = layoutS_fn(dtype)
    is_pipeline = g_region is None

    if is_pipeline:
        n = g_shape[0]
        smem_bytes = functools.reduce(lambda acc, e: acc * e, s_shape, 1)
        smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

        r_smem = [slice(0, s) for s in s_shape]

        def r_gmem(stage):
            return [
                slice(stage, stage + 1),
                *[slice(0, g_shape[i]) for i in range(1, len(g_shape))],
            ]

        # fmt: off
        @T.prim_func
        def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
            B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

            T.device_entry()
            cta_id = T.cta_id([1])
            tid = T.thread_id([thread_cnt])
            dyn = T.alloc_buffer([smem_bytes + 8], "uint8", scope="shared.dyn")
            A_smem = T.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)
            mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
            phase: T.int32

            phase = 0
            if tid == 0:
                T.ptx.mbarrier.init(mbarrier.ptr_to([0]), 1)
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.cta_sync()

            for stage in range(n):
                if tid == 0:
                    Tx.copy_async(A_smem[tuple(r_smem)], A[tuple(r_gmem(stage))], dispatch="tma", mbar=mbarrier.ptr_to([0]))  # noqa: E501
                    T.ptx.mbarrier.arrive.expect_tx(mbarrier.ptr_to([0]), smem_bytes)

                T.ptx.mbarrier.try_wait(mbarrier.ptr_to([0]), phase)
                phase = phase ^ 1

                T.ptx.fence.proxy_async("shared::cta")
                T.cuda.cta_sync()
                Tx.cta.copy(B[tuple(r_gmem(stage))], A_smem[tuple(r_smem)])
            # fmt: on

        np_dtype = tvm.testing.np_dtype_from_str(dtype)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": copy_async})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

            np.random.seed(0)
            A_np = tvm.testing.generate_random_array(dtype, g_shape)
            B_np = np.zeros(g_shape, dtype=np_dtype)

            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)
            np.testing.assert_allclose(A_np, B.numpy())
    else:
        total_bytes = functools.reduce(
            lambda acc, region: acc * (region[1] - region[0]), s_region, 1
        )
        total_bytes = total_bytes * tvm.DataType(dtype).bits // 8

        smem_bytes = functools.reduce(lambda acc, e: acc * e, s_shape, 1)
        smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

        r_smem = [slice(s_region[i][0], s_region[i][1]) for i in range(len(s_shape))]
        r_gmem = [slice(g_region[i][0], g_region[i][1]) for i in range(len(g_shape))]

        # fmt: off
        @T.prim_func
        def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
            A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
            B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

            T.device_entry()
            cta_id = T.cta_id([1])
            tid = T.thread_id([thread_cnt])
            dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
            A_smem = T.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)
            mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
            mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

            if tid == 0:
                T.ptx.mbarrier.init(mbar_ptr, 1)
            T.ptx.fence.proxy_async("shared::cta")
            T.cuda.cta_sync()

            if tid == 0:
                Tx.copy_async(A_smem[tuple(r_smem)], A[tuple(r_gmem)], dispatch="tma", mbar=mbar_ptr)  # noqa: E501
                T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, total_bytes)
            T.ptx.mbarrier.try_wait(mbar_ptr, 0)
            T.cuda.cta_sync()
            Tx.cta.copy(B[tuple(r_gmem)], A_smem[tuple(r_smem)])
            # fmt: on

        np_dtype = tvm.testing.np_dtype_from_str(dtype)
        target = tvm.target.Target("cuda")
        with target:
            mod = tvm.IRModule({"main": copy_async})
            mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

            np.random.seed(0)
            A_np = tvm.testing.generate_random_array(dtype, g_shape)
            B_np = np.zeros(g_shape, dtype=np_dtype)

            A = tvm.runtime.tensor(A_np, dev)
            B = tvm.runtime.tensor(B_np, dev)
            mod(A, B)

            B_ref = np.zeros(g_shape, dtype=np_dtype)
            B_ref[tuple(r_gmem)] = A_np[tuple(r_gmem)]
            np.testing.assert_allclose(B_ref, B.numpy())


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_gpu_smoke_s2g(dtype):
    """Smoke test: compile and run TMA S2G store on GPU."""
    g_shape = (3, 8, 256)
    s_shape = (8, 256)
    thread_cnt = 8
    n = g_shape[0]

    shared_layout = mma_shared_layout(dtype, 3, s_shape)

    smem_bytes = functools.reduce(lambda acc, e: acc * e, s_shape, 1)
    smem_bytes = smem_bytes * tvm.DataType(dtype).bits // 8

    r_smem = [slice(0, s) for s in s_shape]

    def r_gmem(stage):
        return [slice(stage, stage + 1), *[slice(0, g_shape[i]) for i in range(1, len(g_shape))]]

    layoutA = TileLayout(S[3, 8, 256])
    layoutB = TileLayout(S[3, 8, 256])

    # fmt: off
    @T.prim_func
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=layoutA)
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=layoutB)

        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([thread_cnt])
        dyn = T.alloc_buffer([smem_bytes], "uint8", scope="shared.dyn")
        A_smem = T.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)

        for stage in range(n):
            Tx.copy(A_smem[tuple(r_smem)], A[tuple(r_gmem(stage))])
            T.cuda.cta_sync()
            T.ptx.fence.proxy_async("shared::cta")
            if tid == 0:
                Tx.copy_async(B[tuple(r_gmem(stage))], A_smem[tuple(r_smem)], dispatch="tma")
                T.ptx.cp_async.bulk.commit_group()
                T.ptx.cp_async.bulk.wait_group()
            T.cuda.cta_sync()
        # fmt: on

    np_dtype = tvm.testing.np_dtype_from_str(dtype)
    target = tvm.target.Target("cuda")
    dev = tvm.cuda(0)

    with target:
        mod = tvm.IRModule({"main": copy_async})
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

        np.random.seed(0)
        A_np = tvm.testing.generate_random_array(dtype, g_shape)
        B_np = np.zeros(g_shape, dtype=np_dtype)

        A = tvm.runtime.tensor(A_np, dev)
        B = tvm.runtime.tensor(B_np, dev)
        mod(A, B)

        np.testing.assert_allclose(A_np, B.numpy())


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize("dtype", ["float16"])
def test_copy_tma_dynamic_cta_mask(dtype):
    """Regression test for B00004: dynamic cta_mask expression in TMA multicast.

    Verifies that a TIR expression (depending on T.cta_id) used as cta_mask in
    copy_async compiles through the full TIRX pipeline without crashing.
    Previously, lower_tirx_scope_ids replaced scope-ID vars via Substitute,
    but Substitute didn't visit TilePrimitiveCall.config values, leaving stale var
    references that caused MakePackedAPI to fail with:
        "variables [...] are used, but are not passed in as API arguments"
    """
    CLUSTER_SIZE = 4
    CTA_GROUP = 2
    BLK_M = 64
    BLK_K = 64
    thread_cnt = 128

    smem_shape = (BLK_M, BLK_K)
    shared_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True), T.TileLayout(T.S[smem_shape : (BLK_K, 1)])
    )
    smem_bytes = BLK_M * BLK_K * tvm.DataType(dtype).bits // 8
    copy_bytes = smem_bytes

    # fmt: off
    @T.prim_func
    def copy_async_dynamic_mask(A_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, [BLK_M, BLK_K], dtype)

        T.device_entry()
        cbx = T.cta_id_in_cluster([CLUSTER_SIZE])
        cta_id = T.cta_id([CLUSTER_SIZE])
        tid = T.thread_id([thread_cnt])

                # Dynamic cta_mask: exact expression from B00004 bug report
        cta_mask = T.meta_var(5 + 5 * cbx)
        dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
        A_smem = T.decl_buffer(
            smem_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout,
        )
        mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
        mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

        if tid == 0:
            T.ptx.mbarrier.init(mbar_ptr, 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if tid == 0:
            Tx.copy_async(
                A_smem[:, :],
                A[:, :],
                dispatch="tma",
                mbar=mbar_ptr,
                cta_mask=cta_mask,
                cta_group=CTA_GROUP,
            )
            T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, copy_bytes)

        T.ptx.mbarrier.try_wait(mbar_ptr, 0)
        # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.IRModule({"main": copy_async_dynamic_mask})
        # This compilation crashed before the B00004 fix with:
        #   "variables [...] are used, but are not passed in as API arguments"
        mod = tvm.compile(mod, target=target, tir_pipeline="tirx")

    # Verify multicast instruction was generated
    src = mod.mod.imports[0].inspect_source()
    assert "multicast" in src, "Expected multicast TMA instruction in generated code"


def test_copy_tma_uint32_shape_extent():
    BK = 64
    A_layout = mma_shared_layout("float16", 3, (128, BK))

    @T.prim_func
    def tma_load(n: T.uint32, a_ptr: T.handle, o_ptr: T.handle) -> None:
        A = T.match_buffer(a_ptr, (n, BK), "float16")
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

    target = tvm.target.Target("cuda")
    with target:
        tvm.compile(tvm.IRModule({"main": tma_load}), target=target, tir_pipeline="tirx")


def test_copy_tma_uint32_slice_base():
    BK = 64
    A_layout = mma_shared_layout("float16", 3, (128, BK))

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
            Tx.copy_async(sm[:, :], A[off : off + 128, 0:BK], dispatch="tma", mbar=mb.ptr_to([0]))
            T.ptx.mbarrier.arrive.expect_tx(mb.ptr_to([0]), 128 * BK * 2)
        T.ptx.mbarrier.try_wait(mb.ptr_to([0]), 0)
        T.cuda.cta_sync()
        reg = T.alloc_local(BK, "float16")
        Tx.copy(reg[:], sm[tid, 0:BK])
        Tx.copy(Out[tid, 0:BK], reg[:])

    target = tvm.target.Target("cuda")
    with target:
        tvm.compile(tvm.IRModule({"main": tma_off}), target=target, tir_pipeline="tirx")


def test_copy_tma_dynamic_cache_hint_g2s_keeps_rank_coords():
    dtype = "float16"
    M, H, K = 64, 4, 64
    smem_shape = (M, H, K)
    A_layout = T.ComposeLayout(
        T.SwizzleLayout(3, 3, 3, swizzle_inner=True),
        T.TileLayout(T.S[smem_shape : (H * K, K, 1)]),
    )

    @T.prim_func
    def tma_dynamic_cache_hint(a_ptr: T.handle) -> None:
        A = T.match_buffer(a_ptr, (8, M, H, K), dtype)
        T.device_entry()
        T.cta_id([1])
        T.warp_id([4])
        T.warpgroup_id([1])
        tid = T.thread_id_in_wg([128])
        sm = T.alloc_buffer(smem_shape, dtype, scope="shared", layout=A_layout)
        mb = T.alloc_shared([1], "uint64")
        if tid == 0:
            T.ptx.mbarrier.init(mb.ptr_to([0]), 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        if tid == 0:
            Tx.copy_async(
                sm[:, :, :],
                A[1:2, 0:M, 0:H, 0:K],
                dispatch="tma",
                mbar=mb.ptr_to([0]),
                cta_group=2,
                cache_hint=T.uint64(0x12F0000000000000),
            )
            T.ptx.mbarrier.arrive.expect_tx(mb.ptr_to([0]), M * H * K * 2)
        T.ptx.mbarrier.try_wait(mb.ptr_to([0]), 0)

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": tma_dynamic_cache_hint}), target=target, tir_pipeline="tirx"
        )

    src = mod.mod.imports[0].inspect_source()
    assert "ptx_cp_async_bulk_tensor_g2s_cluster_tile_4d_cache_hint_mbar_addr" in src
    assert "tvm_builtin_cuda_cvta_generic_to_shared" in src
    assert "4278190079" not in src
    assert (
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
    ) in src


def test_copy_tma_gather4_bar_addr_dynamic_cache_hint_codegen():
    @T.prim_func
    def tma_gather4(A_map: T.TensorMap()) -> None:
        T.device_entry()
        tid = T.thread_id([128])
        sm = T.alloc_buffer((64, 4), "bfloat16", scope="shared")
        mb = T.alloc_shared([1], "uint64")
        if tid == 0:
            T.ptx.mbarrier.init(mb.ptr_to([0]), 1)
            T.ptx.cp_async.bulk.tensor.g2s_cluster(
                2,
                sm.ptr_to([0, 0]),
                T.cuda.sm100_2sm_leader_smem_addr(mb.ptr_to([0])),
                T.address_of(A_map),
                0,
                2,
                "",
                0,
                1,
                2,
                3,
                4,
                cache_policy=T.uint64(0x14F0000000000000),
                load_mode="tile_gather4",
                mbar_is_shared_addr=True,
            )

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.compile(tvm.IRModule({"main": tma_gather4}), target=target, tir_pipeline="tirx")

    src = mod.mod.imports[0].inspect_source()
    helper_name = "ptx_cp_async_bulk_tensor_g2s_cluster_tile_gather4_2d_cache_hint_mbar_addr"
    assert helper_name in src
    assert (
        f"{helper_name}(void* dst, unsigned int mbar_addr, "
        "unsigned long long tensormap_addr, uint16_t cta_mask, unsigned long long cache_policy"
    ) in src
    assert "cta_group2_unicast" not in src
    assert (
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
    ) in src
    assert "4278190079" in src


def _build_tma_gather4_cta2_kernel(cta_mask):
    """2-CTA cluster gather through the dispatcher (``dispatch='tma'``,
    ``cta_group=2``, ``gather_axis=0``), mirroring the FlashMLA KV gather. The
    leader CTA (cbx==0) issues the multicast gather; ``cta_mask`` selects which
    cluster CTAs receive the gathered rows in their own smem. Both CTAs zero
    their smem first, then dump it to ``B[cbx]`` so an un-multicast CTA reads
    back all-zero."""
    dtype = "float16"
    rows, copy_rows, cols, thread_cnt = 256, 16, 64, 128
    shared_layout = mma_shared_layout(dtype, 3, (copy_rows, cols))
    smem_bytes = copy_rows * cols * tvm.DataType(dtype).bits // 8

    # fmt: off
    @T.prim_func
    def tma_gather4_cta2(A_ptr: T.handle, I_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (rows, cols), dtype)
        Idx = T.match_buffer(I_ptr, (copy_rows,), "int32")
        B = T.match_buffer(B_ptr, (2, copy_rows, cols), dtype)
        T.device_entry()
        cbx, cby = T.cta_id_in_cluster([2, 1])
        T.cta_id([2])
        tid = T.thread_id([thread_cnt])
        dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
        A_smem = T.decl_buffer(
            (copy_rows, cols), dtype, dyn.data, elem_offset=0, layout=shared_layout
        )
        mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
        mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))

        if tid == 0:
            T.ptx.mbarrier.init(mbar_ptr, 1)
        for i in range(copy_rows * cols // thread_cnt):
            A_smem[(tid + i * thread_cnt) // cols, (tid + i * thread_cnt) % cols] = T.float16(0)
        T.ptx.fence.mbarrier_init()
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        T.cuda.cluster_sync()

        if cbx == 0:
            if tid == 0:
                Tx.copy_async(
                    A_smem[:, :], A[:, :], dispatch="tma", mbar=mbar_ptr,
                    cta_group=2, cta_mask=T.uint16(cta_mask),
                    cache_hint=T.uint64(0x14F0000000000000),
                    gather_axis=0, dst_gather_axis=0, indexer=[Idx[i] for i in range(copy_rows)],
                )
                T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, smem_bytes)
            T.ptx.mbarrier.try_wait(mbar_ptr, 0)
        T.cuda.cta_sync()
        T.cuda.cluster_sync()
        Tx.cta.copy(B[cbx, :, :], A_smem[:, :])
    # fmt: on

    return tma_gather4_cta2


def _run_tma_gather4_cta2(cta_mask):
    copy_rows, cols, rows = 16, 64, 256
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_100a"})
    with target:
        mod = tvm.compile(
            tvm.IRModule({"main": _build_tma_gather4_cta2_kernel(cta_mask)}),
            target=target,
            tir_pipeline="tirx",
        )
    np.random.seed(0)
    A_np = tvm.testing.generate_random_array("float16", (rows, cols))
    I_np = np.array([7, 3, 19, 5, 31, 11, 2, 23, 43, 13, 47, 17, 59, 29, 61, 37], dtype="int32")
    dev = tvm.cuda(0)
    A = tvm.runtime.tensor(A_np, dev)
    Idx = tvm.runtime.tensor(I_np, dev)
    B = tvm.runtime.tensor(np.zeros((2, copy_rows, cols), A_np.dtype), dev)
    mod(A, Idx, B)
    return A_np[I_np], B.numpy()


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(10), reason="need cuda compute >= 10.0")
def test_copy_tma_gather4_cta_group2_gpu_roundtrip():
    """Dual-CTA cta_group=2 gather round-trip through the dispatcher (the form
    FlashMLA uses). ``cta_mask`` is the multicast destination bitmask: bit i set
    → cluster CTA i receives the gathered rows in its smem. ``cta_mask=3`` (both
    CTAs) must land the gathered rows in both; ``cta_mask=1`` (leader only) must
    land them in CTA 0 and leave CTA 1 all-zero. Verified on B200."""
    exp, both = _run_tma_gather4_cta2(cta_mask=3)
    np.testing.assert_array_equal(both[0], exp, err_msg="cta_mask=3 cta0")
    np.testing.assert_array_equal(both[1], exp, err_msg="cta_mask=3 cta1")

    exp, leader = _run_tma_gather4_cta2(cta_mask=1)
    np.testing.assert_array_equal(leader[0], exp, err_msg="cta_mask=1 cta0")
    assert not leader[1].any(), "cta_mask=1 must leave CTA 1 un-multicast (all-zero)"


def test_copy_tma_declines_non_derivable_fold_layout():
    """A copy that transposes gmem (D, H) into smem (H, D) is declined loudly.

    The two regions are 64x32 (smem) vs 32x64 (gmem): a copy is elementwise
    over one shared logical region, so a transpose must be spelled with a
    permuted buffer view, never hidden inside the copy. The mismatch is
    caught at the copy region gate (not silently mis-placed).
    """

    import pytest

    dtype = "bfloat16"
    D_QK, D_V, H, S_Q = 576, 512, 64, 4
    q_rope_layout = ComposeLayout(
        SwizzleLayout(3, 2, 3, swizzle_inner=True),
        TileLayout(S[(64, 2, 32) : (32, 2048, 1)]),
    )

    with pytest.raises(Exception, match="identical regions"):

        @T.prim_func
        def tma_permuted_extents(a_ptr: T.handle) -> None:
            A = T.match_buffer(a_ptr, (D_QK, H, S_Q), dtype)
            T.device_entry()
            T.cta_id([1])
            T.warp_id([4])
            T.warpgroup_id([1])
            tid = T.thread_id_in_wg([128])

            A_tma = A.view(
                D_QK,
                H,
                S_Q,
                layout=TileLayout(S[(D_QK, H, S_Q) : (1, D_QK, H * D_QK)]),
            )
            sm = T.alloc_buffer((64, 64), dtype, scope="shared", layout=q_rope_layout)
            mb = T.alloc_shared([1], "uint64")
            if tid == 0:
                T.ptx.mbarrier.init(mb.ptr_to([0]), 1)
                T.ptx.fence.mbarrier_init()
                Tx.copy_async(
                    sm[:, 0:32],
                    A_tma[D_V : D_V + 32, :, 1:2],
                    dispatch="tma",
                    mbar=mb.ptr_to([0]),
                    cta_group=1,
                )


def test_copy_tma_rejects_flipped_swizzle_inner():
    """A canonical-family SwizzleLayout with ``swizzle_inner=False`` must be
    rejected at planning: the TMA hardware swizzle modes implement the
    ``swizzle_inner=True`` permutation ``x ^ ((x & outer_mask) >> atom_len)``
    (pinned bit-exactly on hardware by the GPU smoke tests in this file),
    while ``swizzle_inner=False`` is the mirrored
    ``x ^ ((x & inner_mask) << atom_len)`` — extracting only ``swizzle_len``
    would silently plan the wrong placement."""
    import pytest

    canonical = mma_shared_layout("float16", 3, (8, 256))
    # Identical linear tiling and swizzle family; only the permutation
    # direction is flipped, so only a swizzle_inner check can reject it.
    flipped = ComposeLayout(
        SwizzleLayout(3, 3, 3, swizzle_inner=False),
        canonical.tile_layout,
    )
    with pytest.raises(Exception, match="swizzle_inner"):
        _make_tma_call(
            g_shape=(8, 256),
            g_region=((0, 8), (0, 256)),
            s_shape=(8, 256),
            s_region=((0, 8), (0, 256)),
            gmem_layout=TileLayout(S[8, 256]),
            smem_layout=flipped,
            dtype="float16",
        )


# FlashMLA 5D Q fold: gmem is a (64, h_q, 2, D_QK//128, s_q) strided view of a
# (s_q, h_q, D_QK) tensor.  Logical element (a, b, c, d, e) sits at linear
# gmem offset a + 512*b + 256*c + 64*d + 65536*e; head-dim index is
# k = a + 64*d + 256*c (c = 256-half, d = 64-chunk within the half).
_FLASHMLA_Q_GMEM_LAYOUT = TileLayout(S[(64, 128, 2, 4, 3) : (1, 512, 256, 64, 65536)])
_FLASHMLA_Q_G_REGION = ((0, 64), (64, 128), (0, 2), (0, 4), (1, 2))

# TMA writes the box into smem in plain box-linear order (descriptor dim 0
# fastest).  The default ("optimized") planner orders the box>1 descriptor
# dims by the declared smem layout's contiguous chain, so the hardware fill
# lands every element exactly where the declared layout says.  Two different
# declared placements of the same fold must therefore yield two different
# descriptor dim orders, and both must be element-exact on hardware.
_FLASHMLA_Q_SMEM_CASES = [
    pytest.param(
        # Interleaved halves (chunk order d0 c0 d1 c1 ...): the true FlashMLA
        # Q placement.  The derived descriptor is the FlashMLA 5D Q ABI, i.e.
        # identical to the historical FlashMLA ABI order for this gmem view.
        (1, 64, 4096, 8192),
        [5, 64, 128, 2, 4, 3, 1024, 512, 128, 131072, 64, 64, 2, 4, 1],
        id="interleaved-halves-abi",
    ),
    pytest.param(
        # Half-major placement (all of half c=0, then half c=1): the planner
        # must swap the two middle descriptor dims to honor it.
        (1, 64, 16384, 4096),
        [5, 64, 128, 4, 2, 3, 1024, 128, 512, 131072, 64, 64, 4, 2, 1],
        id="half-major",
    ),
]


@pytest.mark.parametrize("smem_strides, encode_head", _FLASHMLA_Q_SMEM_CASES)
def test_copy_tma_optimized_dim_order_derives_from_declared_smem_layout(smem_strides, encode_head):
    """Default dim order follows the declared smem placement, not gmem order."""

    smem_layout = ComposeLayout(
        SwizzleLayout(3, 3, 3, swizzle_inner=True),
        TileLayout(S[(64, 64, 2, 4) : smem_strides]),
    )
    _, host_init_stmts = _make_tma_call(
        g_shape=(64, 128, 2, 4, 3),
        g_region=_FLASHMLA_Q_G_REGION,
        s_shape=(64, 64, 2, 4),
        s_region=((0, 64), (0, 64), (0, 2), (0, 4)),
        gmem_layout=_FLASHMLA_Q_GMEM_LAYOUT,
        smem_layout=smem_layout,
        dtype="bfloat16",
    )

    expected_host = _build_expected_host_init("bfloat16", [*encode_head, 1, 1, 1, 1, 1, 0, 3, 2, 0])
    assert len(host_init_stmts) == 1
    tvm.ir.assert_structural_equal(host_init_stmts[0], expected_host, map_free_vars=True)


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda_compute(9), reason="need cuda compute >= 9.0")
@pytest.mark.parametrize("smem_strides, encode_head", _FLASHMLA_Q_SMEM_CASES)
def test_copy_tma_optimized_folded_view_placement_matches_declared_layout(
    smem_strides, encode_head
):
    """GPU: folded-view TMA fill is element-exact against the declared smem layout.

    Copies the FlashMLA-Q-shaped 5D folded gmem view into a swizzled smem view
    whose middle dims the planner reorders, then reads smem back through the
    declared layout and compares elementwise.  A descriptor dim order that
    desynchronizes from the declared placement (e.g. blindly keeping the gmem
    order for the half-major case) scrambles the two middle dims and fails.
    """
    del encode_head
    dtype = "uint16"
    g_total = 3 * 65536
    n_elems = 64 * 64 * 2 * 4
    smem_bytes = n_elems * 2
    dev = tvm.cuda(0)

    smem_layout = ComposeLayout(
        SwizzleLayout(3, 3, 3, swizzle_inner=True),
        TileLayout(S[(64, 64, 2, 4) : smem_strides]),
    )
    gmem_layout = _FLASHMLA_Q_GMEM_LAYOUT

    # fmt: off
    @T.prim_func
    def copy_async(A_ptr: T.handle, B_ptr: T.handle) -> None:
        A = T.match_buffer(A_ptr, (g_total,), dtype)
        B = T.match_buffer(B_ptr, (64, 64, 2, 4), dtype)
        T.device_entry()
        cta_id = T.cta_id([1])
        tid = T.thread_id([128])
        dyn = T.alloc_buffer([smem_bytes + 64], "uint8", scope="shared.dyn")
        A_smem = T.decl_buffer((64, 64, 2, 4), dtype, dyn.data, elem_offset=0, layout=smem_layout)
        mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
        mbar_ptr = T.meta_var(mbarrier.ptr_to([0]))
        A_tma = A.view(64, 128, 2, 4, 3, layout=gmem_layout)

        if tid == 0:
            T.ptx.mbarrier.init(mbar_ptr, 1)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()

        if tid == 0:
            Tx.copy_async(
                A_smem[:, :, :, :],
                A_tma[0:64, 64:128, 0:2, 0:4, 1:2],
                dispatch="tma",
                mbar=mbar_ptr,
            )
            T.ptx.mbarrier.arrive.expect_tx(mbar_ptr, smem_bytes)
        T.ptx.mbarrier.try_wait(mbar_ptr, 0)
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.cta_sync()
        Tx.cta.copy(B[:, :, :, :], A_smem[:, :, :, :])
    # fmt: on

    target = tvm.target.Target("cuda")
    with target:
        mod = tvm.compile(tvm.IRModule({"main": copy_async}), target=target, tir_pipeline="tirx")

    # Value == linear gmem offset (mod 2^16); every copied element is unique.
    A_np = (np.arange(g_total) % 65536).astype(np.uint16)
    B_np = np.zeros((64, 64, 2, 4), dtype=np.uint16)
    A_t = tvm.runtime.tensor(A_np, dev)
    B_t = tvm.runtime.tensor(B_np, dev)
    mod(A_t, B_t)

    a = np.arange(64).reshape(64, 1, 1, 1)
    b = np.arange(64).reshape(1, 64, 1, 1)
    c = np.arange(2).reshape(1, 1, 2, 1)
    d = np.arange(4).reshape(1, 1, 1, 4)
    B_ref = ((a + 512 * (64 + b) + 256 * c + 64 * d + 65536) % 65536).astype(np.uint16)
    np.testing.assert_array_equal(B_ref, B_t.numpy())


# ---------------------------------------------------------------------------
# Regression tests: merge+promote
# soundness, promoted-unit validation, unfixable-alignment declines, and the
# tensormap cache key.
# ---------------------------------------------------------------------------


def _tensormap_encode_calls(stmt):
    """Collect (dtype_str, ndim, int_args) for each cuTensorMapEncodeTiled call."""

    class _Collector(StmtExprVisitor):
        def __init__(self):
            super().__init__()
            self.calls = []

        def visit_call_(self, op):
            if (
                isinstance(op.op, tvm.ir.Op)
                and op.op.name == "tirx.tvm_call_packed"
                and len(op.args) >= 5
                and isinstance(op.args[0], StringImm)
                and op.args[0].value == "runtime.cuTensorMapEncodeTiled"
            ):
                self.calls.append(
                    (
                        op.args[2].value,
                        int(op.args[3]),
                        [int(a) for a in op.args[5:] if isinstance(a, IntImm)],
                    )
                )
            super().visit_call_(op)

    collector = _Collector()
    collector.visit_stmt(stmt)
    return collector.calls


def test_copy_tma_merge_promote_positive_pin():
    """Positive pin for the merge+promote path (noted it had zero
    test pins repo-wide): a full uint8 (64, 8) copy has a non-innermost byte
    stride of 8 (< 16), the direct merge is blocked by boxDim > 256
    (64*8 = 512), so the plan promotes uint8 -> uint16 and then merges to a
    single rank-1 dim of shape/box 256 uint16 elements (= the original 512
    bytes)."""
    impl, host_init_stmts = _make_tma_call(
        g_shape=(64, 8),
        g_region=((0, 64), (0, 8)),
        s_shape=(64, 8),
        s_region=((0, 64), (0, 8)),
        gmem_layout=TileLayout(S[64, 8]),
        smem_layout=TileLayout(S[64, 8]),
        dtype="uint8",
    )
    assert _count_tma_ops(impl) == 1
    assert len(host_init_stmts) == 1
    encodes = _tensormap_encode_calls(host_init_stmts[0])
    assert encodes == [("uint16", 1, [256, 256, 1, 0, 0, 2, 0])]


def test_copy_tma_promote_declines_odd_box_and_coord():
    """``try_promote`` halves the innermost box and coord_base, so
    both must be provably even. An odd box used to be floordiv'd (silently
    dropping the last element) and an odd coord_base mis-addressed by one
    element; the promotion must now be declined, leaving the plan in the
    original dtype."""
    from tvm.arith import Analyzer
    from tvm.tirx.cuda.operator.tile_primitive.copy_async.tma import (
        DescDim,
        TmaPlan,
        _merge_contig_full_box_dims,
    )
    from tvm.tirx.cuda.operator.tile_primitive.tma_utils import SwizzleMode

    def _plan(inner_box, inner_coord):
        # d1's byte stride (8) violates the 16B rule -> merge machinery runs;
        # (d0, d1) merge is blocked only by box (8*64 = 512 > 256) -> promote
        # is attempted on the innermost dim.
        return TmaPlan(
            swizzle_mode=SwizzleMode.SWIZZLE_NONE,
            dims=[
                DescDim(shape=8, stride=512, box=8, coord_base=0),
                DescDim(shape=64, stride=8, box=64, coord_base=0),
                DescDim(shape=8, stride=1, box=inner_box, coord_base=inner_coord),
            ],
            issue_axes=[],
            tensor_ptr=Var("p", "handle"),
            elem_bytes=1,
            elem_dtype="uint8",
        )

    # Even box/coord: promotion happens (uint8 -> uint16 at least).
    promoted = _merge_contig_full_box_dims(_plan(8, 0), Analyzer())
    assert promoted.elem_dtype != "uint8"

    # Odd box: promotion declined, plan stays uint8 with the box untouched.
    kept = _merge_contig_full_box_dims(_plan(5, 0), Analyzer())
    assert kept.elem_dtype == "uint8"
    assert int(kept.dims[-1].box) == 5

    # Odd coord_base: promotion declined as well.
    kept2 = _merge_contig_full_box_dims(_plan(4, 1), Analyzer())
    assert kept2.elem_dtype == "uint8"
    assert int(kept2.dims[-1].coord_base) == 1


def test_copy_tma_declines_odd_box_promotion_end_to_end():
    """Negative, end-to-end: a copy whose innermost box is odd
    (5 of 8 uint8 columns) cannot be promoted and the 8-byte non-innermost
    stride cannot be merged away, so the dispatch must decline loudly
    instead of encoding a tensormap that halves the odd box."""
    with pytest.raises(Exception, match="all chain prefix lengths rejected"):
        _make_tma_call(
            g_shape=(8, 64, 8),
            g_region=((0, 8), (0, 64), (0, 5)),
            s_shape=(8, 64, 5),
            s_region=((0, 8), (0, 64), (0, 5)),
            gmem_layout=TileLayout(S[8, 64, 8]),
            smem_layout=TileLayout(S[8, 64, 5]),
            dtype="uint8",
        )


def test_copy_tma_validate_hw_constraints_uses_promoted_dtype():
    """the swizzle-atom box-fit check must compare the plan's box
    (promoted units) against the atom width in ``plan.elem_dtype`` units. A
    box of 128 uint16 elements spans 256B and does NOT fit the 128B swizzle
    atom; validating it against the original uint8 buffer dtype (atom width
    128 elements) used to accept it and fail late in the host wrapper."""
    from tvm.tirx.cuda.operator.tile_primitive.copy_async.tma import (
        DescDim,
        TmaPlan,
        _validate_hw_constraints,
    )
    from tvm.tirx.cuda.operator.tile_primitive.tma_utils import SwizzleMode

    plan = TmaPlan(
        swizzle_mode=SwizzleMode.SWIZZLE_128B_ATOM,
        dims=[DescDim(shape=128, stride=1, box=128, coord_base=0)],
        issue_axes=[],
        tensor_ptr=Var("p", "handle"),
        elem_bytes=2,
        elem_dtype="uint16",  # promoted from a uint8 buffer
    )
    ok, reason = _validate_hw_constraints(plan)
    assert not ok
    assert "swizzle atom" in reason

    # The same box in a plan that was NOT promoted (uint8) fits: 128 x 1B.
    plan_u8 = TmaPlan(
        swizzle_mode=SwizzleMode.SWIZZLE_128B_ATOM,
        dims=[DescDim(shape=128, stride=1, box=128, coord_base=0)],
        issue_axes=[],
        tensor_ptr=Var("p", "handle"),
        elem_bytes=1,
        elem_dtype="uint8",
    )
    ok_u8, _ = _validate_hw_constraints(plan_u8)
    assert ok_u8


def test_copy_tma_declines_illegal_boxdim():
    """cuTensorMap forbids boxDim[i] > 256 and requires boxDim[0]*elementSize
    to be a multiple of 16 B (the host wrapper ICHECKs both). ``_validate_hw_
    constraints`` must decline such plans at dispatch — a native oversized box
    has no merge/promote fold and a sub-16-byte inner box cannot be widened —
    instead of emitting a descriptor the host rejects late in init."""
    from tvm.tirx.cuda.operator.tile_primitive.copy_async.tma import (
        DescDim,
        TmaPlan,
        _validate_hw_constraints,
    )
    from tvm.tirx.cuda.operator.tile_primitive.tma_utils import SwizzleMode

    def plan(dims, elem_bytes=1, dtype="uint8"):
        return TmaPlan(
            swizzle_mode=SwizzleMode.SWIZZLE_NONE,
            dims=dims,
            issue_axes=[],
            tensor_ptr=Var("p", "handle"),
            elem_bytes=elem_bytes,
            elem_dtype=dtype,
        )

    # boxDim > 256 (single contiguous dim): only the per-dim <=256 check catches it.
    ok, reason = _validate_hw_constraints(plan([DescDim(512, 1, 512, 0)]))
    assert not ok and "256" in reason
    # boxDim > 256 on a NON-innermost dim, inner box 16B-legal: same check.
    ok, reason = _validate_hw_constraints(plan([DescDim(300, 16, 300, 0), DescDim(16, 1, 16, 0)]))
    assert not ok and "256" in reason
    # boxDim[0]*elementSize not a multiple of 16 (7 uint8 = 7 B): declined.
    ok, reason = _validate_hw_constraints(plan([DescDim(7, 1, 7, 0)]))
    assert not ok and "multiple of 16" in reason
    # Legal: box 256 uint8 = 256 B (<=256 and 16B-aligned) passes both checks.
    ok, _ = _validate_hw_constraints(plan([DescDim(256, 1, 256, 0)]))
    assert ok


def test_copy_tma_declines_oversized_box_end_to_end():
    """A contiguous uint8[512] copy needs a single boxDim of 512 (> 256, no
    fold), so it is declined at dispatch rather than failing the host wrapper's
    ICHECK at kernel launch."""
    with pytest.raises(Exception, match="256|multiple of 16"):
        _make_tma_call(
            g_shape=(512,),
            g_region=((0, 512),),
            s_shape=(512,),
            s_region=((0, 512),),
            gmem_layout=TileLayout(S[(512,)]),
            smem_layout=TileLayout(S[(512,)]),
            dtype="uint8",
        )


def test_copy_tma_oob_nan_declined_after_promotion():
    """Same family: ``oob='nan'`` is validated against the buffer
    dtype before planning, but merge+promote re-types the descriptor as
    uintN, which the host wrapper rejects for NaN fill. The dispatch must
    decline instead of failing late in host init."""
    kwargs = dict(
        g_shape=(128, 4),
        g_region=((0, 128), (0, 4)),
        s_shape=(128, 4),
        s_region=((0, 128), (0, 4)),
        gmem_layout=TileLayout(S[128, 4]),
        smem_layout=TileLayout(S[128, 4]),
        dtype="float16",
    )
    # Sanity: the same copy without oob promotes fine (fp16 -> uint32).
    _, host_init_stmts = _make_tma_call(**kwargs)
    encodes = _tensormap_encode_calls(host_init_stmts[0])
    assert encodes == [("uint32", 1, [256, 256, 1, 0, 0, 2, 0])]

    with pytest.raises(Exception, match="floating-point tensormap dtype"):
        _make_tma_call(config={"oob": "nan"}, **kwargs)


def test_copy_tma_declines_unfixable_alignment_at_dispatch():
    """When the merge cannot fix an unaligned plan (here: the innermost dim is
    partially boxed, 7 of 8 uint8 columns, so no merge is legal), the plan must
    be declined at dispatch — enabling variant fallback — rather than accepted
    and failed late by the host wrapper's 16-byte ICHECK. The partial box
    leaves boxDim[0]*elementSize un-16B-aligned, so the boxDim[0] check
    declines it."""
    with pytest.raises(Exception, match="not a multiple of 16"):
        _make_tma_call(
            g_shape=(64, 8),
            g_region=((0, 64), (0, 7)),
            s_shape=(64, 7),
            s_region=((0, 64), (0, 7)),
            gmem_layout=TileLayout(S[64, 8]),
            smem_layout=TileLayout(S[64, 7]),
            dtype="uint8",
        )


def test_copy_tma_tensormap_cache_key_includes_promotion_dtype():
    """two copies over the same tensor pointer whose plans differ
    only in promotion level collide on every numeric cache-key field: a
    (32, 8) uint8 view merges un-promoted to shape/box 256 uint8, while the
    (64, 8) view promotes once and merges to shape/box 256 uint16. The cache
    key must keep them apart (two distinct encodes, uint8 + uint16), not
    alias the second copy to the first (half-sized) tensormap."""
    from tvm.ir import PointerType, PrimType
    from tvm.tirx.exec_scope import ExecScope
    from tvm.tirx.tile_primitive import DispatchContext

    data = Var("A", PointerType(PrimType("uint8"), "global"))
    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a"})
    sctx = DispatchContext(target, ExecScope("thread"), {}, {})

    _, host1 = _make_tma_call(
        g_shape=(32, 8),
        g_region=((0, 32), (0, 8)),
        s_shape=(32, 8),
        s_region=((0, 32), (0, 8)),
        gmem_layout=TileLayout(S[32, 8]),
        smem_layout=TileLayout(S[32, 8]),
        dtype="uint8",
        g_data=data,
        sctx=sctx,
    )
    assert [e[0] for e in _tensormap_encode_calls(host1[0])] == ["uint8"]

    _, host2 = _make_tma_call(
        g_shape=(64, 8),
        g_region=((0, 64), (0, 8)),
        s_shape=(64, 8),
        s_region=((0, 64), (0, 8)),
        gmem_layout=TileLayout(S[64, 8]),
        smem_layout=TileLayout(S[64, 8]),
        dtype="uint8",
        g_data=data,
        sctx=sctx,
    )
    assert len(host2) == 2, "second copy must NOT reuse the first tensormap"
    encodes = [_tensormap_encode_calls(st) for st in host2]
    assert [e[0][0] for e in encodes] == ["uint8", "uint16"]
    # Identical numeric fields — only the element dtype separates the keys.
    assert encodes[0][0][2] == encodes[1][0][2] == [256, 256, 1, 0, 0, 2, 0]


if __name__ == "__main__":
    tvm.testing.main()
