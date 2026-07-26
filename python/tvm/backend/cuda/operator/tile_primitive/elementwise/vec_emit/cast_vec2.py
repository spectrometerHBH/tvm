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

"""Packed vec_len=2 cast via PTX ``cvt``."""

from __future__ import annotations

from tvm.ir.expr import Expr
from tvm.script import tirx as T

from .._common import dtype_name
from ..ops import VecImpl

_VEC2_CAST_PAIRS = {
    ("float32", "float16"),
    ("float16", "float32"),
    ("bfloat16", "float32"),
    ("float32", "bfloat16"),
}


def _cast_vec2_applies(op_call, sctx, plan):
    if len(plan.srcs) != 1 or plan.srcs[0].is_scalar:
        return False, "cast requires 1 buffer src"
    src = plan.srcs[0]
    if src.index_fn is not None:
        return False, "broadcasting src not supported by cast vec2"
    src_dtype = dtype_name(src.buf_region.buffer.dtype)
    dst_dtype = dtype_name(plan.dst.buffer.dtype)
    if (src_dtype, dst_dtype) not in _VEC2_CAST_PAIRS:
        return False, f"no vec2 PTX cvt form for {src_dtype}->{dst_dtype}"
    return True, None


def _emit_cast_vec2(dst_buf, dst_lane_indices, src_args, extras) -> Expr:
    src_arg = src_args[0]
    # cast_vec2 requires buffer src (guarded by applies()).
    assert isinstance(src_arg, tuple), "cast vec2 src must be a buffer"
    src_buf, src_lane_indices = src_arg
    src_dtype = dtype_name(src_buf.dtype)
    dst_dtype = dtype_name(dst_buf.dtype)
    if src_dtype == "float32":
        # PTX packed conversion operands are high lane first, so reverse the
        # logical (x, y) pair to preserve CUDA float2/half2 lane order.
        return T.ptx.cvt(
            src_buf[tuple(src_lane_indices[1])],
            src_buf[tuple(src_lane_indices[0])],
            dtype="f16x2" if dst_dtype == "float16" else "bf16x2",
            atype="f32",
            dst=T.address_of(dst_buf[tuple(dst_lane_indices[0])]),
            rounding="rn",
        )

    return T.ptx.cvt(
        T.reinterpret("uint16", src_buf[tuple(src_lane_indices[0])]),
        dtype="f32",
        atype="f16" if src_dtype == "float16" else "bf16",
        dst=T.address_of(dst_buf[tuple(dst_lane_indices[0])]),
    )


def _emit_cast_vec2_second(dst_buf, dst_lane_indices, src_args, extras) -> Expr:
    src_buf, src_lane_indices = src_args[0]
    src_dtype = dtype_name(src_buf.dtype)
    if src_dtype == "float32":
        return T.int32(0)
    return T.ptx.cvt(
        T.reinterpret("uint16", src_buf[tuple(src_lane_indices[1])]),
        dtype="f32",
        atype="f16" if src_dtype == "float16" else "bf16",
        dst=T.address_of(dst_buf[tuple(dst_lane_indices[1])]),
    )


CAST_VEC2_IMPL = VecImpl(
    vec_len=2,
    applies=_cast_vec2_applies,
    emit=_emit_cast_vec2,
    emit_second=_emit_cast_vec2_second,
)
