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

"""Explicit fixed-width vector copy dispatches."""

from tvm.arith.analyzer import Analyzer
from tvm.runtime import DataType
from tvm.script import tirx as T
from tvm.tirx import Buffer, PrimFunc
from tvm.tirx.operator.tile_primitive.dispatcher import predicate, register_dispatch
from tvm.tirx.operator.tile_primitive.registry import DispatchContext
from tvm.tirx.stmt import BufferRegion, TilePrimitiveCall

from ._common import copy_ptx_form, copy_ptx_ld_return_type
from .utils import _scope_allowed


def _region_start(buffer_region: BufferRegion):
    return [r.min for r in buffer_region.region]


def _region_elements(buffer_region: BufferRegion):
    product = 1
    for r in buffer_region.region:
        product *= r.extent
    return product


def _can_prove_equal(lhs, rhs) -> bool:
    if isinstance(lhs, int) and isinstance(rhs, int):
        return lhs == rhs
    return Analyzer().can_prove_equal(lhs, rhs)


def _ptx_space(scope: str) -> str:
    if scope.startswith("shared"):
        return "shared"
    return scope


def _is_forced_vec_copy(
    op_call: TilePrimitiveCall,
    sctx: DispatchContext,
    *,
    variant: str,
    num_bytes: int,
):
    if getattr(op_call, "dispatch", None) != variant:
        return False, f"requires explicit dispatch={variant!r}"
    if sctx.scope_kind != "thread":
        return False, f"expected thread exec_scope, got {sctx.scope_kind}"

    scope_ok, scope_reason = _scope_allowed(op_call, sctx)
    if not scope_ok:
        return False, scope_reason

    op_call = TilePrimitiveCall.downcast(op_call)
    src: Buffer = op_call.src.buffer
    dst: Buffer = op_call.dst.buffer
    if src.dtype != dst.dtype:
        return False, f"dtype mismatch: src={src.dtype}, dst={dst.dtype}"

    elem_bits = DataType(src.dtype).bits
    width_bits = num_bytes * 8
    if width_bits % elem_bits != 0:
        return False, f"{variant} is not an integral number of {src.dtype} elements"
    expected_elements = width_bits // elem_bits

    src_elements = _region_elements(op_call.src)
    dst_elements = _region_elements(op_call.dst)
    if not _can_prove_equal(src_elements, expected_elements):
        return False, f"src region does not contain exactly {expected_elements} elements"
    if not _can_prove_equal(dst_elements, expected_elements):
        return False, f"dst region does not contain exactly {expected_elements} elements"
    return True, None


def _emit_forced_vec_copy(op_call: TilePrimitiveCall, _sctx: DispatchContext, num_bytes: int):
    op_call = TilePrimitiveCall.downcast(op_call)
    src: Buffer = op_call.src.buffer
    dst: Buffer = op_call.dst.buffer
    src_scope = src.scope()
    dst_scope = dst.scope()
    src_is_local = src_scope == "local"
    dst_is_local = dst_scope == "local"
    src_space = _ptx_space(src_scope)
    dst_space = _ptx_space(dst_scope)
    src_ptr = src.ptr_to(_region_start(op_call.src))
    dst_ptr = dst.ptr_to(_region_start(op_call.dst))
    elem_bits = DataType(src.dtype).bits
    n_elements = num_bytes * 8 // elem_bits
    vec, ptx_type = copy_ptx_form(num_bytes)
    return_type = copy_ptx_ld_return_type(ptx_type)

    # fmt: off
    @T.prim_func(check_well_formed=False)
    def impl():
        if src_is_local:
            T.ptx.st(dst_ptr, src=src_ptr, space=dst_space, vec=vec, ptx_type=ptx_type)
        elif dst_is_local:
            T.ptx.ld(src_ptr, return_type, ptx_type, dst=dst_ptr, space=src_space, vec=vec)
        else:
            tmp = T.alloc_local((n_elements,), src.dtype)
            tmp_ptr = tmp.ptr_to([0])
            T.ptx.ld(src_ptr, return_type, ptx_type, dst=tmp_ptr, space=src_space, vec=vec)
            T.ptx.st(dst_ptr, src=tmp_ptr, space=dst_space, vec=vec, ptx_type=ptx_type)
    # fmt: on
    return impl


def _register_forced_vec_copy(variant: str, num_bytes: int) -> None:
    @register_dispatch(
        "copy",
        "cuda",
        variant=variant,
        priority=20,
        when=[
            predicate(
                f"{variant}_applicable",
                _is_forced_vec_copy,
                variant=variant,
                num_bytes=num_bytes,
            )
        ],
    )
    def _copy_schedule_forced_vec(
        op_call: TilePrimitiveCall,
        sctx: DispatchContext,
        _num_bytes=num_bytes,
    ) -> PrimFunc:
        return _emit_forced_vec_copy(op_call, sctx, _num_bytes)


_register_forced_vec_copy("vec_128b", 16)
_register_forced_vec_copy("vec_64b", 8)
_register_forced_vec_copy("vec_32b", 4)
_register_forced_vec_copy("vec_16b", 2)
