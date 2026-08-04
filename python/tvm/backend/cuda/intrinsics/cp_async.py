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
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments, too-many-locals, too-many-positional-arguments
"""PTX cp.async / cp.async.bulk / cp.async.bulk.tensor intrinsics.

Each PTX form table entry is registered as one ``device_intrinsic``.
User-facing wrappers in ``tvm.tirx.op`` expose separate PTX instruction
families; ``register_codegen`` dispatchers below decode attrs such as
``load_mode`` / ``cta_group`` / ``multicast`` to pick the right form variant.
Bodies are hand-written ``asm volatile(...)`` strings.  The file is grouped
as cp.async, cp.async.bulk.tensor, cp.async.bulk non-TMA, and CUDA
compatibility helpers.
"""

from tvm.backend.cuda.op import cuda_func_call

from ._schema import device_intrinsic
from .registry import CODEGEN_REGISTRY, register_codegen
from .utils import parse_str

_PREFETCH_CHOICES = ("", "64", "128", "256")


def _safe(s):
    return s.replace("::", "_").replace(".", "_")


# =============================================================================
# cp.async forms from the PTX Syntax block.
#
# Non-bulk shared/global copy forms (commit/wait moved to ptxd).
# =============================================================================

# cp.async non-bulk copy forms:
#   Form 1: cp.async.ca.shared.global ... [dst], [src], cp-size{, src-size}{, cache-policy}
#   Form 2: cp.async.cg.shared.global ... [dst], [src], 16{, src-size}{, cache-policy}
#   Form 3: cp.async.ca.shared.global ... [dst], [src], cp-size{, ignore-src}{, cache-policy}
#   Form 4: cp.async.cg.shared.global ... [dst], [src], 16{, ignore-src}{, cache-policy}


def _cp_async_modifier_str(has_cache_hint, prefetch_size):
    s = ""
    if has_cache_hint:
        s += ".L2::cache_hint"
    if prefetch_size:
        s += f".L2::{prefetch_size}B"
    return s


def _make_form_parts(ca_or_cg, fixed_cp_size, extra):
    """Build a parts callable for one of the cp.async PTX forms.

    Args layout: (dst, src [, extra_int], cache_policy, has_cache, prefetch_size [, cp_size_attr])
    Forwarded operands: dst, src [, extra_int], cache_policy.
    Trailing attrs: has_cache, prefetch_size [, cp_size if .ca].
    """
    n_op = 3 if extra is not None else 2
    n_attrs = 2 if fixed_cp_size is not None else 3
    extra_in_name = f"_with_{extra}" if extra is not None else ""

    def _parts(*args):
        # Operand args (forwarded) come first, then attr args.
        raw_address = str(args[0].ty) == "uint32"
        attr_args = args[-n_attrs:]
        has_cache = _bool_attr(attr_args[0])
        prefetch_size = parse_str(attr_args[1])
        cp_size = fixed_cp_size if fixed_cp_size is not None else int(attr_args[2])
        modifier = _cp_async_modifier_str(has_cache, prefetch_size)
        cache_operand = ', "l"(cache_policy)' if has_cache else ""
        # name parts
        name_cache = "_cache_hint" if has_cache else ""
        name_prefetch = f"_prefetch_{prefetch_size}" if prefetch_size else ""
        name = (
            f"tvm_builtin_ptx_cp_async_{ca_or_cg}_{cp_size}"
            f"{name_cache}{name_prefetch}{extra_in_name}"
            f"{'_raw_u32' if raw_address else ''}"
        )
        dst_type = "unsigned int" if raw_address else "void*"
        sig = (
            f"({dst_type} dst, void* src"
            + (f", int {extra}" if extra else "")
            + ", unsigned long long cache_policy)"
        )
        address_decl = (
            "" if raw_address else "    unsigned int dst_addr = __cvta_generic_to_shared(dst);\n"
        )
        address = "dst" if raw_address else "dst_addr"
        instr_base = f"cp.async.{ca_or_cg}.shared.global{modifier}"
        if extra is None:
            cache_arg = ", %2" if has_cache else ""
            body = (
                address_decl
                + f'    asm volatile("{instr_base} [%0], [%1], {cp_size}{cache_arg};\\n"\n'
                f'                 :: "r"({address}), "l"(src){cache_operand} : "memory");'
            )
        else:
            cache_arg = ", %3" if has_cache else ""
            body = (
                address_decl
                + f'    asm volatile("{instr_base} [%0], [%1], {cp_size}, %2{cache_arg};\\n"\n'
                f'                 :: "r"({address}), "l"(src), "r"({extra})'
                f'{cache_operand} : "memory");'
            )
        return name, sig, body

    return _parts, n_op + n_attrs - n_op  # n_attrs


def _register_nb_form(op_name, ca_or_cg, fixed_cp_size, extra):
    parts_fn, n_attrs = _make_form_parts(ca_or_cg, fixed_cp_size, extra)
    n_op = 3 if extra is not None else 2
    device_intrinsic(
        f"ptx_cp_async_{op_name}",
        n_attrs=n_attrs,
        c_signature=lambda *a, fn=parts_fn: fn(*a)[1],
        helper_name=lambda *a, fn=parts_fn: fn(*a)[0],
        body=lambda *a, fn=parts_fn: fn(*a)[2],
    )
    return n_op


# Form 1: .ca + src-size (cp-size ∈ {4, 8}). src-size is required when present.
_register_nb_form("ca_src_size", "ca", fixed_cp_size=None, extra="src_size")
# Form 2: .cg + src-size (cp-size = 16).
_register_nb_form("cg_src_size", "cg", fixed_cp_size=16, extra="src_size")
# Form 3: .ca + ignore-src.
_register_nb_form("ca_ignore_src", "ca", fixed_cp_size=None, extra="ignore_src")
# Form 4: .cg + ignore-src.
_register_nb_form("cg_ignore_src", "cg", fixed_cp_size=16, extra="ignore_src")
# Plain degenerate of forms 1+2 with optional src-size omitted.
_register_nb_form("ca", "ca", fixed_cp_size=None, extra=None)
_register_nb_form("cg", "cg", fixed_cp_size=16, extra=None)


def _make_setp_at_p_helper(ca_or_cg, cp_size, has_cache, prefetch, raw_address):
    """Wrapper convenience: ``setp+@p`` around a form 1/2 cp.async (predicate-
    gated skip with dst untouched on false). Not a PTX form — emitted directly
    here as a one-off helper rather than a separate device_intrinsic."""
    modifier = _cp_async_modifier_str(has_cache, prefetch)
    cache_arg = ", %4" if has_cache else ""
    cache_operand = ', "l"(cache_policy)' if has_cache else ""
    func_name = (
        f"tvm_builtin_ptx_cp_async_{cp_size}"
        + ("_cache_hint" if has_cache else "")
        + (f"_prefetch_{prefetch}" if prefetch else "")
        + "_predicate"
        + ("_raw_u32" if raw_address else "")
    )
    dst_type = "unsigned int" if raw_address else "void*"
    address_decl = (
        "" if raw_address else "  unsigned int dst_addr = __cvta_generic_to_shared(dst);\n"
    )
    address = "dst" if raw_address else "dst_addr"
    body = (
        address_decl + "  __asm__ __volatile__(\n"
        '    "{\\n"\n'
        '    " .reg .pred p;\\n"\n'
        '    " setp.eq.u32 p, %3, 1;\\n"\n'
        f'    " @p cp.async.{ca_or_cg}.shared.global{modifier}'
        f' [%0], [%1], %2{cache_arg};\\n"\n'
        '    "}\\n"\n'
        f'    :: "r"({address}), "l"(src), "n"({cp_size}), "r"(predicate){cache_operand}\n'
        "  );"
    )
    source_code = (
        f"\n__forceinline__ __device__ void {func_name}"
        f"({dst_type} dst, void* src, int predicate, unsigned long long cache_policy) {{\n"
        f"{body}\n"
        "}\n"
    )
    return func_name, source_code


@register_codegen("ptx_cp_async")
def codegen_ptx_cp_async(*args):
    """Map the wrapper API to the 4 PTX form table entries.

    Accepts three call shapes (sorted by arity):

    * 5 args ``(dst_ptr, dst_offset, src_ptr, src_offset, cp_size)`` —
      the legacy form emitted by
      ``tvm.backend.cuda.transform.InjectPTXAsyncCopy``.
      Offsets are folded into the pointers via ``tvm_access_ptr`` (in
      bytes; offsets are pre-scaled by the pass) and the call is
      forwarded with default cache / predicate / fill_mode.
    * 6 args ``(dst_ptr, dst_offset, src_ptr, src_offset, cp_size,
      predicate)`` — same as 5-arg form with an explicit predicate,
      zero-filling the destination when the predicate is false.
    * 8 args ``(dst_ptr, src_ptr, cp_size, cache_policy, has_cache_hint,
      prefetch_size, predicate, fill_mode)`` — the fork-native wrapper
      API.

    The three resulting form_kinds:

    * ``fill_mode == "zero"`` -> form 1/2 (src-size = predicate ? cp_size : 0)
    * ``predicate != -1`` and no fill_mode -> form 1/2 wrapped in setp+@p
      (wrapper convenience; not a PTX form)
    * else -> form 1/2 with src-size omitted (the "plain" degenerate)
    """
    from tvm.tirx.op import if_then_else

    if len(args) in (5, 6):
        # Legacy InjectPTXAsyncCopy emission: (dst_ptr, dst_off, src_ptr,
        # src_off, cp_size [, predicate]). Offsets are element indices into
        # the typed buffers (the pass uses index_factor=1 except for the
        # shared.dyn-merged byte-buffer path). Emit a C helper that scales
        # the offset by the buffer element size, then runs cp.async.
        #
        # PTX plain form for both .ca and .cg is just
        # ``cp.async.<v>.shared.global [dst], [src], cp_size;`` — three
        # operands, no trailing src-size / cache-policy.
        from tvm import DataType

        dst_ptr_in, dst_offset, src_ptr_in, src_offset, cp_size = args[:5]
        predicate = args[5] if len(args) == 6 else -1
        cp_size_v = int(cp_size)
        ca_or_cg = "cg" if cp_size_v == 16 else "ca"

        # Recover the per-side element dtype from each pointer's type
        # type (Var has ty = PointerType(PrimType(dtype))).
        # InjectPTXAsyncCopy emits offsets in element-units of each side's
        # buffer dtype (dst gets dst_offset * src_elem_size only when dst is a
        # merged shared.dyn byte buffer, in which case dst_elem_dtype is uint8
        # and the resulting scale-by-1 is a no-op).
        def _elem_bytes(ptr):
            ta = getattr(ptr, "ty", None)
            if ta is None or getattr(ta, "element_type", None) is None:
                return 1
            et = ta.element_type
            if not hasattr(et, "dtype"):
                return 1
            bits = DataType(str(et.dtype)).bits
            assert bits % 8 == 0, f"non-byte element dtype: {et.dtype}"
            return bits // 8

        dst_elem_bytes = _elem_bytes(dst_ptr_in)
        src_elem_bytes = _elem_bytes(src_ptr_in)
        has_predicate = not (
            (isinstance(predicate, int) and predicate == -1)
            or (hasattr(predicate, "value") and int(predicate.value) == -1)
        )

        def _scale(n):
            return "" if n == 1 else f" * {n}"

        dst_scale = _scale(dst_elem_bytes)
        src_scale = _scale(src_elem_bytes)
        if has_predicate:
            func_name = (
                f"ptx_cp_async_legacy_pred_{ca_or_cg}_{cp_size_v}_{dst_elem_bytes}_{src_elem_bytes}"
            )
            if cp_size_v == 4:
                zero_fill = '    " @!p st.shared.u32 [%0], {%4};\\n"\n'
            elif cp_size_v == 8:
                zero_fill = '    " @!p st.shared.v2.u32 [%0], {%4, %4};\\n"\n'
            elif cp_size_v == 16:
                zero_fill = '    " @!p st.shared.v4.u32 [%0], {%4, %4, %4, %4};\\n"\n'
            else:
                raise ValueError(f"unsupported legacy predicated cp.async size: {cp_size_v}")
            body = (
                f"  uint8_t* dst_p = (uint8_t*)dst + dst_off{dst_scale};\n"
                f"  uint8_t* src_p = (uint8_t*)src + src_off{src_scale};\n"
                "  unsigned int dst_addr = __cvta_generic_to_shared(dst_p);\n"
                "  __asm__ __volatile__(\n"
                '    "{\\n"\n'
                '    " .reg .pred p;\\n"\n'
                '    " setp.eq.u32 p, %3, 1;\\n"\n'
                f'    " @p cp.async.{ca_or_cg}.shared.global'
                ' [%0], [%1], %2;\\n"\n'
                f"{zero_fill}"
                '    "}\\n"\n'
                f'    :: "r"(dst_addr), "l"(src_p), "n"({cp_size_v}), "r"(predicate), "r"(0)\n'
                "  );"
            )
            source_code = (
                f"\n__forceinline__ __device__ void {func_name}"
                "(void* dst, int dst_off, void* src, int src_off, int predicate) {\n"
                f"{body}\n"
                "}\n"
            )
            return cuda_func_call(
                func_name,
                dst_ptr_in,
                dst_offset,
                src_ptr_in,
                src_offset,
                predicate,
                source_code=source_code,
            )
        # No predicate — plain cp.async.
        func_name = f"ptx_cp_async_legacy_{ca_or_cg}_{cp_size_v}_{dst_elem_bytes}_{src_elem_bytes}"
        body = (
            f"  uint8_t* dst_p = (uint8_t*)dst + dst_off{dst_scale};\n"
            f"  uint8_t* src_p = (uint8_t*)src + src_off{src_scale};\n"
            "  unsigned int dst_addr = __cvta_generic_to_shared(dst_p);\n"
            f'  asm volatile("cp.async.{ca_or_cg}.shared.global'
            ' [%0], [%1], %2;"\n'
            f'    :: "r"(dst_addr), "l"(src_p), "n"({cp_size_v}));'
        )
        source_code = (
            f"\n__forceinline__ __device__ void {func_name}"
            "(void* dst, int dst_off, void* src, int src_off) {\n"
            f"{body}\n"
            "}\n"
        )
        return cuda_func_call(
            func_name,
            dst_ptr_in,
            dst_offset,
            src_ptr_in,
            src_offset,
            source_code=source_code,
        )
    elif len(args) == 8:
        (
            dst_ptr,
            src_ptr,
            cp_size,
            cache_policy,
            has_cache_hint,
            prefetch_size,
            predicate,
            fill_mode,
        ) = args
    else:
        raise ValueError(f"ptx_cp_async codegen expects 5/6/8 args, got {len(args)}")

    cp_size_v = int(cp_size)
    ca_or_cg = "cg" if cp_size_v == 16 else "ca"
    pref = "" if int(prefetch_size) == -1 else str(int(prefetch_size))
    fill = parse_str(fill_mode)
    has_cache = _bool_attr(has_cache_hint)
    has_predicate = not (
        (isinstance(predicate, int) and predicate == -1)
        or (hasattr(predicate, "value") and int(predicate.value) == -1)
    )

    if fill == "zero":
        src_size = if_then_else(predicate != 0, cp_size_v, 0)
        op = f"tirx.ptx_cp_async_{ca_or_cg}_src_size"
        if cp_size_v == 16:
            args = [dst_ptr, src_ptr, src_size, cache_policy, has_cache, pref]
        else:
            args = [dst_ptr, src_ptr, src_size, cache_policy, has_cache, pref, cp_size_v]
        result = CODEGEN_REGISTRY[op](args)
        return result[0] if isinstance(result, tuple) else result

    if has_predicate:
        raw_address = str(dst_ptr.ty) == "uint32"
        func_name, source_code = _make_setp_at_p_helper(
            ca_or_cg, cp_size_v, has_cache, pref, raw_address
        )
        return cuda_func_call(
            func_name, dst_ptr, src_ptr, predicate, cache_policy, source_code=source_code
        )

    # Plain — form 1/2 with src-size omitted.
    op = f"tirx.ptx_cp_async_{ca_or_cg}"
    if cp_size_v == 16:
        args = [dst_ptr, src_ptr, cache_policy, has_cache, pref]
    else:
        args = [dst_ptr, src_ptr, cache_policy, has_cache, pref, cp_size_v]
    result = CODEGEN_REGISTRY[op](args)
    return result[0] if isinstance(result, tuple) else result


CODEGEN_REGISTRY["tirx.ptx.cp_async_raw"] = CODEGEN_REGISTRY["tirx.ptx.cp_async"]


# =============================================================================
# cp.async.bulk non-TMA forms from the PTX Syntax block. Each form is one
# device_intrinsic; optional PTX modifiers are attrs, not separate fixed ops.
# =============================================================================


def _bool_attr(value):
    return bool(int(value)) if hasattr(value, "value") else bool(value)


def _bulk_cache_operand_constraint(has_cache):
    return ', "l"(cache_policy)' if has_cache else ""


def _bulk_cache_operand_suffix(has_cache):
    return ".L2::cache_hint" if has_cache else ""


# PTX cp.async.bulk global -> shared::cta form:
#   cp.async.bulk.dst.src.completion_mechanism{.level::cache_hint}{.ignore_oob}
#       [dstMem], [srcMem], size{, ignoreBytesLeft, ignoreBytesRight}, [mbar] {, cache-policy}
#   .dst = {.shared::cta}; .src = {.global}
#   .completion_mechanism = {.mbarrier::complete_tx::bytes}
#   .level::cache_hint = {.L2::cache_hint}
def _bulk_g2s_cta_parts(*args):
    has_cache = _bool_attr(args[-2])
    ignore_oob = _bool_attr(args[-1])
    instr = (
        "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
        f"{_bulk_cache_operand_suffix(has_cache)}{'.ignore_oob' if ignore_oob else ''}"
    )
    if ignore_oob:
        asm_args = (
            '"r"(dst), "l"(src_ptr), "r"(num_bytes), "r"(ignore_bytes_left), '
            '"r"(ignore_bytes_right), "r"(mbarrier)'
        )
        operands = "%2, %3, %4, [%5]"
        cache_slot = ", %6" if has_cache else ""
    else:
        asm_args = '"r"(dst), "l"(src_ptr), "r"(num_bytes), "r"(mbarrier)'
        operands = "%2, [%3]"
        cache_slot = ", %4" if has_cache else ""
    body = (
        "    unsigned int dst = (unsigned int)__cvta_generic_to_shared(dst_ptr);\n"
        "    unsigned int mbarrier = (unsigned int)__cvta_generic_to_shared(mbarrier_ptr);\n"
        f'    asm volatile("{instr} [%0], [%1], {operands}{cache_slot};"\n'
        "                 :\n"
        f"                 : {asm_args}{_bulk_cache_operand_constraint(has_cache)}\n"
        '                 : "memory");'
    )
    name = (
        "tvm_builtin_ptx_cp_async_bulk_g2s_cta"
        f"{'_cache_hint' if has_cache else ''}{'_ignore_oob' if ignore_oob else ''}"
    )
    return name, body


device_intrinsic(
    "ptx_cp_async_bulk_g2s_cta",
    n_attrs=2,
    helper_name=lambda *a: _bulk_g2s_cta_parts(*a)[0],
    c_signature=(
        "(void* dst_ptr, void* src_ptr, unsigned int num_bytes, "
        "unsigned int ignore_bytes_left, unsigned int ignore_bytes_right, "
        "void* mbarrier_ptr, unsigned long long cache_policy)"
    ),
    body=lambda *a: _bulk_g2s_cta_parts(*a)[1],
)


# PTX cp.async.bulk global -> shared::cluster form:
#   cp.async.bulk.dst.src.completion_mechanism{.multicast}{.level::cache_hint}
#       [dstMem], [srcMem], size, [mbar] {, ctaMask} {, cache-policy}
#   .dst = {.shared::cluster}; .src = {.global}
#   .completion_mechanism = {.mbarrier::complete_tx::bytes}
#   .level::cache_hint = {.L2::cache_hint}
#   .multicast = {.multicast::cluster}
def _bulk_g2s_cluster_parts(*args):
    has_cache = _bool_attr(args[-2])
    multicast = _bool_attr(args[-1])
    instr = (
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        f"{'.multicast::cluster' if multicast else ''}{_bulk_cache_operand_suffix(has_cache)}"
    )
    cta_constraint = ', "h"(cta_mask)' if multicast else ""
    mask_slot = ", %4" if multicast else ""
    cache_slot = ", %5" if multicast and has_cache else ", %4" if has_cache else ""
    body = (
        "    unsigned int dst = (unsigned int)__cvta_generic_to_shared(dst_ptr);\n"
        "    unsigned int mbarrier = (unsigned int)__cvta_generic_to_shared(mbarrier_ptr);\n"
        f'    asm volatile("{instr} [%0], [%1], %2, [%3]'
        f'{mask_slot}{cache_slot};"\n'
        "                 :\n"
        '                 : "r"(dst), "l"(src_ptr), "r"(num_bytes), "r"(mbarrier)'
        f"{cta_constraint}{_bulk_cache_operand_constraint(has_cache)}\n"
        '                 : "memory");'
    )
    name = (
        "tvm_builtin_ptx_cp_async_bulk_g2s_cluster"
        f"{'_multicast' if multicast else ''}{'_cache_hint' if has_cache else ''}"
    )
    return name, body


device_intrinsic(
    "ptx_cp_async_bulk_g2s_cluster",
    n_attrs=2,
    helper_name=lambda *a: _bulk_g2s_cluster_parts(*a)[0],
    c_signature=(
        "(void* dst_ptr, void* src_ptr, unsigned int num_bytes, "
        "void* mbarrier_ptr, unsigned short cta_mask, unsigned long long cache_policy)"
    ),
    body=lambda *a: _bulk_g2s_cluster_parts(*a)[1],
)


# PTX cp.async.bulk shared::cta -> shared::cluster form:
#   cp.async.bulk.dst.src.completion_mechanism [dstMem], [srcMem], size, [mbar]
#   .dst = {.shared::cluster}; .src = {.shared::cta}
#   .completion_mechanism = {.mbarrier::complete_tx::bytes}
device_intrinsic(
    "ptx_cp_async_bulk_s2s_cluster",
    helper_name="tvm_builtin_ptx_cp_async_bulk_s2s_cluster",
    c_signature="(uint64_t dst, void* src, int size, uint64_t mbar)",
    body=r"""    unsigned int dst_addr = static_cast<unsigned int>(dst);
    unsigned int src_addr = __cvta_generic_to_shared(src);
    unsigned int mbar_addr = static_cast<unsigned int>(mbar);
    asm volatile(
        "cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :
        : "r"(dst_addr), "r"(src_addr), "r"(size), "r"(mbar_addr)
        : "memory");""",
)


@register_codegen("ptx_cp_async_bulk_shared_to_cluster")
def codegen_ptx_cp_async_bulk_shared_to_cluster(dst_ptr, src_ptr, size, mbar):
    result = CODEGEN_REGISTRY["tirx.ptx_cp_async_bulk_s2s_cluster"]([dst_ptr, src_ptr, size, mbar])
    return result[0] if isinstance(result, tuple) else result


# PTX cp.async.bulk shared::cta -> global form:
#   cp.async.bulk.dst.src.completion_mechanism{.level::cache_hint}{.cp_mask}
#       [dstMem], [srcMem], size {, cache-policy} {, byteMask}
#   .dst = {.global}; .src = {.shared::cta}
#   .completion_mechanism = {.bulk_group}
#   .level::cache_hint = {.L2::cache_hint}
def _bulk_s2g_parts(*args):
    has_cache = _bool_attr(args[-2])
    cp_mask = _bool_attr(args[-1])
    if cp_mask and not has_cache:
        raise ValueError("cp.async.bulk shared::cta -> global .cp_mask requires .L2::cache_hint")
    instr = f"cp.async.bulk.global.shared::cta.bulk_group{_bulk_cache_operand_suffix(has_cache)}"
    if cp_mask:
        instr += ".cp_mask"
    cache_slot = ", %3" if has_cache else ""
    mask_slot = ", %4" if cp_mask else ""
    mask_constraint = ', "r"(byte_mask)' if cp_mask else ""
    body = (
        "    unsigned int src = (unsigned int)__cvta_generic_to_shared(src_ptr);\n"
        f'    asm volatile("{instr} [%0], [%1], %2'
        f'{cache_slot}{mask_slot};"\n'
        "                 :\n"
        '                 : "l"(dst_ptr), "r"(src), "r"(num_bytes)'
        f"{_bulk_cache_operand_constraint(has_cache)}{mask_constraint}\n"
        '                 : "memory");'
    )
    name = (
        "tvm_builtin_ptx_cp_async_bulk_s2g"
        f"{'_cache_hint' if has_cache else ''}{'_cp_mask' if cp_mask else ''}"
    )
    return name, body


device_intrinsic(
    "ptx_cp_async_bulk_s2g",
    n_attrs=2,
    helper_name=lambda *a: _bulk_s2g_parts(*a)[0],
    c_signature=(
        "(void* dst_ptr, void* src_ptr, unsigned int num_bytes, "
        "unsigned int byte_mask, unsigned long long cache_policy)"
    ),
    body=lambda *a: _bulk_s2g_parts(*a)[1],
)
