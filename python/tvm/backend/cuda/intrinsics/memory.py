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
# ruff: noqa: E501
# pylint: disable=redefined-builtin, invalid-name, too-many-arguments
"""Memory ops (load / store / copy / atomic / address conversion / type punning).

PTX side:
* ``ld.acquire.scope{.ss}.type`` scalar load forms.
* ``ld.volatile{.ss}.type`` scalar load forms.
* Legacy ``ld.global.acquire.gpu`` / ``ld.global.cg`` result-argument helper.
* ``mapa.u64`` — map a SMEM ptr to a peer CTA's SMEM in the cluster.

CUDA side:
* ``__ldg`` (cache-as-read-only load).
* Templated ``atomicAdd`` / ``atomicCAS``.
* half↔float type-punned conversions (single, packed, batch-of-8).
* ``__cvta_generic_to_shared`` and ``cluster_addr → shared u32`` casts.
"""

from tvm import DataType
from tvm.backend.cuda.op import cuda_func_call

from ._schema import device_intrinsic
from .registry import register_codegen
from .utils import parse_str

# __ldg — typed read-only cached load. Source is ``void*`` so callers may pass
# a typed pointer or handle_add_byte_offset result; the helper casts per ``dtype``.
_CUDA_LDG_CTYPES = {
    "int8": "signed char",
    "uint8": "unsigned char",
    "int16": "short",
    "uint16": "unsigned short",
    "int32": "int",
    "uint32": "unsigned int",
    "int64": "long long",
    "uint64": "unsigned long long",
    "float16": "half",
    "bfloat16": "nv_bfloat16",
    "float32": "float",
    "float64": "double",
}

_CUDA_LDG_VECTOR_CTYPES = {
    "int32": "int",
    "uint32": "unsigned int",
    "float32": "float",
}
_CUDA_LDG_VECTOR_BASES = {"int32": "int", "uint32": "uint", "float32": "float"}


def _cuda_ldg_suffix(dtype: str) -> str:
    return dtype.replace("float", "f").replace("uint", "u").replace("int", "i")


@register_codegen("cuda_ldg")
def codegen_cuda_ldg(*args):
    if len(args) == 2:
        addr, dtype = args
        dtype = str(DataType(parse_str(dtype)))
        if dtype not in _CUDA_LDG_CTYPES:
            raise ValueError(f"Unsupported CUDA __ldg dtype {dtype!r}")
        c_type = _CUDA_LDG_CTYPES[dtype]
        func_name = f"tvm_builtin_cuda_ldg_{_cuda_ldg_suffix(dtype)}"
        source_code = f"""
__forceinline__ __device__ {c_type} {func_name}(void* src) {{
    return __ldg(reinterpret_cast<const {c_type}*>(src));
}}
"""
        return cuda_func_call(func_name, addr, source_code=source_code, return_type=dtype)

    if len(args) < 5:
        raise ValueError(f"cuda_ldg expects 2 args or vector form, got {len(args)}")
    *dsts, addr, dtype, vec, dst_count = args
    dtype = str(DataType(parse_str(dtype)))
    vec = parse_str(vec)
    dst_count = _int_attr(dst_count)
    vec_len = int(vec[1:]) if vec else 1
    if dtype not in _CUDA_LDG_VECTOR_CTYPES:
        raise ValueError(f"Unsupported vector CUDA __ldg dtype {dtype!r}")
    if vec not in ("v2", "v4") or dst_count != vec_len or len(dsts) != vec_len:
        raise ValueError(
            f"vector CUDA __ldg expects dst_count=len(dsts)=vec_len for v2/v4, "
            f"got vec={vec!r}, dst_count={dst_count}, len(dsts)={len(dsts)}"
        )
    c_type = _CUDA_LDG_VECTOR_CTYPES[dtype]
    vec_type = f"{_CUDA_LDG_VECTOR_BASES[dtype]}{vec_len}"
    members = ("x", "y", "z", "w")[:vec_len]
    func_name = f"tvm_builtin_cuda_ldg_{_cuda_ldg_suffix(dtype)}_{vec}_to_dst{dst_count}"
    params = ", ".join(f"void* dst{i}" for i in range(vec_len))
    stores = "\n".join(
        f"    *reinterpret_cast<{c_type}*>(dst{i}) = v.{member};"
        for i, member in enumerate(members)
    )
    source_code = f"""
__forceinline__ __device__ void {func_name}({params}, void* src) {{
    {vec_type} v = __ldg(reinterpret_cast<const {vec_type}*>(src));
{stores}
}}
"""
    return cuda_func_call(func_name, *dsts, addr, source_code=source_code, return_type="void")


# Shared PTX scalar type metadata (ld/st/red/atom).
_PTX_SCALAR_TYPE_INFO = {
    "b8": ("unsigned int", "r", "uint32"),
    "u8": ("unsigned int", "r", "uint32"),
    "s8": ("int", "r", "int32"),
    "b16": ("unsigned short", "h", "uint16"),
    "u16": ("unsigned short", "h", "uint16"),
    "s16": ("short", "h", "int16"),
    "b32": ("unsigned int", "r", "uint32"),
    "u32": ("unsigned int", "r", "uint32"),
    "s32": ("int", "r", "int32"),
    "b64": ("unsigned long long", "l", "uint64"),
    "u64": ("unsigned long long", "l", "uint64"),
    "s64": ("long long", "l", "int64"),
    "f32": ("float", "f", "float32"),
    "f64": ("double", "d", "float64"),
}
_PTX_LD_TYPE_RETURNS = {
    "b32": {"uint32": "unsigned int", "int32": "int"},
    "b64": {"uint64": "unsigned long long", "int64": "long long"},
}
_PTX_VEC_STORE_TYPE = {
    16: "uint4",
    8: "uint2",
    4: "unsigned int",
    2: "unsigned short",
    1: "unsigned char",
}
_PTX_ST_SRC_CTYPES = {
    "b8": "unsigned char",
    "u8": "unsigned char",
    "s8": "signed char",
    "b16": "unsigned short",
    "u16": "unsigned short",
    "s16": "short",
    "b32": "unsigned int",
    "u32": "unsigned int",
    "s32": "int",
    "b64": "unsigned long long",
    "u64": "unsigned long long",
    "s64": "long long",
    "f32": "float",
    "f64": "double",
}
_PTX_ST_SRC_VECTOR_CTYPES = {
    "b8": "uchar",
    "u8": "uchar",
    "s8": "char",
    "b16": "ushort",
    "u16": "ushort",
    "s16": "short",
    "b32": "uint",
    "u32": "uint",
    "s32": "int",
    "f32": "float",
    "f64": "double",
}


def _safe_attr(value):
    return parse_str(value).replace("::", "_").replace(".", "_")


def _dot(value):
    value = parse_str(value)
    return f".{value}" if value else ""


def _cache_suffix(cache):
    return ".L2::cache_hint" if cache else ""


def _type_info(ptx_type):
    ptx_type = parse_str(ptx_type)
    if ptx_type not in _PTX_SCALAR_TYPE_INFO:
        raise ValueError(
            f"Unsupported PTX scalar type {ptx_type!r}; expected {sorted(_PTX_SCALAR_TYPE_INFO)}"
        )
    return (ptx_type, *_PTX_SCALAR_TYPE_INFO[ptx_type])


# =============================================================================
# PTX ld forms (ISA table entries registered via ``ptx_ld`` and siblings):
#   ld{.weak}{.ss}{.cop}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};
#   ld{.weak}{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};
#   ld.global{.cop}.nc{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};
#   ld.global.nc{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};
#   ld.volatile{.ss}{.level::prefetch_size}{.vec}.type  d, [a];
#   ld.relaxed.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};
#   ld.acquire.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.level::prefetch_size}{.vec}.type  d, [a]{, cache-policy};
#   ld.mmio.sem.sys{.global}.type  d, [a];
# =============================================================================
_PTX_LD_SCOPES = {"cta", "cluster", "gpu", "sys"}
_PTX_LD_SPACES = {"global", "shared", "shared::cta", "shared::cluster", "local"}
_PTX_LD_VOLATILE_SPACES = _PTX_LD_SPACES | {"const"}
_PTX_LD_WEAK_SPACES = _PTX_LD_SPACES | {"const", "param::entry", "param::func"}
_PTX_LD_COPS = {"", "ca", "cg", "cs", "lu", "cv"}
_PTX_LD_GLOBAL_NC_COPS = {"", "ca", "cg", "cs"}
_PTX_VEC = {"", "v2", "v4", "v8"}
_PTX_L1_EVICT = {
    "",
    "L1::evict_normal",
    "L1::evict_unchanged",
    "L1::evict_first",
    "L1::evict_last",
    "L1::no_allocate",
}
_PTX_L2_EVICT = {"", "L2::evict_normal", "L2::evict_first", "L2::evict_last"}
_PTX_PREFETCH = {"", "L2::64B", "L2::128B", "L2::256B"}


def _bool_attr(value):
    return bool(int(value)) if hasattr(value, "value") else bool(value)


def _int_attr(value):
    return int(value.value) if hasattr(value, "value") else int(value)


def _parse_ld_attrs(return_dtype, ptx_type, scope=None, space="global"):
    return_dtype = parse_str(return_dtype)
    ptx_type = parse_str(ptx_type)
    scope = None if scope is None else parse_str(scope)
    space = parse_str(space)
    ptx_type, _ptx, constraint, default_tvm = _type_info(ptx_type)
    if ptx_type in _PTX_LD_TYPE_RETURNS:
        returns = _PTX_LD_TYPE_RETURNS[ptx_type]
        if return_dtype not in returns:
            raise ValueError(
                f"PTX ld type {ptx_type!r} cannot return TVM dtype {return_dtype!r}; "
                f"expected one of {sorted(returns)}"
            )
        c_type = returns[return_dtype]
    else:
        if return_dtype != default_tvm:
            raise ValueError(
                f"PTX ld type {ptx_type!r} cannot return TVM dtype {return_dtype!r}; "
                f"expected {default_tvm!r}"
            )
        c_type = _ptx
    if scope is not None and scope not in _PTX_LD_SCOPES:
        raise ValueError(
            f"Unsupported PTX ld scope {scope!r}; expected one of {sorted(_PTX_LD_SCOPES)}"
        )
    return return_dtype, ptx_type, scope, space, c_type, constraint


def _validate_ld_space(space: str, allowed: set[str]) -> None:
    if space not in allowed:
        raise ValueError(
            f"Unsupported PTX ld state space {space!r}; expected one of {sorted(allowed)}"
        )


def _ptx_level_suffix(has_cache, l1_evict, l2_evict, prefetch_size):
    suffix = _cache_suffix("cache" if has_cache else "")
    l1_evict = parse_str(l1_evict)
    l2_evict = parse_str(l2_evict)
    prefetch_size = parse_str(prefetch_size)
    if l1_evict:
        suffix += f".{l1_evict}"
    if l2_evict:
        suffix += f".{l2_evict}"
    if prefetch_size:
        suffix += f".{prefetch_size}"
    return suffix


def _ptx_global_nc_level_suffix(has_cache, l1_evict, l2_evict, prefetch_size):
    suffix = ""
    l1_evict = parse_str(l1_evict)
    l2_evict = parse_str(l2_evict)
    prefetch_size = parse_str(prefetch_size)
    if l1_evict:
        suffix += f".{l1_evict}"
    if l2_evict:
        suffix += f".{l2_evict}"
    suffix += _cache_suffix("cache" if has_cache else "")
    if prefetch_size:
        suffix += f".{prefetch_size}"
    return suffix


def _ptx_shared_addr(space, ptr_name="address"):
    if parse_str(space).startswith("shared"):
        return (
            f"    unsigned int addr = (unsigned int)__cvta_generic_to_shared({ptr_name});\n",
            '"r"(addr)',
        )
    return "", f'"l"({ptr_name})'


def _ptx_addr_operand(space, ptr_name, raw_address):
    if raw_address:
        return "", f'"r"({ptr_name})'
    return _ptx_shared_addr(space, ptr_name)


def _is_raw_u32_address(address):
    return str(address.ty) == "uint32"


# =============================================================================
# Atomics — templated wrappers around CUDA's ``atomicAdd`` / ``atomicCAS``.
# =============================================================================
device_intrinsic(
    "cuda_atomic_add",
    helper_name="tvm_builtin_cuda_atomic_add",
    c_signature="(T* addr, T value)",
    body="    return atomicAdd(addr, value);",
    return_type="T",
    templated=True,
    tvm_return_type=lambda _addr, value: value.ty,
)
device_intrinsic(
    "cuda_atomic_cas",
    helper_name="tvm_builtin_cuda_atomic_cas",
    c_signature="(T* address, T compare, T val)",
    body="    return atomicCAS(address, compare, val);",
    return_type="T",
    templated=True,
    tvm_return_type=lambda _p, old, _n: old.ty,
)


# =============================================================================
# half / bfloat16 ↔ float type-punned conversions.
# =============================================================================
device_intrinsic(
    "cuda_half2float",
    c_signature="(half src)",
    body="    return __half2float(src);",
    return_type="float",
    tvm_return_type="float32",
)
device_intrinsic(
    "cuda_bfloat162float",
    c_signature="(nv_bfloat16 src)",
    body="    return __bfloat162float(src);",
    return_type="float",
    tvm_return_type="float32",
)
device_intrinsic(
    "cuda_float22half2",
    c_signature="(void* dst, void* src)",
    body=(
        "    half2* dst_p = (half2*) dst;\n"
        "    float2* src_p = (float2*) src;\n"
        "    *dst_p = __float22half2_rn(*src_p);"
    ),
)
device_intrinsic(
    "cuda_half8tofloat8",
    c_signature="(void* src_addr, void* dst_addr)",
    body=(
        "    half2* source = (half2*) src_addr;\n"
        "    float2* dest = (float2*) dst_addr;\n"
        "    for (int i = 0; i < 4; i++) {\n"
        "        dest[i] = __half22float2(source[i]);\n"
        "    }"
    ),
)
device_intrinsic(
    "cuda_float8tohalf8",
    c_signature="(void* src_addr, void* dst_addr)",
    body=(
        "    float2* source = (float2*) src_addr;\n"
        "    half2* dest = (half2*) dst_addr;\n"
        "    for (int i = 0; i < 4; i++) {\n"
        "        dest[i] = __float22half2_rn(source[i]);\n"
        "    }"
    ),
)


# =============================================================================
# Address-conversion helpers used by op-wrapper-side dispatch in tvm.tirx.op.
# Each precomputes a value that the schema's specialized op then takes as a
# typed scalar input (instead of doing the conversion inside the asm helper).
# =============================================================================
device_intrinsic(
    "cuda_cvta_generic_to_shared",
    c_signature="(void* p)",
    body="    return __cvta_generic_to_shared(p);",
    return_type="unsigned int",
    tvm_return_type="uint32",
)

device_intrinsic(
    "cuda_smem_addr_from_uint64",
    c_signature="(uint64_t cluster_addr)",
    body="    return static_cast<unsigned int>(cluster_addr);",
    return_type="unsigned int",
    tvm_return_type="uint32",
)

# =============================================================================
# PTX mapa form:
#   mapa{.space}.type d, a, b;
#   .space = {.shared::cluster}; .type = {.u32, .u64}


# =============================================================================
# Generic PTX memory forms. Compatibility wrappers in ``tvm.tirx.op`` bind
# concrete sem/scope/space/op/type parameters for existing call sites.
# =============================================================================


# PTX st forms (ISA table entries registered via ``ptx_st`` and siblings):
#   st{.weak}{.ss}{.cop}{.level::cache_hint}{.vec}.type [a], b{, cache-policy};
#   st{.weak}{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.vec}.type [a], b{, cache-policy};
#   st.volatile{.ss}{.vec}.type [a], b;
#   st.relaxed.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.vec}.type [a], b{, cache-policy};
#   st.release.scope{.ss}{.level1::eviction_priority}{.level2::eviction_priority}{.level::cache_hint}{.vec}.type [a], b{, cache-policy};
#   st.mmio.sem.sys{.global}.type [a], b;
_PTX_ST_COPS = {"", "wb", "cg", "cs", "wt"}
_PTX_ST_SPACES = {"global", "shared", "shared::cta", "shared::cluster", "local", "param::func"}


def _ptx_st_load_src(_num_bytes, vec_len, ptx_type, c_type):
    src_c_type = _PTX_ST_SRC_CTYPES[ptx_type]

    def _assign(dst, src):
        if src_c_type == c_type:
            return f"    {c_type} {dst} = {src};\n"
        return f"    {c_type} {dst} = static_cast<{c_type}>({src});\n"

    if vec_len > 1:
        vec_base = _PTX_ST_SRC_VECTOR_CTYPES.get(ptx_type)
        if vec_base is not None and vec_len in (2, 3, 4):
            vec_type = f"{vec_base}{vec_len}"
            loads = "".join(_assign(f"r{i}", f"src_.{c}") for i, c in enumerate("xyzw"[:vec_len]))
            return f"    {vec_type} src_ = *reinterpret_cast<{vec_type}*>(src_ptr);\n{loads}"

        loads = "".join(_assign(f"r{i}", f"src[{i}]") for i in range(vec_len))
        return f"    {src_c_type}* src = reinterpret_cast<{src_c_type}*>(src_ptr);\n{loads}"

    return _assign("r0", f"*reinterpret_cast<{src_c_type}*>(src_ptr)")


def _ptx_st_form_parts(form, attr_args, from_src, raw_address):
    if form == "weak":
        weak, space, cop, vec, ptx_type, has_cache_hint, l1_evict, l2_evict = attr_args
        sem, scope = "", ""
    elif form == "relaxed":
        scope, space, vec, ptx_type, has_cache_hint, l1_evict, l2_evict = attr_args
        sem, weak, cop = "relaxed", False, ""
    elif form == "release":
        scope, space, vec, ptx_type, has_cache_hint, l1_evict, l2_evict = attr_args
        sem, weak, cop = "release", False, ""
    elif form == "volatile":
        space, vec, ptx_type = attr_args
        sem, scope, weak, cop = "", "", False, ""
        has_cache_hint, l1_evict, l2_evict = False, "", ""
    elif form == "mmio":
        sem, scope, space, ptx_type = attr_args
        weak, cop, vec = False, "", ""
        has_cache_hint, l1_evict, l2_evict = False, "", ""
    else:
        raise ValueError(f"unknown st form {form!r}")

    sem = parse_str(sem)
    scope = parse_str(scope)
    space = parse_str(space)
    cop = parse_str(cop)
    vec = parse_str(vec)
    l1_evict = parse_str(l1_evict)
    l2_evict = parse_str(l2_evict)
    weak = _bool_attr(weak)
    has_cache = _bool_attr(has_cache_hint)
    ptx_type = parse_str(ptx_type)
    if ptx_type == "b128":
        # The b128 form mirrors the FlashMLA dequantization store exactly.
        # Its source pointer materializes one scalar 128-bit register for the
        # PTX ``q`` constraint.  Unlike generic scalar/vector stores, this is
        # weak shared::cta only, accepts no cache policy, and deliberately
        # omits a memory clobber to preserve the source dependency shape.
        if not (
            form == "weak"
            and weak
            and space == "shared::cta"
            and not cop
            and not vec
            and from_src
            and not has_cache
            and not l1_evict
            and not l2_evict
        ):
            raise ValueError(
                "PTX st.b128 requires src=, weak=True, space='shared::cta', "
                "and no cache/vector modifiers"
            )
        name = "tvm_builtin_ptx_st_weak_shared_cta_b128_from_src"
        if raw_address:
            name += "_raw_u32"
        address_type = "unsigned int" if raw_address else "void*"
        addr_decl, addr_operand = _ptx_addr_operand(space, "address", raw_address)
        body = (
            f"{addr_decl}"
            "    unsigned __int128 value = *reinterpret_cast<unsigned __int128*>(src_ptr);\n"
            '    asm volatile("st.weak.shared::cta.b128 [%0], %1;"\n'
            "                 :\n"
            f'                 : {addr_operand}, "q"(value));'
        )
        return (
            name,
            (f"({address_type} address, void* src_ptr, unsigned long long cache_policy)"),
            body,
        )
    ptx_type, c_type, constraint, _tvm_dtype = _type_info(ptx_type)
    if cop and cop not in _PTX_ST_COPS:
        raise ValueError(f"Unsupported PTX st cache operation {cop!r}")
    if vec and vec not in _PTX_VEC:
        raise ValueError(f"Unsupported PTX st vector modifier {vec!r}")
    if space not in _PTX_ST_SPACES and not (form == "mmio" and space == "global"):
        raise ValueError(f"Unsupported PTX st state space {space!r}")
    vec_len = int(vec[1:]) if vec else 1
    elem_bytes = (
        8
        if ptx_type.endswith("64")
        else 2
        if ptx_type in ("u16", "s16", "b16")
        else 1
        if ptx_type in ("u8", "s8", "b8")
        else 4
    )
    num_bytes = vec_len * elem_bytes if vec else elem_bytes
    use_cache_policy = form in ("weak", "relaxed", "release")
    if form == "mmio":
        if sem not in ("acquire", "relaxed", "release") or scope != "sys" or space != "global":
            raise ValueError("st.mmio requires sem, scope=sys, space=global")
        prefix = f"st.mmio.{sem}.{scope}"
    elif form == "relaxed":
        if not scope:
            raise ValueError("st.relaxed requires scope")
        prefix = f"st.relaxed.{scope}{_dot(space)}"
    elif form == "release":
        if not scope:
            raise ValueError("st.release requires scope")
        prefix = f"st.release.{scope}{_dot(space)}"
    elif form == "volatile":
        prefix = f"st.volatile{_dot(space)}"
    else:
        prefix = f"st{'.weak' if weak else ''}{_dot(space)}{_dot(cop)}"
    level = _ptx_level_suffix(has_cache, l1_evict, l2_evict, "")
    instr = f"{prefix}{level}{_dot(vec)}.{ptx_type}"
    name_parts = ["tvm_builtin_ptx_st", form if form != "weak" else ("weak" if weak else "plain")]
    if sem:
        name_parts.append(_safe_attr(sem))
    if scope:
        name_parts.append(_safe_attr(scope))
    name_parts.extend(
        [
            _safe_attr(space),
            _safe_attr(cop) if cop else "",
            _safe_attr(vec) if vec else "",
            ptx_type,
            "from_src" if from_src else "values",
        ]
    )
    if has_cache:
        name_parts.append("cache_hint")
    if raw_address:
        name_parts.append("raw_u32")
    name = "_".join(p for p in name_parts if p)
    values = f"{{{', '.join(f'%{i + 1}' for i in range(vec_len))}}}" if vec_len > 1 else "%1"
    value_constraints = "".join(f', "{constraint}"(value{i})' for i in range(vec_len))
    cache_slot = f", %{vec_len + 1}" if has_cache else ""
    cache_operand = ', "l"(cache_policy)' if has_cache else ""
    addr_decl, addr_operand = _ptx_addr_operand(space, "address", raw_address)
    address_type = "unsigned int" if raw_address else "void*"
    if from_src:
        load_regs = _ptx_st_load_src(num_bytes, vec_len, ptx_type, c_type)
        if vec_len > 1:
            in_constraints = ", ".join(f'"{constraint}"(r{i})' for i in range(vec_len))
            value_args = f", {in_constraints}"
        else:
            value_args = f', "{constraint}"(r0)'
        body = (
            f"{addr_decl}{load_regs}"
            f'    asm volatile("{instr} [%0], {values}{cache_slot};"\n'
            "                 :\n"
            f"                 : {addr_operand}{value_args}{cache_operand}\n"
            '                 : "memory");'
        )
        if use_cache_policy:
            sig = f"({address_type} address, void* src_ptr, unsigned long long cache_policy)"
        else:
            sig = f"({address_type} address, void* src_ptr)"
    else:
        body = (
            f"{addr_decl}"
            f'    asm volatile("{instr} [%0], {values}{cache_slot};"\n'
            "                 :\n"
            f"                 : {addr_operand}{value_constraints}{cache_operand}\n"
            '                 : "memory");'
        )
        if form == "mmio":
            sig = f"({address_type} address, {c_type} value0)"
        elif use_cache_policy:
            value_params = ", ".join(f"{c_type} value{i}" for i in range(vec_len))
            sig = f"({address_type} address, {value_params}, unsigned long long cache_policy)"
        else:
            value_params = ", ".join(f"{c_type} value{i}" for i in range(vec_len))
            sig = f"({address_type} address, {value_params})"
    return name, sig, body


def _register_ptx_st(op_name, form, n_attrs):
    def codegen(*args):
        from_src = _bool_attr(args[-1])
        st_attrs = args[-n_attrs:-1]
        forward = args[:-(n_attrs)]
        raw_address = form == "weak" and _is_raw_u32_address(forward[0])
        parts = _ptx_st_form_parts(form, st_attrs, from_src, raw_address)
        name, sig, body_str = parts
        source_code = f"\n__forceinline__ __device__ void {name}{sig} {{\n{body_str}\n}}\n"
        return cuda_func_call(name, *forward, source_code=source_code)

    codegen.__name__ = f"codegen_{op_name}"
    register_codegen(op_name)(codegen)


_register_ptx_st("ptx_st", "weak", 9)


device_intrinsic(
    "cuda_uint_as_float",
    helper_name="tvm_builtin_uint_as_float",
    c_signature="(unsigned int bits)",
    return_type="float",
    body="    return __uint_as_float(bits);",
)
device_intrinsic(
    "cuda_float_as_uint",
    helper_name="tvm_builtin_float_as_uint",
    c_signature="(float x)",
    return_type="unsigned int",
    body="    return __float_as_uint(x);",
)
