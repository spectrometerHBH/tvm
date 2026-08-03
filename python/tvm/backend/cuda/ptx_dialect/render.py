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
"""Pure CUDA-helper rendering for the ptxd dialect.

tvm-free (imports only :mod:`.table`), so it is shared by the codegen engine
and by :mod:`.gen_helpers`, which dumps the generated helpers for humans to
inspect without compiling a kernel.
"""

from typing import NamedTuple

from .table import (
    InstructionEntry,
    canonical_dtypes,
    imm_slots,
    lanes_of,
    mods,
    operand_space,
)


class CBinding(NamedTuple):
    """How one TVM dtype crosses the C / inline-asm boundary.

    ``carrier`` is the C type actually bound to the asm register. When it equals
    ``c_type`` the value binds its register directly; otherwise it rides a
    differently-typed register and converts at each boundary, because 8-bit
    values have no constraint letter of their own and __half/__nv_bfloat16
    cannot bind one at all (nvcc: "more than one conversion function applies").

    Every conversion here is a bit pun, never an arithmetic cast. That is the
    whole point of the dtype axis: handing a float to a uint32_t parameter is a
    *numeric* conversion and emits `cvt.rzi.u32.f32`, silently changing the
    value. ``suffix`` is the token that names this dtype in a helper name.
    """

    c_type: str
    constraint: str
    carrier: str
    to_carrier: str  # format string, applied on the way into the asm
    from_carrier: str  # format string, applied on the way out
    suffix: str


# Named fields, not a positional tuple: this is the same table shape whose
# modifier-token twin needed `tokens_for`, and a new column would otherwise
# shift every unpacking site.
C_BINDING = {
    "uint8": CBinding("uint8_t", "h", "uint16_t", "(uint16_t){}", "(uint8_t){}", "u8"),
    "int8": CBinding("int8_t", "h", "int16_t", "(int16_t){}", "(int8_t){}", "s8"),
    "uint16": CBinding("uint16_t", "h", "uint16_t", "{}", "{}", "u16"),
    "int16": CBinding("int16_t", "h", "int16_t", "{}", "{}", "s16"),
    "float16": CBinding(
        "__half", "h", "uint16_t", "__half_as_ushort({})", "__ushort_as_half({})", "f16"
    ),
    "bfloat16": CBinding(
        "__nv_bfloat16",
        "h",
        "uint16_t",
        "__bfloat16_as_ushort({})",
        "__ushort_as_bfloat16({})",
        "bf16",
    ),
    "uint32": CBinding("uint32_t", "r", "uint32_t", "{}", "{}", "u32"),
    "int32": CBinding("int32_t", "r", "int32_t", "{}", "{}", "s32"),
    "float32": CBinding("float", "f", "float", "{}", "{}", "f32"),
    "uint64": CBinding("uint64_t", "l", "uint64_t", "{}", "{}", "u64"),
    "int64": CBinding("int64_t", "l", "int64_t", "{}", "{}", "s64"),
    "float64": CBinding("double", "d", "double", "{}", "{}", "f64"),
    # 128-bit: the "q" constraint needs __int128 support in the host compiler.
    "uint128": CBinding("__uint128_t", "q", "__uint128_t", "{}", "{}", "u128"),
    "int128": CBinding("__int128_t", "q", "__int128_t", "{}", "{}", "s128"),
}


def render_variant(entry: InstructionEntry, tokens, predicated=False, dtypes=None, imms=None):
    """Render one variant: ``(opcode, helper_name, helper_source)``.

    Every helper is ``void`` and its C parameter list is the PTX operand list
    in order, so the generated call reads like the PTX text it wraps.
    Destinations (``role="dst"``) are taken by reference; the caller passes a
    writable lvalue.

    ``dtypes`` picks one TVM dtype per typed operand (see ``table.dtype_combos``);
    None means the canonical choice, which is what every non-bit-typed operand
    has anyway. A non-canonical choice appends the dtypes to the helper name, so
    the names a table without bit-typed operands would produce are untouched.

    ``imms`` picks one value per caller-chosen immediate (see
    ``table.imm_slots``); the value is part of the instruction's identity, so
    it lands in the asm text and in the helper name, and the helper has no C
    parameter for it.

    ``predicated`` is a framework-level axis (never in the table): the helper
    gains a trailing ``uint32_t __pred`` operand, and the instruction is
    guarded with ``.reg .pred p; setp.ne.b32 p, %N, 0; @p ...``. Only valid
    for instructions without a destination — see ``InstructionEntry.has_dst``.
    """
    mod_map = mods(entry, tokens)
    written = [tok for tok in tokens if tok]
    opcode = ".".join([entry.ptx_name, *written])
    canonical = canonical_dtypes(entry, tokens)
    if dtypes is None:
        dtypes = canonical
    # A helper name is the instruction's ISA identity plus, only when it is no
    # longer enough, a signature discriminator. The opcode alone stopped being
    # enough once an operand could take several dtypes; a non-canonical choice
    # then names *every* typed operand, positionally, because naming only the
    # ones that changed collides whenever two operands swap which of them is
    # non-canonical (atom's d and b do exactly that).
    imm_of = dict(zip(imm_slots(entry), imms or (), strict=True))
    isa_name = [entry.name, *written, *(imms or ())]  # table name, not mnemonic: see below
    discriminator = (
        [] if tuple(dtypes) == tuple(canonical) else [C_BINDING[d].suffix for d in dtypes]
    )
    # The name keys off the table name rather than the mnemonic. For every
    # single-shape family the two agree (st.bulk normalizes to st_bulk anyway);
    # it matters only where several entries share a mnemonic because PTX puts
    # their difference in the operand list, as `mov`'s pack/unpack shapes do.
    helper = "tvm_builtin_ptxd_" + "_".join([*isa_name, *discriminator]).replace(
        "::", "__"
    ).replace(".", "_")
    if predicated:
        assert not entry.has_dst, "@p is only supported on instructions without a destination"
        helper += "_pred"

    params, inputs, outputs, ptx_operands = [], [], [], []
    pre, post = [], []  # carrier declarations / boundary conversions
    dtype_of = dict(zip(entry.typed_operands, dtypes, strict=True))
    idx = 0
    for slot in entry.operands:
        pname = f"__{slot.name}"
        if slot.role == "imm":
            # An immediate lives in the instruction text. Either the ISA fixed
            # its value (`literal`) or the caller chose it from a closed set
            # (`choices`); neither is a C parameter.
            ptx_operands.append(slot.literal if slot.choices is None else imm_of[slot])
            continue
        regs = []
        n_lanes = lanes_of(slot, mod_map)
        # A group operand is one whose declared shape is a vector -- statically
        # (`lanes > 1`) or by modifier (callable). ptxas wants the braces even
        # when such a vector has one element ("Vector of size 1 is expected"),
        # so vector-ness follows the declaration, not the resolved count.
        is_group = callable(slot.lanes) or slot.lanes > 1
        for lane in range(n_lanes):
            # One operand, `lanes` registers: PTX writes the group in the
            # operand list, so a lane is a C parameter but not an operand.
            lname = f"{pname}{lane}" if is_group else pname
            if slot.role == "dst":
                cb = C_BINDING[dtype_of[slot]]
                c_ty, constraint, carrier = cb.c_type, cb.constraint, cb.carrier
                params.append(f"{c_ty}& {lname}")
                if carrier == c_ty:
                    outputs.append(f'"={constraint}"({lname})')
                else:
                    # The value cannot bind this register class directly (8-bit
                    # has no constraint letter; __half binds none at all), so the
                    # asm writes a carrier local that is bit-punned back out.
                    reg = f"{lname}_reg"
                    pre.append(f"{carrier} {reg};")
                    outputs.append(f'"={constraint}"({reg})')
                    post.append(f"{lname} = {cb.from_carrier.format(reg)};")
            elif slot.role == "addr":
                if operand_space(slot, mod_map).startswith("shared"):
                    params.append(f"uint32_t {lname}")
                    inputs.append(f'"r"({lname})')
                else:
                    params.append(f"const void* {lname}")
                    inputs.append(f'"l"({lname})')
            elif slot.role == "ptr":
                params.append(f"const void* {lname}")
                inputs.append(f'"l"({lname})')
            else:  # value
                cb = C_BINDING[dtype_of[slot]]
                params.append(f"{cb.c_type} {lname}")
                inputs.append(f'"{cb.constraint}"({cb.to_carrier.format(lname)})')
            regs.append(f"[%{idx}]" if slot.role == "addr" else f"%{idx}")
            idx += 1
        ptx_operands.append("{" + ", ".join(regs) + "}" if is_group else regs[0])

    instr = f"{opcode} {', '.join(ptx_operands)};" if ptx_operands else f"{opcode};"
    if predicated:
        params.append("uint32_t __pred")
        inputs.append('"r"(__pred)')
        asm_text = f"{{ .reg .pred p; setp.ne.b32 p, %{idx}, 0; @p {instr} }}"
    else:
        asm_text = instr
    # Two independent C-level properties, deliberately not conflated:
    #
    #   asm volatile  - may nvcc reorder or drop the emitted asm?
    #   "memory"      - does the instruction touch memory?
    #
    # `volatile` is stated per entry so each instruction keeps the barrier its
    # legacy helper had. The clobber is derived instead: an instruction with no
    # memory operand cannot clobber memory, and claiming it does is a needless
    # optimization barrier around register-only instructions like ex2.
    volatile = " volatile" if entry.asm_volatile else ""
    # Two ways to clobber memory: name an address, or order everyone else's
    # accesses (a fence names nothing but must not let loads/stores move past).
    touches_memory = any(s.role == "addr" for s in entry.operands) or entry.orders_memory
    clobber = ' : "memory"' if touches_memory else ""
    asm_line = f'asm{volatile}("{asm_text}" : {", ".join(outputs)} : {", ".join(inputs)}{clobber});'
    body = "\n".join(f"  {line}" for line in [*pre, asm_line, *post])
    source = f"__forceinline__ __device__ void {helper}({', '.join(params)}) {{\n{body}\n}}\n"
    return opcode, helper, source
