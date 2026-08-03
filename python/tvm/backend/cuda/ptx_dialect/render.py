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

from .table import (
    DTYPE_C,
    DTYPE_SUFFIX,
    InstructionEntry,
    dtype_combos,
    mods,
    operand_space,
    typed_operands,
)


def render_variant(entry: InstructionEntry, tokens, predicated=False, dtypes=None):
    """Render one variant: ``(opcode, helper_name, helper_source)``.

    Every helper is ``void`` and its C parameter list is the PTX operand list
    in order, so the generated call reads like the PTX text it wraps.
    Destinations (``role="dst"``) are taken by reference; the caller passes a
    writable lvalue.

    ``dtypes`` picks one TVM dtype per typed operand (see ``dtype_combos``);
    None means the canonical choice, which is what every non-bit-typed operand
    has anyway. A non-canonical choice appends the dtypes to the helper name, so
    the names a table without bit-typed operands would produce are untouched.

    ``predicated`` is a framework-level axis (never in the table): the helper
    gains a trailing ``uint32_t __pred`` operand, and the instruction is
    guarded with ``.reg .pred p; setp.ne.b32 p, %N, 0; @p ...``. Only valid
    for instructions without a destination — see ``InstructionEntry.has_dst``.
    """
    mod_map = mods(entry, tokens)
    written = [tok for tok in tokens if tok]
    opcode = ".".join([entry.ptx_name, *written])
    canonical = dtype_combos(entry, tokens)[0]
    if dtypes is None:
        dtypes = canonical
    # The opcode alone no longer determines the signature once an operand may
    # take several dtypes. A non-canonical choice names *every* typed operand,
    # positionally: naming only the ones that changed would collide whenever two
    # operands swap which of them is non-canonical (atom's d and b, for one).
    if tuple(dtypes) != tuple(canonical):
        written = written + [DTYPE_SUFFIX[d] for d in dtypes]
    # The helper name keys off the table name rather than the mnemonic. For every
    # single-shape family the two agree (st.bulk normalizes to st_bulk anyway);
    # it matters only where several entries share a mnemonic because PTX puts
    # their difference in the operand list, as `mov`'s pack/unpack shapes do.
    helper = "tvm_builtin_ptxd_" + "_".join([entry.name, *written]).replace("::", "__").replace(
        ".", "_"
    )
    if predicated:
        assert not entry.has_dst, "@p is only supported on instructions without a destination"
        helper += "_pred"

    params, inputs, outputs, ptx_operands = [], [], [], []
    pre, post = [], []  # carrier declarations / boundary conversions
    dtype_of = dict(zip(typed_operands(entry), dtypes, strict=True))
    idx = 0
    for slot in entry.operands:
        pname = f"__{slot.name}"
        if slot.role == "imm":
            # ISA-fixed immediate: part of the instruction text, not an operand
            # the caller supplies.
            ptx_operands.append(slot.literal)
            continue
        regs = []
        for lane in range(slot.lanes):
            # One operand, `lanes` registers: PTX writes the group in the
            # operand list, so a lane is a C parameter but not an operand.
            lname = pname if slot.lanes == 1 else f"{pname}{lane}"
            if slot.role == "dst":
                c_ty, constraint, carrier, _, out_fmt = DTYPE_C[dtype_of[slot]]
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
                    post.append(f"{lname} = {out_fmt.format(reg)};")
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
                c_ty, constraint, carrier, in_fmt, _ = DTYPE_C[dtype_of[slot]]
                params.append(f"{c_ty} {lname}")
                inputs.append(f'"{constraint}"({in_fmt.format(lname)})')
            regs.append(f"[%{idx}]" if slot.role == "addr" else f"%{idx}")
            idx += 1
        ptx_operands.append(regs[0] if slot.lanes == 1 else "{" + ", ".join(regs) + "}")

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
    clobber = ' : "memory"' if any(s.role == "addr" for s in entry.operands) else ""
    asm_line = f'asm{volatile}("{asm_text}" : {", ".join(outputs)} : {", ".join(inputs)}{clobber});'
    body = "\n".join(f"  {line}" for line in [*pre, asm_line, *post])
    source = f"__forceinline__ __device__ void {helper}({', '.join(params)}) {{\n{body}\n}}\n"
    return opcode, helper, source
