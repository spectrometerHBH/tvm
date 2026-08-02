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

from .table import PTX_TYPES, InstructionEntry, mods, operand_type


def render_variant(entry: InstructionEntry, tokens, predicated=False):
    """Render one variant: ``(opcode, helper_name, helper_source)``.

    Every helper is ``void`` and its C parameter list is the PTX operand list
    in order, so the generated call reads like the PTX text it wraps.
    Destinations (``role="dst"``) are taken by reference; the caller passes a
    writable lvalue.

    ``predicated`` is a framework-level axis (never in the table): the helper
    gains a trailing ``uint32_t __pred`` operand, and the instruction is
    guarded with ``.reg .pred p; setp.ne.b32 p, %N, 0; @p ...``. Only valid
    for instructions without a destination — see ``InstructionEntry.has_dst``.
    """
    mod_map = mods(entry, tokens)
    opcode = ".".join([entry.ptx_name] + [tok for tok in tokens if tok])
    # The helper name keys off the table name rather than the mnemonic. For every
    # single-shape family the two agree (st.bulk normalizes to st_bulk anyway);
    # it matters only where several entries share a mnemonic because PTX puts
    # their difference in the operand list, as `mov`'s pack/unpack shapes do.
    helper = "tvm_builtin_ptxd_" + "_".join([entry.name] + [tok for tok in tokens if tok]).replace(
        "::", "__"
    ).replace(".", "_")
    if predicated:
        assert not entry.has_dst, "@p is only supported on instructions without a destination"
        helper += "_pred"

    params, inputs, outputs, ptx_operands = [], [], [], []
    pre, post = [], []  # carrier declarations / narrowing write-backs
    idx = 0
    for slot in entry.operands:
        pname = f"__{slot.name}"
        if slot.role == "dst" and slot.lanes > 1:
            # `{%k, %k+1, ...}` -- the group is one PTX operand but N C params.
            _, c_ty, constraint, _ = PTX_TYPES[operand_type(slot, mod_map)]
            lane_regs = []
            for lane in range(slot.lanes):
                lname = f"{pname}{lane}"
                params.append(f"{c_ty}& {lname}")
                outputs.append(f'"={constraint}"({lname})')
                lane_regs.append(f"%{idx}")
                idx += 1
            ptx_operands.append("{" + ", ".join(lane_regs) + "}")
            continue
        if slot.role == "value" and slot.lanes > 1:
            _, _, constraint, value_carrier = PTX_TYPES[operand_type(slot, mod_map)]
            lane_regs = []
            for lane in range(slot.lanes):
                lname = f"{pname}{lane}"
                params.append(f"{value_carrier} {lname}")
                inputs.append(f'"{constraint}"({lname})')
                lane_regs.append(f"%{idx}")
                idx += 1
            ptx_operands.append("{" + ", ".join(lane_regs) + "}")
            continue
        if slot.role == "dst":
            _, c_ty, constraint, carrier = PTX_TYPES[operand_type(slot, mod_map)]
            params.append(f"{c_ty}& {pname}")
            if carrier == c_ty:
                outputs.append(f'"={constraint}"({pname})')
            else:
                # 8-bit destinations have no asm constraint of their own and
                # ride a 16-bit register (ISA: "A destination register wider
                # than the specified type may be used"), so the asm writes a
                # carrier local that is then narrowed into the reference.
                reg = f"{pname}_reg"
                pre.append(f"{carrier} {reg};")
                outputs.append(f'"={constraint}"({reg})')
                post.append(f"{pname} = ({c_ty}){reg};")
            ptx_operands.append(f"%{idx}")
        elif slot.role == "addr":
            space = slot.space or mod_map.get("space", "")
            if space.startswith("shared"):
                params.append(f"uint32_t {pname}")
                inputs.append(f'"r"({pname})')
            else:
                params.append(f"const void* {pname}")
                inputs.append(f'"l"({pname})')
            ptx_operands.append(f"[%{idx}]")
        elif slot.role == "imm":
            # ISA-fixed immediate: part of the instruction text, not a operand
            # the caller supplies.
            ptx_operands.append(slot.literal)
            continue
        elif slot.role == "ptr":
            params.append(f"const void* {pname}")
            inputs.append(f'"l"({pname})')
            ptx_operands.append(f"%{idx}")
        else:  # value
            _, _, constraint, value_carrier = PTX_TYPES[operand_type(slot, mod_map)]
            params.append(f"{value_carrier} {pname}")
            inputs.append(f'"{constraint}"({pname})')
            ptx_operands.append(f"%{idx}")
        idx += 1

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
