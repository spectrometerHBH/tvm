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

from .table import EFFECT_PURE, PTX_TYPES, InstructionEntry, mods


def render_variant(entry: InstructionEntry, tokens, predicated=False):
    """Render one variant: ``(opcode, helper_name, helper_source)``.

    ``predicated`` is a framework-level axis (never in the table): the helper
    gains a trailing ``uint32_t __pred`` operand, and the instruction is
    guarded with ``.reg .pred p; setp.ne.b32 p, %N, 0; @p ...``. Only valid
    for void instructions (a false predicate leaves destinations unwritten).
    """
    mod_map = mods(entry, tokens)
    opcode = ".".join([entry.name] + [tok for tok in tokens if tok])
    helper = "tvm_builtin_ptxd_" + opcode.replace("::", "__").replace(".", "_")
    if predicated:
        assert entry.returns is None, "@p is only supported on void instructions"
        helper += "_pred"

    params, inputs, outputs, ptx_operands = [], [], [], []
    idx = 0
    c_ret = carrier = "void"
    if entry.returns is not None:
        _, c_ret, ret_constraint, carrier = PTX_TYPES[mod_map[entry.returns]]
        outputs.append(f'"={ret_constraint}"(__ret)')
        ptx_operands.append(f"%{idx}")
        idx += 1
    for slot in entry.operands:
        pname = f"__{slot.name}"
        if slot.role == "addr":
            space = slot.space or mod_map.get("space", "")
            if space.startswith("shared"):
                params.append(f"uint32_t {pname}")
                inputs.append(f'"r"({pname})')
            else:
                params.append(f"const void* {pname}")
                inputs.append(f'"l"({pname})')
            ptx_operands.append(f"[%{idx}]")
        elif slot.role == "ptr":
            params.append(f"const void* {pname}")
            inputs.append(f'"l"({pname})')
            ptx_operands.append(f"%{idx}")
        else:  # value
            _, _, constraint, value_carrier = PTX_TYPES[slot.dtype or mod_map["type"]]
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
    # Three independent properties, deliberately not conflated:
    #
    #   entry.effect  - may the *IR* reorder, CSE or drop this call?
    #   asm volatile  - may *nvcc* reorder or drop the emitted asm?
    #   "memory"      - does the instruction touch memory?
    #
    # `volatile` defaults to the IR's answer but may be stated per entry, because
    # the two genuinely disagree in practice: ex2/rcp are pure yet carry the
    # barrier, fns is pure and does not.
    #
    # The memory clobber is separate again: an instruction with no memory
    # operand cannot clobber memory, and claiming it does is a needless
    # optimization barrier around pure-register instructions like ex2.
    touches_memory = any(slot.role == "addr" for slot in entry.operands)
    is_volatile = (
        entry.asm_volatile if entry.asm_volatile is not None else entry.effect != EFFECT_PURE
    )
    volatile = " volatile" if is_volatile else ""
    clobber = ' : "memory"' if entry.effect != EFFECT_PURE and touches_memory else ""
    asm_line = f'asm{volatile}("{asm_text}" : {", ".join(outputs)} : {", ".join(inputs)}{clobber});'
    if entry.returns is None:
        body = f"  {asm_line}"
    else:
        # __ret lives in the asm carrier type; narrow on return when they
        # differ (8-bit loads ride in 16-bit "h" registers).
        ret_stmt = "return __ret;" if carrier == c_ret else f"return ({c_ret})__ret;"
        body = f"  {carrier} __ret;\n  {asm_line}\n  {ret_stmt}"
    source = f"__forceinline__ __device__ {c_ret} {helper}({', '.join(params)}) {{\n{body}\n}}\n"
    return opcode, helper, source
