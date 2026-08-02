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
"""Instruction table for the ``T.ptxd`` table-driven PTX dialect prototype.

Pure data + pure functions: this module deliberately imports nothing from
``tvm`` so the thin generators (``gen_stubs``, ``gen_coverage``,
``gen_helpers``) can load it standalone.

The converged :class:`InstructionEntry` design:

- ``name`` is a single identifier-safe token. Multi-token PTX mnemonics
  (``cvta.to.shared``, ``prefetch.global.L2``) are expressed as a family
  plus single-choice modifier slots — PTX itself treats those dots as
  modifiers, and it keeps the namespace machinery trivial.
- ``slots`` declares each modifier position's domain (name, tokens,
  optional). This drives attribute-chain resolution, stub generation, and
  variant enumeration.
- ``check`` is the single cross-slot constraint mechanism: a plain, pure
  Python function ``mod_map -> error-string | None`` (mod_map maps every
  slot name to its token, ``""`` = omitted). It runs at trace time to
  reject illegal combinations with a readable message, and as a filter in
  :func:`variants` — so exhaustive nvcc gating still works. Give it a
  one-line docstring; the generators surface it as documentation.
- One entry = one *syntax shape*: fixed operand list and return convention.
  Variants that change operand/result structure (e.g. vector DPS loads)
  become separate entries, each declaring only the slots it uses.
- Predication (``@p``) is framework-level: every void instruction accepts
  ``pred=`` at the call site; entries never mention it.
"""

import itertools
import keyword
from collections.abc import Callable
from dataclasses import dataclass

# Mirrors tvm::tirx::CallEffectKind (include/tvm/tirx/op_attr_types.h).
EFFECT_PURE = 1
EFFECT_OPAQUE = 3
EFFECT_NAMES = {EFFECT_PURE: "pure", EFFECT_OPAQUE: "opaque"}

# PTX operand type token -> (TVM dtype, C type, inline-asm constraint).
PTX_TYPES = {
    "b32": ("uint32", "uint32_t", "r"),
    "u32": ("uint32", "uint32_t", "r"),
    "s32": ("int32", "int32_t", "r"),
    "f32": ("float32", "float", "f"),
    "b64": ("uint64", "uint64_t", "l"),
    "u64": ("uint64", "uint64_t", "l"),
}


def escape_token(token: str) -> str:
    """PTX token -> Python attribute name (``::`` -> ``__``, keyword -> trailing ``_``)."""
    token = token.replace("::", "__")
    if keyword.iskeyword(token):
        token += "_"
    return token


def unescape_token(token: str) -> str:
    """Python attribute name -> PTX token. Inverse of :func:`escape_token`."""
    token = token.replace("__", "::")
    if token.endswith("_"):
        token = token[:-1]
    return token


@dataclass(frozen=True)
class ModifierSlot:
    """One modifier position of an instruction family, in asm render order."""

    name: str
    choices: tuple[str, ...]
    optional: bool = False  # optional => omitted token is simply not rendered


@dataclass(frozen=True)
class OperandSlot:
    """One operand of an instruction family, in PTX operand order.

    role:
      - ``"addr"``  memory operand, rendered ``[%k]``; state space comes from
        ``space`` (fixed per operand, for instructions whose operands live in
        different spaces like cp.async.bulk) or else the entry's ``space``
        modifier slot. Shared-space addresses are auto-coerced (generic
        pointer -> cvta) or accepted as raw ``uint32``.
      - ``"ptr"``   raw pointer value, rendered ``%k`` (e.g. ``cvta`` input).
      - ``"value"`` register operand typed by ``dtype`` (fixed per operand)
        or else the entry's ``type`` modifier slot.
    """

    name: str
    role: str
    space: str | None = None
    dtype: str | None = None


CheckFn = Callable[[dict], str | None]


@dataclass(frozen=True)
class InstructionEntry:
    """One PTX instruction family (one syntax shape)."""

    name: str  # single-token family name, e.g. "cvta"
    operands: tuple[OperandSlot, ...]
    effect: int  # CallEffectKind int (TCallEffectKind attr)
    returns: str | None  # None (void) or the slot naming the result dtype
    slots: tuple[ModifierSlot, ...] = ()
    check: CheckFn | None = None  # cross-slot validation, mod_map -> error | None

    @property
    def op_name(self) -> str:
        return f"tirx.ptxd.{self.name}"


def mods(entry: InstructionEntry, tokens) -> dict:
    """The canonical modifier map: every slot name -> token, ``""`` = omitted."""
    return {slot.name: tok or "" for slot, tok in zip(entry.slots, tokens)}


def variants(entry: InstructionEntry):
    """Yield every legal modifier combination: the slot product filtered by ``check``."""
    axes = [(*slot.choices, "") if slot.optional else slot.choices for slot in entry.slots]
    for combo in itertools.product(*axes):
        if entry.check is not None and entry.check(mods(entry, combo)):
            continue
        yield combo


def _check_ld(m):
    """acquire/relaxed require a scope; weak (omitted) and volatile forms take none."""
    if m["sem"] in ("acquire", "relaxed") and not m["scope"]:
        return f"ld.{m['sem']} requires a scope (cta/gpu/sys)"
    if m["sem"] in ("", "volatile") and m["scope"]:
        return f"{'ld.volatile' if m['sem'] else 'weak ld'} takes no scope"
    return None


def _check_st(m):
    """release/relaxed require a scope; weak/volatile take none; shared::cta caps scope at cta."""
    if m["sem"] in ("release", "relaxed") and not m["scope"]:
        return f"st.{m['sem']} requires a scope (cta/gpu/sys)"
    if m["sem"] in ("", "weak", "volatile") and m["scope"]:
        return "unscoped st forms (weak/volatile) take no scope"
    if m["space"] == "shared::cta" and m["scope"] not in ("", "cta"):
        return "st.shared::cta is CTA-local; scope must be cta or omitted"
    return None


_LDST_TYPES = ("b32", "b64", "u32", "f32")
_SCOPES = ("cta", "gpu", "sys")

_ENTRIES = [
    InstructionEntry(
        name="prefetch",
        slots=(
            ModifierSlot("space", ("global",)),
            ModifierSlot("level", ("L2",)),
        ),
        operands=(OperandSlot("addr", role="addr"),),
        effect=EFFECT_OPAQUE,
        returns=None,
    ),
    InstructionEntry(
        name="ld",
        slots=(
            ModifierSlot("sem", ("acquire", "relaxed", "volatile"), optional=True),
            ModifierSlot("scope", _SCOPES, optional=True),
            ModifierSlot("space", ("global",)),
            ModifierSlot("type", _LDST_TYPES),
        ),
        check=_check_ld,
        operands=(OperandSlot("addr", role="addr"),),
        effect=EFFECT_OPAQUE,
        returns="type",
    ),
    InstructionEntry(
        name="st",
        slots=(
            ModifierSlot("sem", ("weak", "release", "relaxed", "volatile"), optional=True),
            ModifierSlot("scope", _SCOPES, optional=True),
            ModifierSlot("space", ("global", "shared::cta")),
            ModifierSlot("type", _LDST_TYPES),
        ),
        check=_check_st,
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("value", role="value"),
        ),
        effect=EFFECT_OPAQUE,
        returns=None,
    ),
    InstructionEntry(
        name="red",
        slots=(
            ModifierSlot("sem", ("relaxed", "release")),
            ModifierSlot("scope", ("gpu", "sys")),
            ModifierSlot("space", ("global",)),
            ModifierSlot("op", ("add",)),
            ModifierSlot("type", ("u32", "s32", "f32")),
        ),
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("value", role="value"),
        ),
        effect=EFFECT_OPAQUE,
        returns=None,
    ),
    InstructionEntry(
        name="cvta",
        slots=(
            ModifierSlot("dir", ("to",)),
            ModifierSlot("space", ("shared",)),
            ModifierSlot("type", ("u64",)),
        ),
        operands=(OperandSlot("ptr", role="ptr"),),
        effect=EFFECT_PURE,
        returns="type",
    ),
    # Mixed-space operands: dst/mbar are shared::cta (u32 carriers), src is
    # global (pointer carrier), size is a plain u32 register — each operand
    # declares its own space/dtype instead of reading the entry-level slots.
    InstructionEntry(
        name="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("dst_space", ("shared::cta",)),
            ModifierSlot("src_space", ("global",)),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
        ),
        operands=(
            OperandSlot("dst", role="addr", space="shared::cta"),
            OperandSlot("src", role="addr", space="global"),
            OperandSlot("size", role="value", dtype="u32"),
            OperandSlot("mbar", role="addr", space="shared::cta"),
        ),
        effect=EFFECT_OPAQUE,
        returns=None,
    ),
]

TABLE: dict[str, InstructionEntry] = {e.name: e for e in _ENTRIES}
