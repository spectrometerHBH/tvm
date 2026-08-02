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

import functools
import itertools
import keyword
from collections.abc import Callable
from dataclasses import dataclass

# Mirrors tvm::tirx::CallEffectKind (include/tvm/tirx/op_attr_types.h).
EFFECT_PURE = 1
EFFECT_OPAQUE = 3
EFFECT_NAMES = {EFFECT_PURE: "pure", EFFECT_OPAQUE: "opaque"}

# PTX operand type token -> (TVM dtype, C type, inline-asm constraint, C carrier).
# The carrier is the C type actually bound to the asm register: 8-bit values
# have no asm constraint of their own, so they ride in 16-bit "h" registers
# (PTX ld/st sign/zero-extend into wider registers; ISA "Notes").
PTX_TYPES = {
    "b8": ("uint8", "uint8_t", "h", "uint16_t"),
    "u8": ("uint8", "uint8_t", "h", "uint16_t"),
    "s8": ("int8", "int8_t", "h", "int16_t"),
    "b16": ("uint16", "uint16_t", "h", "uint16_t"),
    "u16": ("uint16", "uint16_t", "h", "uint16_t"),
    "s16": ("int16", "int16_t", "h", "int16_t"),
    "b32": ("uint32", "uint32_t", "r", "uint32_t"),
    "u32": ("uint32", "uint32_t", "r", "uint32_t"),
    "s32": ("int32", "int32_t", "r", "int32_t"),
    "b64": ("uint64", "uint64_t", "l", "uint64_t"),
    "u64": ("uint64", "uint64_t", "l", "uint64_t"),
    "s64": ("int64", "int64_t", "l", "int64_t"),
    "f32": ("float32", "float", "f", "float"),
    "f64": ("float64", "double", "d", "double"),
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


@functools.cache
def variants(entry: InstructionEntry) -> tuple:
    """Every legal modifier combination: the slot product filtered by ``check``.

    Cached: for wide entries like ld the raw product is in the millions.
    """
    axes = [(*slot.choices, "") if slot.optional else slot.choices for slot in entry.slots]
    return tuple(
        combo
        for combo in itertools.product(*axes)
        if entry.check is None or not entry.check(mods(entry, combo))
    )


def _check_ld(m):
    """Scalar ld grammar per PTX ISA 9.7.9.8 (ld) and 9.7.9.9 (ld.global.nc)."""
    sem, scope, ss = m["sem"], m["scope"], m["ss"]
    mmio, cop, nc = m["mmio"], m["cop"], m["nc"]
    l1ev, prefetch = m["l1ev"], m["prefetch"]
    # "ld.relaxed.scope / ld.acquire.scope" — scope is mandatory there and
    # invalid on the weak/volatile forms (syntax lines).
    if sem in ("relaxed", "acquire") and not scope:
        return f"ld.{sem} requires a scope (cta/cluster/gpu/sys)"
    if sem in ("", "weak", "volatile") and scope:
        return "only ld.relaxed/ld.acquire take a scope"
    if mmio:
        # "ld.mmio.sem.sys{.global}": "Only .sys thread scope is valid";
        # global or generic addressing only. The ISA also allows .acquire
        # (PTX ISA 9.3+), but the current toolchain assembles 9.2 and ptxas
        # rejects it — widen when the toolchain catches up.
        if sem != "relaxed":
            return "ld.mmio requires .relaxed"
        if scope != "sys":
            return "only the sys scope is valid for ld.mmio"
        if ss not in ("", "global"):
            return "ld.mmio may only be used with .global or generic addressing"
        if cop or nc or l1ev or prefetch:
            return "ld.mmio takes no cache qualifiers"
    if sem in ("relaxed", "acquire") and not mmio:
        # "May be used with .global, .shared spaces, or generic addressing.
        # Cache operations are not allowed."
        if ss == "local":
            return f"ld.{sem} is not valid on .local"
        if cop:
            return f"cache operations are not allowed with ld.{sem}"
        if nc:
            return "ld.global.nc has no memory-synchronization forms"
    if sem == "volatile" and (cop or l1ev or nc):
        # "ld.volatile{.ss}{.level::prefetch_size}" — prefetch is its only
        # cache qualifier; allowed spaces global/shared/local/generic.
        return "ld.volatile only takes the prefetch_size cache qualifier"
    if cop and l1ev:
        # cop and eviction_priority appear on separate syntax lines.
        return "cache operators and eviction priorities are mutually exclusive"
    if nc:
        # "ld.global{.cop}.nc": .global only, no sem/scope, cop in {ca,cg,cs}.
        if ss != "global":
            return "ld.global.nc requires the .global state space"
        if sem:
            return "ld.global.nc has no sem qualifier"
        if cop in ("lu", "cv"):
            return "ld.global.nc cache operators are limited to ca/cg/cs"
    if prefetch and ss not in ("", "global"):
        # "may only be used with .global state space and generic addressing"
        return "prefetch_size may only be used with .global or generic addressing"
    if l1ev and ss not in ("", "global"):
        # ptxas: "Modifier '.evict_*' cannot be applied to '<ss>' space" —
        # implicit in the ISA prose, enforced by the assembler.
        return "L1 eviction priorities apply only to .global or generic addressing"
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


_LDST_TYPES = ("b32", "b64", "u32", "u64", "s32", "f32")
_LD_TYPES = tuple(tok for tok in PTX_TYPES)  # all 14 scalar types (b128 excluded)
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
    # Complete scalar `ld` per PTX ISA 9.7.9.8 + the 9.7.9.9 ld.global.nc
    # forms. Deliberately excluded (each needs a mechanism this shape lacks):
    # - .vec/.b128 and the DPS destination forms (multi-register results)
    # - .level::cache_hint + the cache_policy operand (optional operand)
    # - .level2::eviction_priority (vector-only per the ISA)
    # - .unified (variable-attribute addressing)
    # - .param/.const spaces (require kernel-parameter / const addresses,
    #   which cannot flow through the helper-function ABI)
    InstructionEntry(
        name="ld",
        slots=(
            ModifierSlot("mmio", ("mmio",), optional=True),
            ModifierSlot("sem", ("weak", "acquire", "relaxed", "volatile"), optional=True),
            ModifierSlot("scope", ("cta", "cluster", "gpu", "sys"), optional=True),
            ModifierSlot(
                "ss",
                ("global", "shared", "shared::cta", "shared::cluster", "local"),
                optional=True,  # omitted = generic addressing
            ),
            ModifierSlot("cop", ("ca", "cg", "cs", "lu", "cv"), optional=True),
            ModifierSlot("nc", ("nc",), optional=True),
            ModifierSlot(
                "l1ev",
                (
                    "L1::evict_normal",
                    "L1::evict_unchanged",
                    "L1::evict_first",
                    "L1::evict_last",
                    "L1::no_allocate",
                ),
                optional=True,
            ),
            ModifierSlot("prefetch", ("L2::64B", "L2::128B", "L2::256B"), optional=True),
            ModifierSlot("type", _LD_TYPES),
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
