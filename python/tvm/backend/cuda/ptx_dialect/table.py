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
    # Whether the emitted inline asm carries `volatile`. This is a C-level
    # optimization barrier — it never changes *which* PTX instruction is
    # emitted, only whether nvcc may common up or drop identical calls. It is
    # deliberately independent of `effect`, which answers the same question
    # for the IR: ex2/rcp are pure (the IR shares a let-bound result) yet
    # carry the barrier, while fns is pure and does not. None derives it from
    # `effect`; set it explicitly to preserve an instruction's established
    # barrier when migrating.
    asm_volatile: bool | None = None

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


def _check_rcp(m):
    """rcp.approx is f32-only; .f64 is IEEE-rounded and takes no .ftz (PTX ISA 9.7.3.13)."""
    # Syntax lines: rcp.approx{.ftz}.f32 / rcp.rnd{.ftz}.f32 / rcp.rnd.f64
    if m["mode"] == "approx" and m["type"] != "f32":
        return "rcp.approx is only defined for .f32"
    if m["type"] == "f64":
        if m["mode"] == "approx":
            return "rcp.f64 requires an IEEE rounding mode (.rn/.rz/.rm/.rp)"
        if m["ftz"]:
            return "rcp.rnd.f64 takes no .ftz"
    return None


def _check_max(m):
    """`max.f64` is the bare form; .ftz/.NaN belong to the .f32 line (PTX ISA 9.7.3.12)."""
    if m["type"] == "f64" and (m["ftz"] or m["nan"]):
        return "max.f64 takes no .ftz or .NaN"
    return None


def _check_prefetch(m):
    """Each prefetch syntax line names exactly one target (PTX ISA 9.7.9.16)."""
    level, evict, tmap = m["level"], m["evict"], m["tensormap"]
    space = m["space"]
    if sum(bool(x) for x in (level, evict, tmap)) != 1:
        return "exactly one of .level, .level::eviction_priority or .tensormap"
    if tmap:
        # "prefetch{.tensormap_space}.tensormap", .tensormap_space = .const/.param
        if space not in ("", "const", "param"):
            return ".tensormap takes .const or .param (or generic addressing)"
    else:
        # "prefetch{.space}.level", .space = .global/.local
        if space in ("const", "param"):
            return ".const/.param are only valid with .tensormap"
        if evict and space != "global":
            return "eviction priority requires .global"
    return None


_RED_SEM = ("relaxed", "release")
_ATOM_SEM = ("relaxed", "acquire", "release", "acq_rel")
_ATOM_SCOPES = ("cta", "cluster", "gpu", "sys")
_ATOM_SPACES = ("global", "shared", "shared::cta", "shared::cluster")
_ATOM_OPS = ("and", "or", "xor", "add", "inc", "dec", "min", "max")
_ATOM_TYPES = ("b32", "b64", "u32", "u64", "s32", "s64", "f32", "f64")


def _check_atomic(m):
    """op x type pairings for red/atom (PTX ISA 9.7.14.5 / 9.7.14.6).

    The allowed type set per op is exactly what ptxas enforces; the ISA prose
    lists the union across ops rather than the per-op pairing. Half-precision
    types appear in ptxas' message but are excluded from this entry (they need
    .noftz and a half carrier type).
    """
    op, ty = m["op"], m["type"]
    allowed = {
        "and": ("b32", "b64"),
        "or": ("b32", "b64"),
        "xor": ("b32", "b64"),
        "inc": ("u32",),
        "dec": ("u32",),
        "min": ("u32", "s32", "u64", "s64"),
        "max": ("u32", "s32", "u64", "s64"),
        "add": ("u32", "s32", "u64", "f32", "f64"),
    }[op]
    if ty not in allowed:
        return f".{op} requires {' or '.join('.' + t for t in allowed)}"
    return None


_LDST_TYPES = ("b32", "b64", "u32", "u64", "s32", "f32")
_LD_TYPES = tuple(tok for tok in PTX_TYPES)  # all 14 scalar types (b128 excluded)
_SCOPES = ("cta", "gpu", "sys")

_ENTRIES = [
    # prefetch per PTX ISA 9.7.9.16, covering three of its four syntax lines:
    #   prefetch{.space}.level [a]
    #   prefetch.global.level::eviction_priority [a]
    #   prefetch{.tensormap_space}.tensormap [a]
    # `prefetchu.L1` is a different mnemonic and would be its own entry.
    InstructionEntry(
        name="prefetch",
        slots=(
            ModifierSlot("space", ("global", "local", "const", "param"), optional=True),
            ModifierSlot("level", ("L1", "L2"), optional=True),
            ModifierSlot("evict", ("L2::evict_last", "L2::evict_normal"), optional=True),
            ModifierSlot("tensormap", ("tensormap",), optional=True),
        ),
        check=_check_prefetch,
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
    # red / atom scalar `.op` forms per PTX ISA 9.7.14.6 and 9.7.14.5.
    # Deliberately excluded (each needs a mechanism this shape lacks):
    # - {.level::cache_hint} with its trailing cache_policy operand
    # - the .vec_16_bit/.vec_32_bit vector forms
    # - .f16/.bf16/.f16x2/.bf16x2 (add.noftz), which need half carrier types
    # - atom's .cas (3 operands) and .exch (own type set): other syntax shapes
    InstructionEntry(
        name="red",
        slots=(
            ModifierSlot("sem", _RED_SEM, optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot("space", _ATOM_SPACES, optional=True),
            ModifierSlot("op", _ATOM_OPS),
            ModifierSlot("type", _ATOM_TYPES),
        ),
        check=_check_atomic,
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("value", role="value"),
        ),
        effect=EFFECT_OPAQUE,
        returns=None,
    ),
    InstructionEntry(
        name="atom",
        slots=(
            ModifierSlot("sem", _ATOM_SEM, optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot("space", _ATOM_SPACES, optional=True),
            ModifierSlot("op", _ATOM_OPS),
            ModifierSlot("type", _ATOM_TYPES),
        ),
        check=_check_atomic,
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("value", role="value"),
        ),
        effect=EFFECT_OPAQUE,
        returns="type",
    ),
    # ex2 per PTX ISA 9.7.3.21 (`ex2.approx{.ftz}.f32`). The half-precision
    # forms of 9.7.4.10 (.f16/.f16x2/.bf16/.bf16x2) are deliberately excluded:
    # they need f16/bf16 carrier types, which PTX_TYPES does not model yet.
    InstructionEntry(
        name="ex2",
        slots=(
            ModifierSlot("mode", ("approx",)),
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("type", ("f32",)),
        ),
        operands=(OperandSlot("value", role="value"),),
        effect=EFFECT_PURE,
        asm_volatile=True,  # legacy barrier: preserved so migration is byte-identical
        returns="type",
    ),
    # rcp per PTX ISA 9.7.3.13. `rcp.approx.ftz.f64` (a separate syntax line in
    # the ISA, with its own sm floor) is excluded until it is needed.
    InstructionEntry(
        name="rcp",
        slots=(
            ModifierSlot("mode", ("approx", "rn", "rz", "rm", "rp")),
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("type", ("f32", "f64")),
        ),
        check=_check_rcp,
        operands=(OperandSlot("value", role="value"),),
        effect=EFFECT_PURE,
        asm_volatile=True,  # legacy barrier: preserved so migration is byte-identical
        returns="type",
    ),
    # fns per PTX ISA 9.7.1.18: `fns.b32 d, mask, base, offset;` — one form.
    # The operands carry three different types (mask .b32, base .b32/.u32/.s32,
    # offset .s32), so each declares its own dtype rather than sharing the
    # entry's type slot.
    InstructionEntry(
        name="fns",
        slots=(ModifierSlot("type", ("b32",)),),
        operands=(
            OperandSlot("mask", role="value", dtype="b32"),
            OperandSlot("base", role="value", dtype="b32"),
            OperandSlot("offset", role="value", dtype="s32"),
        ),
        effect=EFFECT_PURE,
        returns="type",
    ),
    # max per PTX ISA 9.7.3.12, two-source form. Deliberately excluded:
    # {.xorsign.abs} (a paired qualifier), the three-source
    # `max{.ftz}{.NaN}{.abs}.f32 d, a, b, c` line (a different operand shape,
    # so its own entry), and the half-precision forms of 9.7.4.8.
    InstructionEntry(
        name="max",
        slots=(
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("nan", ("NaN",), optional=True),
            ModifierSlot("type", ("f32", "f64")),
        ),
        check=_check_max,
        operands=(
            OperandSlot("a", role="value"),
            OperandSlot("b", role="value"),
        ),
        effect=EFFECT_PURE,
        asm_volatile=True,  # legacy barrier: preserved so migration is byte-identical
        returns="type",
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
