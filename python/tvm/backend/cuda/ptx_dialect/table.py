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
- One entry = one *syntax shape*: fixed operand list and result structure.
  Variants that change either (e.g. vector destinations) become separate
  entries, each declaring only the slots it uses.
- Predication (``@p``) is framework-level: every instruction without a
  destination accepts ``pred=`` at the call site; entries never mention it.

Calling convention: PTX has no defining form — a register is declared first
and instructions then write into it — so a ptxd call mirrors the PTX text
exactly. Destinations are ordinary operands (``role="dst"``, in PTX operand
order), every helper is ``void``, and every call is a statement::

    acc: T.float32                        # .reg .f32 acc;
    T.ptxd.add.rn.f32(acc, x, acc)        # add.rn.f32 acc, x, acc;
"""

import functools
import itertools
import keyword
from collections.abc import Callable
from dataclasses import dataclass

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
    # `.f32x2` operands "have .b64 type" (ISA 9.7.3.3): a pair of packed floats
    # in one 64-bit register, so the carrier is the bit container, not a float.
    "f32x2": ("uint64", "uint64_t", "l", "uint64_t"),
    # Mixed-precision sources live in a plain 16-bit register -- the ISA's own
    # example declares them `.reg .b16` -- so they ride the b16 carrier.
    "f16": ("uint16", "uint16_t", "h", "uint16_t"),
    "bf16": ("uint16", "uint16_t", "h", "uint16_t"),
    # 128-bit: only reachable as a whole register (mov.b128, ld/st .b128). The
    # "q" constraint requires __int128 support in the host compiler.
    "b128": ("uint128", "__uint128_t", "q", "__uint128_t"),
}


# TVM dtype -> (C type, asm constraint, carrier, into-carrier, out-of-carrier).
# `carrier == c_type` means the value binds its register directly. Where it does
# not, the value rides a differently-typed register and converts at the asm
# boundary: 8-bit values have no constraint letter of their own, and
# __half/__nv_bfloat16 cannot bind one at all (nvcc: "more than one conversion
# function ... applies"), so both go through a 16-bit integer register.
#
# Every conversion here is a bit pun, never an arithmetic cast. That distinction
# is the whole point of the axis: handing a float to a uint32_t parameter is a
# *numeric* conversion and emits `cvt.rzi.u32.f32`, silently changing the value.
DTYPE_C = {
    "uint8": ("uint8_t", "h", "uint16_t", "(uint16_t){}", "(uint8_t){}"),
    "int8": ("int8_t", "h", "int16_t", "(int16_t){}", "(int8_t){}"),
    "uint16": ("uint16_t", "h", "uint16_t", "{}", "{}"),
    "int16": ("int16_t", "h", "int16_t", "{}", "{}"),
    "float16": ("__half", "h", "uint16_t", "__half_as_ushort({})", "__ushort_as_half({})"),
    "bfloat16": (
        "__nv_bfloat16",
        "h",
        "uint16_t",
        "__bfloat16_as_ushort({})",
        "__ushort_as_bfloat16({})",
    ),
    "uint32": ("uint32_t", "r", "uint32_t", "{}", "{}"),
    "int32": ("int32_t", "r", "int32_t", "{}", "{}"),
    "float32": ("float", "f", "float", "{}", "{}"),
    "uint64": ("uint64_t", "l", "uint64_t", "{}", "{}"),
    "int64": ("int64_t", "l", "int64_t", "{}", "{}"),
    "float64": ("double", "d", "double", "{}", "{}"),
    "uint128": ("__uint128_t", "q", "__uint128_t", "{}", "{}"),
    "int128": ("__int128_t", "q", "__int128_t", "{}", "{}"),
}

# A bit-size PTX type names a width, not an interpretation -- ISA 5.2: "The
# bit-size type is compatible with any fundamental type having the same size."
# So an operand typed `.bN` accepts every TVM dtype of that width, and each gets
# its own helper. The canonical dtype is first: a variant that uses only
# canonical dtypes keeps the helper name it had before this axis existed.
#
# Concretely typed operands (.u32/.s32/.f32/...) are NOT widened: those name an
# interpretation, and substituting one is a semantic change rather than a
# relabelling. `.f32x2` likewise stays fixed -- it names a packed layout whose
# container happens to be .b64, not a bit container.
BIT_DTYPES = {
    "b8": ("uint8", "int8"),
    "b16": ("uint16", "int16", "float16", "bfloat16"),
    "b32": ("uint32", "int32", "float32"),
    "b64": ("uint64", "int64", "float64"),
    "b128": ("uint128", "int128"),
}


# Short token appended to a helper name for a non-canonical dtype choice.
DTYPE_SUFFIX = {
    "uint8": "u8", "int8": "s8",
    "uint16": "u16", "int16": "s16", "float16": "f16", "bfloat16": "bf16",
    "uint32": "u32", "int32": "s32", "float32": "f32",
    "uint64": "u64", "int64": "s64", "float64": "f64",
    "uint128": "u128", "int128": "s128",
}  # fmt: skip


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
      - ``"imm"``   an operand the ISA fixes to a single value (st.bulk's
        initval "must be zero"). Rendered straight into the asm text; it
        takes no C parameter and no call argument.
      - ``"dst"``   a destination the instruction writes, typed like
        ``"value"``. The helper takes it as a C++ reference and the caller
        passes a writable lvalue (a scalar or a buffer element), mirroring
        PTX's own "declare the register, then name it as an operand".

    ``lanes`` > 1 makes the operand a brace-enclosed register group. PTX writes
    the group in the operand list (``mov.b64 d, {lo, hi}``), so the group is
    part of the *shape*, never of the dotted modifier text.
    """

    name: str
    role: str
    space: str | None = None
    dtype: str | None = None
    literal: str | None = None  # role="imm" only
    # A PTX register group: `{%k, %k+1, ...}`. The operand takes `lanes` call
    # arguments and renders as one brace-enclosed vector expression. 1 = a plain
    # scalar operand, which is every operand of every other family.
    lanes: int = 1


CheckFn = Callable[[dict], str | None]


@dataclass(frozen=True)
class InstructionEntry:
    """One PTX instruction family (one syntax shape)."""

    name: str  # table key, e.g. "cvta", "st_bulk"; the surface name is `family`
    operands: tuple[OperandSlot, ...]
    slots: tuple[ModifierSlot, ...] = ()
    check: CheckFn | None = None  # cross-slot validation, mod_map -> error | None
    # Whether the emitted inline asm carries `volatile`. This is purely a
    # C-level optimization barrier — it never changes *which* PTX instruction
    # is emitted, only whether nvcc may common up or drop identical calls.
    # Every op registers as kOpaque (a void call has to survive RemoveNoOp),
    # so the IR shares nothing either way; this flag exists only to preserve
    # each instruction's established barrier byte-for-byte across migration.
    asm_volatile: bool = True
    # The PTX mnemonic, when it is not spellable as a Python identifier.
    # `st.bulk` is one instruction whose name contains a dot, so the table key
    # (st_bulk) and the emitted mnemonic (st.bulk) have to differ.
    mnemonic: str | None = None
    # The -arch every variant of this family must be certified at: the maximum
    # floor over its variants, not the minimum. Certifying below a variant's
    # floor makes ptxas report legal forms as illegal, and those verdicts would
    # then get baked into a check() and silently delete coverage.
    cert_arch: str | None = None

    @property
    def op_name(self) -> str:
        return f"tirx.ptxd.{self.name}"

    @property
    def ptx_name(self) -> str:
        return self.mnemonic or self.name

    @property
    def family(self) -> str:
        """The attribute users type: ``T.ptxd.<family>``.

        The mnemonic with dots folded to underscores. Equal to ``name`` for
        every single-shape family (``st.bulk`` -> ``st_bulk``); the shared
        surface name where several entries differ only in operand shape, as
        all the ``mov_*`` entries do.
        """
        return self.ptx_name.replace(".", "_")

    @property
    def has_dst(self) -> bool:
        """Whether the instruction writes a destination operand.

        Gates ``@p``: a false predicate leaves destinations unwritten, and the
        ``"="`` output constraint tells nvcc the prior value is dead, so a
        predicated destination silently loses it. Lifting this needs a
        read-modify-write ("+") constraint, which no entry declares yet.
        """
        return any(slot.role == "dst" for slot in self.operands)


def mods(entry: InstructionEntry, tokens) -> dict:
    """The canonical modifier map: every slot name -> token, ``""`` = omitted."""
    return {slot.name: tok or "" for slot, tok in zip(entry.slots, tokens)}


def operand_type(slot: OperandSlot, mod_map: dict) -> str:
    """The PTX type token of one operand.

    ``OperandSlot.dtype`` either names a modifier slot or is a literal type
    token; ``None`` means the entry's ``type`` slot. Naming an *optional* slot
    that was omitted falls back to ``type``, which is how the mixed-precision
    forms are typed: ``add.rn.f32.bf16``'s ``a`` is the ``.bf16`` source, while
    plain ``add.rn.f32``'s ``a`` is just ``.f32``.
    """
    key = slot.dtype or "type"
    if key in mod_map:
        return mod_map[key] or mod_map["type"]
    return key


def operand_space(slot: OperandSlot, mod_map: dict) -> str:
    """The state space of one ``addr`` operand.

    Fixed per operand for instructions whose operands live in different spaces
    (cp.async.bulk), else the entry's ``space`` modifier slot. This is the one
    definition: an address operand's C carrier (32-bit shared window vs generic
    pointer) hangs off it, so a family whose space slot is named anything else
    silently renders every shared form as a generic pointer.
    """
    return slot.space or mod_map.get("space", "")


def call_slots(entry: InstructionEntry) -> list[OperandSlot]:
    """The operand slot behind each call argument, in order.

    ISA-fixed immediates take no argument; a register group (``lanes`` > 1)
    takes one argument per lane, mirroring PTX's ``{%k, %k+1, ...}``.
    """
    return [s for s in entry.operands if s.role != "imm" for _ in range(s.lanes)]


def typed_operands(entry: InstructionEntry) -> list[OperandSlot]:
    """The operands that carry a dtype, in order: the dtype tuple aligns with these."""
    return [s for s in entry.operands if s.role in ("value", "dst")]


def operand_dtypes(slot: OperandSlot, mod_map: dict) -> tuple[str, ...]:
    """Every TVM dtype this operand accepts; the canonical one first."""
    token = operand_type(slot, mod_map)
    return BIT_DTYPES.get(token, (PTX_TYPES[token][0],))


def dtype_combos(entry: InstructionEntry, tokens) -> tuple[tuple[str, ...], ...]:
    """Every dtype assignment for one modifier combination, canonical first.

    One dtype per *operand*, never per lane: a register group is one operand
    that occupies N registers, and ISA 6.4.3 calls a brace list "similarly typed
    scalars". Operands multiply, so an instruction with two bit-typed operands
    has the product of their choices.
    """
    mod_map = mods(entry, tokens)
    axes = [operand_dtypes(s, mod_map) for s in typed_operands(entry)]
    return tuple(itertools.product(*axes)) if axes else ((),)


def pred_forms(entry: InstructionEntry) -> tuple[bool, ...]:
    """Which ``predicated`` renderings exist: both, unless the entry has a destination."""
    return (False,) if entry.has_dst else (False, True)


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
    sem, scope, ss = m["sem"], m["scope"], m["space"]
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
    """Scalar st grammar per PTX ISA 9.7.9.11 (the mirror of _check_ld)."""
    sem, scope, ss = m["sem"], m["scope"], m["space"]
    mmio, cop, l1ev = m["mmio"], m["cop"], m["l1ev"]
    # "st.relaxed.scope / st.release.scope" -- scope is mandatory there and
    # invalid on the weak/volatile forms (syntax lines).
    if sem in ("relaxed", "release") and not scope:
        return f"st.{sem} requires a scope (cta/cluster/gpu/sys)"
    if sem in ("", "weak", "volatile") and scope:
        return "only st.relaxed/st.release take a scope"
    if mmio:
        # "st.mmio.sem.sys{.global}": "Only .sys thread scope is valid for the
        # st.mmio operation." .release with .mmio arrives in PTX ISA 9.3; the
        # toolchain here assembles 9.2 and ptxas rejects it, so keep .relaxed
        # only and widen when the toolchain catches up.
        if sem != "relaxed":
            return "st.mmio requires .relaxed"
        if scope != "sys":
            return "only the sys scope is valid for st.mmio"
        if ss not in ("", "global"):
            return "st.mmio may only be used with .global or generic addressing"
        if cop or l1ev:
            return "st.mmio takes no cache qualifiers"
    if sem in ("relaxed", "release") and not mmio:
        # ".relaxed and .release: May be used with .global, .shared spaces or
        # with generic addressing... Cache operations are not allowed."
        if ss == "local":
            return f"st.{sem} is not valid on .local"
        if cop:
            return f"cache operations are not allowed with st.{sem}"
    if sem == "volatile" and (cop or l1ev):
        # "st.volatile{.ss}{.vec}.type" -- no cache qualifiers at all.
        return "st.volatile takes no cache qualifiers"
    if cop and l1ev:
        # cop and eviction_priority appear on separate syntax lines.
        return "cache operators and eviction priorities are mutually exclusive"
    if l1ev and ss not in ("", "global"):
        # ptxas: "Modifier '.evict_*' cannot be applied to '<ss>' space"
        return "L1 eviction priorities apply only to .global or generic addressing"
    return None


def _check_rcp(m):
    """This entry's rcp.approx is f32-only; .f64 is IEEE-rounded, no .ftz (PTX ISA 9.7.3.13)."""
    # Syntax lines: rcp.approx{.ftz}.f32 / rcp.rnd{.ftz}.f32 / rcp.rnd.f64
    if m["mode"] == "approx" and m["type"] != "f32":
        return (
            "rcp.approx.f64 is a separate syntax line (PTX ISA 9.7.3.14, where .ftz is "
            "mandatory) and is not registered here"
        )
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


def _check_farith(m):
    """Which qualifiers each add/sub/mul/fma syntax line allows (PTX ISA 9.7.3.{3,4,5,6}, 9.7.5).

    Same-precision lines:  op{.rnd}{.ftz}{.sat}.f32 | op{.rnd}{.ftz}.f32x2 | op{.rnd}.f64
    Mixed-precision lines: op{.rnd}{.sat}.f32.atype  (.atype = .f16 | .bf16)
    """
    ty, src = m["type"], m.get("srctype", "")
    if src:
        if ty != "f32":
            return f"mixed-precision .{src} source only exists on the .f32 line"
        if m.get("ftz"):
            return "the mixed-precision line takes no .ftz"
        return None
    if m.get("ftz") and ty == "f64":
        return "only the .f32/.f32x2 lines take .ftz"
    if m.get("sat") and ty != "f32":
        return "only the .f32 line takes .sat"
    return None


def _check_prefetch(m):
    """Each prefetch syntax line names exactly one target (PTX ISA 9.7.9.16).

    `.level::eviction_priority` stays bound to `.global` on purpose: its syntax
    line is `prefetch.global.level::eviction_priority`, with `.global` written
    in rather than the `{.ss}` that the `ld` lines carry. Generic addressing is
    not offered there, so neither is it here.
    """
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
    """op x type pairings for atom/red (PTX ISA 9.7.14.5 / 9.7.14.6).

    Normative source: ISA Table 35 (atom) and Table 36 (red), which give the
    pairing cell by cell. The `.type = {...}` line in the Syntax block is only
    the union across ops, which is why it cannot be transcribed directly. Half-precision
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


_LD_TYPES = (
    "b8", "u8", "s8", "b16", "u16", "s16", "b32", "u32",
    "s32", "b64", "u64", "s64", "f32", "f64",
)  # fmt: skip
_FRND = ("rn", "rz", "rm", "rp")  # .rnd on the floating-point arithmetic lines

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
    ),
    # Complete scalar `ld` per PTX ISA 9.7.9.8 + the 9.7.9.9 ld.global.nc
    # forms. Deliberately excluded (each needs a mechanism this shape lacks):
    # - .vec and the multi-register destination forms
    # - .b128: single-register and already in PTX_TYPES, but its "q" constraint
    #   needs __int128 in the host compiler; register on demand
    # - .level::cache_hint + the cache_policy operand (optional operand)
    # - .level2::eviction_priority (vector-only per the ISA)
    # - .unified (variable-attribute addressing)
    # - .param/.const spaces (require kernel-parameter / const addresses,
    #   which cannot flow through the helper-function ABI)
    # The ISA permits @p on this instruction; ptxd does not, because it writes a
    # destination -- see InstructionEntry.has_dst for why that needs a "+"
    # constraint first.
    InstructionEntry(
        name="ld",
        slots=(
            ModifierSlot("mmio", ("mmio",), optional=True),
            ModifierSlot("sem", ("weak", "acquire", "relaxed", "volatile"), optional=True),
            ModifierSlot("scope", ("cta", "cluster", "gpu", "sys"), optional=True),
            # Must be named "space": that is the slot an `addr` operand reads to
            # choose between a 32-bit shared-window address and a generic
            # pointer. Any other name silently leaves every shared form
            # rendering a 64-bit generic pointer, which ptxas accepts (it
            # truncates) but which addresses the wrong thing.
            ModifierSlot(
                "space",
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
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("addr", role="addr"),
        ),
    ),
    # Complete scalar `st` per PTX ISA 9.7.9.11, at parity with `ld`.
    # Deliberately excluded (each needs a mechanism this shape lacks):
    # - .vec/.b128 and the multi-register source forms
    # - .level::cache_hint + its trailing cache_policy operand (optional operand)
    # - .level2::eviction_priority (vector-only per the ISA)
    # - .param::func (kernel-parameter addresses cannot flow through the
    #   helper-function ABI)
    InstructionEntry(
        name="st",
        slots=(
            ModifierSlot("mmio", ("mmio",), optional=True),
            ModifierSlot("sem", ("weak", "release", "relaxed", "volatile"), optional=True),
            ModifierSlot("scope", _ATOM_SCOPES, optional=True),
            ModifierSlot(
                "space",
                ("global", "shared", "shared::cta", "shared::cluster", "local"),
                optional=True,  # omitted = generic addressing
            ),
            ModifierSlot("cop", ("wb", "cg", "cs", "wt"), optional=True),
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
            ModifierSlot("type", _LD_TYPES),
        ),
        check=_check_st,
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("value", role="value"),
        ),
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
    ),
    # The ISA permits @p on this instruction; ptxd does not, because it writes a
    # destination -- see InstructionEntry.has_dst for why that needs a "+"
    # constraint first.
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
            OperandSlot("d", role="dst"),
            OperandSlot("addr", role="addr"),
            OperandSlot("value", role="value"),
        ),
    ),
    # ex2 per PTX ISA 9.7.3.21 (`ex2.approx{.ftz}.f32`). The half-precision
    # forms of 9.7.4.10 (.f16/.f16x2/.bf16/.bf16x2) are deliberately excluded:
    # .f16/.bf16 have carriers now, but .f16x2/.bf16x2 need a b32 carrier and
    # .ftz is mandatory on the bf16 line while illegal on the f16 line, so they
    # cannot share this entry's optional ftz slot.
    InstructionEntry(
        name="ex2",
        slots=(
            ModifierSlot("mode", ("approx",)),
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("type", ("f32",)),
        ),
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("value", role="value"),
        ),
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
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("value", role="value"),
        ),
    ),
    # fns per PTX ISA 9.7.1.18: `fns.b32 d, mask, base, offset;` — one form.
    # The operands carry three different types (mask .b32, base .b32/.u32/.s32,
    # offset .s32), so each declares its own dtype rather than sharing the
    # entry's type slot.
    InstructionEntry(
        name="fns",
        slots=(ModifierSlot("type", ("b32",)),),
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("mask", role="value", dtype="b32"),
            OperandSlot("base", role="value", dtype="b32"),
            OperandSlot("offset", role="value", dtype="s32"),
        ),
        asm_volatile=False,  # legacy fns carried no barrier
    ),
    # max per PTX ISA 9.7.3.12, two-source form. Deliberately excluded:
    # {.xorsign.abs} (a paired qualifier), the three-source
    # `max{.ftz}{.NaN}{.abs}.f32 d, a, b, c` line (a different operand shape,
    # so its own entry), the half-precision forms of 9.7.4.8, and the entire
    # integer max family (9.7.1.14) -- ten .type tokens plus {.relu}, same
    # d, a, b shape, so a type-slot widening whenever it is wanted.
    InstructionEntry(
        name="max",
        slots=(
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("nan", ("NaN",), optional=True),
            ModifierSlot("type", ("f32", "f64")),
        ),
        check=_check_max,
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("a", role="value"),
            OperandSlot("b", role="value"),
        ),
    ),
    # st.bulk per PTX ISA 9.7.9.14:
    #   st.bulk{.weak}{.shared::cta} [a], size, initval;  // initval must be zero
    # Unregistered: the 32-bit `size` form (ISA: "The 32-bit or 64-bit integer
    # operand size ..."), because no modifier token distinguishes the two -- it
    # would be an operand-shape axis, like mov's. Two ISA constraints are also
    # unenforceable here, being properties of a value rather than of the
    # modifier map that check() sees: "size must be a multiple of 8" and "The
    # maximum value of size operand can be 16777216".
    InstructionEntry(
        name="st_bulk",
        mnemonic="st.bulk",
        cert_arch="sm_100",  # PTX ISA 8.6; ISA: "Requires sm_100 or higher."
        slots=(
            ModifierSlot("weak", ("weak",), optional=True),
            ModifierSlot("space", ("shared::cta",), optional=True),
        ),
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("size", role="value", dtype="u64"),
            OperandSlot("initval", role="imm", literal="0"),
        ),
    ),
    # Floating-point add/sub/mul (PTX ISA 9.7.3.{3,4,5}) together with their
    # mixed-precision lines (9.7.5.{1,2}); `mul` has no mixed-precision line.
    #   add{.rnd}{.ftz}{.sat}.f32  d, a, b;   add{.rnd}{.ftz}.f32x2  d, a, b;
    #   add{.rnd}.f64              d, a, b;   add{.rnd}{.sat}.f32.atype  d, a, c;
    # cert_arch is the family's ceiling, not its floor: .f32/.f64 assemble
    # everywhere, but .f32x2 and the mixed lines need sm_100, and certification
    # has to run somewhere every legal variant is legal.
    *[
        InstructionEntry(
            name=name,
            cert_arch="sm_100",
            slots=(
                ModifierSlot("rnd", _FRND, optional=True),
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("sat", ("sat",), optional=True),
                ModifierSlot("type", ("f32", "f64", "f32x2")),
                *((ModifierSlot("srctype", ("f16", "bf16"), optional=True),) if mixed else ()),
            ),
            check=_check_farith,
            operands=(
                OperandSlot("d", role="dst"),
                # On the mixed line `a` is the converted 16-bit source; on every
                # other line it is just the instruction type.
                OperandSlot("a", role="value", dtype="srctype" if mixed else None),
                OperandSlot("b", role="value"),
            ),
        )
        # `mul` is the one line with no mixed-precision form (ISA 9.7.5).
        for name, mixed in (("add", True), ("sub", True), ("mul", False))
    ],
    # Unregistered across this whole arithmetic group: the integer lines
    # (9.7.1.{1,2,3}), extended-precision add.cc/sub.cc (9.7.2.{1,3}), and the
    # half-precision lines (9.7.4.{1,2,3,4}), which additionally need .relu and
    # .oob slots.
    #
    # fma differs in shape, so it is its own entry: three sources, and .rnd is
    # mandatory on every line (PTX ISA 9.7.3.6 / 9.7.5.3).
    #   fma.rnd{.ftz}{.sat}.f32  d, a, b, c;   fma.rnd{.ftz}.f32x2  d, a, b, c;
    #   fma.rnd.f64              d, a, b, c;   fma.rnd{.sat}.f32.abtype  d, a, b, c;
    InstructionEntry(
        name="fma",
        cert_arch="sm_100",
        slots=(
            ModifierSlot("rnd", _FRND),
            ModifierSlot("ftz", ("ftz",), optional=True),
            ModifierSlot("sat", ("sat",), optional=True),
            ModifierSlot("type", ("f32", "f64", "f32x2")),
            ModifierSlot("srctype", ("f16", "bf16"), optional=True),
        ),
        check=_check_farith,
        operands=(
            OperandSlot("d", role="dst"),
            # .abtype converts both a and b; c is always the instruction type.
            OperandSlot("a", role="value", dtype="srctype"),
            OperandSlot("b", role="value", dtype="srctype"),
            OperandSlot("c", role="value"),
        ),
    ),
    # ------------------------------------------------------------------
    # mov, vector pack/unpack form (PTX ISA 9.7.9.4)
    #
    #   mov.type  d, a;   .type = {.b16, .b32, .b64, .b128}
    #
    # `.type` is the *aggregate* width, not the element width -- the ISA only
    # requires that "the overall size of the vector and the size of the scalar
    # must match the size of the instruction type". Neither the lane count nor
    # the lane type appears in the instruction text (the doc's own example is
    # `mov.b64 {lo,hi}, %x;  // %x is a double; lo,hi are .u32`), so both are
    # part of the operand shape, and each (direction, lanes, lane type) is its
    # own entry. They all share mnemonic "mov", so the emitted opcode is right
    # and the call spelling stays `T.ptxd.mov.b64(...)`.
    #
    # NOT REGISTERED -- legal PTX that CUDA C inline asm cannot express:
    #   mov.b16 d, {a,b}      2 x 8-bit lanes
    #   mov.b32 d, {a,b,c,d}  4 x 8-bit lanes
    # Inline asm has no 8-bit register constraint (only "h"/"r"/"l"/"q"/"f"/"d"),
    # so 8-bit values ride a 16-bit carrier and ptxas rejects the widths with
    # "Arguments mismatch for instruction 'mov'" (4x16 != 32). Both forms are
    # legal in hand-written PTX, and they are unreachable via make_uchar4 too
    # (nvcc emits shl/or, not the mov). Declaring `.reg .b8` inside the asm
    # block assembles but needs four cvt instructions to get the values in --
    # a multi-instruction template, which this dialect forbids.
    #
    # Also unregistered: the sink symbol `_` (ISA: "the sink symbol '_' may be
    # used for one or more elements"), which does assemble from inline asm but
    # needs an operand role for "this lane is discarded"; and scalar mov
    # (9.7.9.3), a different instruction that shares the mnemonic.
    *[
        InstructionEntry(
            name=f"mov_{direction}_{lane_dtype}x{lanes}",
            mnemonic="mov",
            slots=(ModifierSlot("type", (agg,)),),
            operands=(
                OperandSlot(
                    "d",
                    role="dst",
                    dtype=lane_dtype if unpack else agg,
                    lanes=lanes if unpack else 1,
                ),
                OperandSlot(
                    "a",
                    role="value",
                    dtype=agg if unpack else lane_dtype,
                    lanes=1 if unpack else lanes,
                ),
            ),
            # .b128 needs PTX ISA 8.3 / sm_70; the sm_90 certification default
            # already clears that, so no cert_arch is needed.
            asm_volatile=False,  # a register shuffle: let nvcc common it up
        )
        # Lane types are the bit types only: the dtype axis already lets a
        # `.b32` lane be int32 or float32, so a separate `f32` entry would be a
        # shape-for-shape duplicate and make the shared-mnemonic dispatch
        # ambiguous.
        for agg, lanes, lane_dtype in (
            ("b32", 2, "b16"),
            ("b64", 2, "b32"),
            ("b64", 4, "b16"),
            ("b128", 2, "b64"),
            ("b128", 4, "b32"),
        )
        for direction, unpack in (("pack", False), ("unpack", True))
    ],
    # cvta per PTX ISA 9.7.9.7. This entry exists to serve the engine's
    # shared-address coercion, so it registers exactly the one combination that
    # needs: 1 of the 32 legal (direction x space x size) forms. Unregistered:
    # the whole space->generic direction, seven of the eight state spaces, and
    # `.u32` -- the last genuinely unusable, since ptxas rejects the 32-bit ABI
    # on sm_90 and higher.
    InstructionEntry(
        name="cvta",
        slots=(
            ModifierSlot("dir", ("to",)),
            ModifierSlot("space", ("shared",)),
            ModifierSlot("type", ("u64",)),
        ),
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("ptr", role="ptr"),
        ),
        asm_volatile=False,  # legacy cvta carried no barrier
    ),
    # cp.async.bulk per PTX ISA 9.7.9.26. One of the ISA's eight syntax lines is
    # registered; unregistered are the .sem/.scope/.type form, .L2::cache_hint,
    # .ignore_oob, .multicast::cluster, .cp_mask, and three of the four copy
    # directions. `{.sem}` is additionally blocked by the toolchain: it is PTX
    # ISA 9.3 and ptxas 13.2 assembles 9.2, the same situation _check_ld already
    # records for ld.mmio.acquire.
    #
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
    ),
]

TABLE: dict[str, InstructionEntry] = {e.name: e for e in _ENTRIES}
