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
import re
from collections.abc import Callable
from dataclasses import dataclass

# PTX operand type token -> the TVM dtypes it accepts, canonical first.
#
# A *bit-size* type names a width, not an interpretation -- ISA 5.2: "The
# bit-size type is compatible with any fundamental type having the same size."
# So `.bN` accepts every TVM dtype of that width and each gets its own helper
# (see `operand_dtypes`). The canonical one is listed first: a variant using
# only canonical dtypes keeps the helper name it had before the axis existed.
#
# A concretely typed operand (.u32/.s32/.f32/...) names an interpretation, so
# it accepts that one dtype -- substituting another would be a semantic change
# rather than a relabelling.
PTX_TYPE_DTYPES = {
    "b8": ("uint8", "int8"),
    "b16": ("uint16", "int16", "float16", "bfloat16"),
    "b32": ("uint32", "int32", "float32"),
    "b64": ("uint64", "int64", "float64"),
    "b128": ("uint128", "int128"),
    "u8": ("uint8",),
    "s8": ("int8",),
    "u16": ("uint16",),
    "s16": ("int16",),
    "u32": ("uint32",),
    "s32": ("int32",),
    "u64": ("uint64",),
    "s64": ("int64",),
    "f32": ("float32",),
    "f64": ("float64",),
    # `.f32x2` operands "have .b64 type" (ISA 9.7.3.3): a pair of packed floats
    # in one 64-bit register. It names a packed layout whose container happens
    # to be .b64, not a bit container, so it is not widened.
    "f32x2": ("uint64",),
    # Mixed-precision sources live in a plain 16-bit register -- the ISA's own
    # example declares them `.reg .b16`.
    "f16": ("uint16",),
    "bf16": ("uint16",),
    # Packed SIMD types, same reasoning as `.f32x2`: the token names a layout,
    # its container is one ordinary register of the matching width.
    "u16x2": ("uint32",),
    "s16x2": ("uint32",),
    "u8x4": ("uint32",),
    "s8x4": ("uint32",),
    "f16x2": ("uint32",),
    "bf16x2": ("uint32",),
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


LanesFn = Callable[[dict], int]  # modifier map -> registers in this operand's group


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
      - ``"imm"``   an operand that lives in the instruction *text*, not in a
        register. With ``literal`` the ISA fixes its value (st.bulk's initval
        "must be zero"): no C parameter, no call argument. With ``choices``
        the caller picks the value -- a compile-time constant, validated
        against the closed set at trace time and baked into the text, with
        one helper generated per value (the way `cp.async.wait_group N` or
        `setmaxnreg`'s nreg exist only as integer literals in the ISA).
      - ``"dst"``   a destination the instruction writes, typed like
        ``"value"``. The helper takes it as a C++ reference and the caller
        passes a writable lvalue (a scalar or a buffer element), mirroring
        PTX's own "declare the register, then name it as an operand".
      - ``"acc"``   a register the instruction reads AND writes -- an in-place
        accumulator, bound "+" where a dst binds "=". Takes an lvalue like a
        dst; unlike a dst it does not block ``@p``, because "+" keeps the old
        value live under a false predicate.
      - ``"pred_dst"`` a ``.pred`` result. Inline asm has no constraint letter
        for predicate registers, so the boundary conversion lives inside the
        block: the instruction writes a predicate register and a trailing
        ``selp.b32`` materializes it as 0/1 into a "=r" uint32 the caller
        receives through a reference parameter, exactly like a dst (and it
        gates ``@p`` like one).
      - ``"pred_src"`` a ``.pred`` argument. The mirror conversion: a leading
        ``setp.ne.b32`` turns a "r"-bound uint32 into the predicate register
        the instruction reads. These are the block's second and third
        boundary-conversion exceptions -- ``@p``'s own setp was the first.

    ``lanes`` > 1 makes the operand a brace-enclosed register group. PTX writes
    the group in the operand list (``mov.b64 d, {lo, hi}``), so the group is
    part of the *shape*, never of the dotted modifier text.
    """

    name: str
    role: str
    space: str | None = None
    dtype: str | None = None
    literal: str | None = None  # role="imm": ISA-fixed value
    choices: tuple[str, ...] | None = None  # role="imm": caller-chosen value
    # A PTX register group: `{%k, %k+1, ...}`. The operand takes `lanes` call
    # arguments and renders as one brace-enclosed vector expression. 1 = a plain
    # scalar operand, which is every operand of every other family.
    #
    # A callable makes the group length a function of the modifier map -- the
    # ISA states these lengths as formulas over the modifiers ("r is a register
    # vector of length shape * num / 32b"), and the modifiers are a closed set,
    # so every resulting length is still enumerable and certifiable. Same
    # contract as `check`: total over every combination `variants()` generates.
    lanes: int | LanesFn = 1
    # Whether PTX writes this operand as a brace-enclosed vector. Defaults to
    # "it has more than one register", which a length function cannot answer
    # ahead of time -- an operand whose length varies between 0 and 1 is a
    # bracketed-optional scalar, not a vector.
    vector: bool | None = None
    # A composite memory operand: adjacent slots naming the same bracket render
    # inside one pair of square brackets, each keeping its own C parameters and
    # constraints. This is how TMA spells its address -- `[tensorMap,
    # tensorCoords]` is a single PTX operand holding a 64-bit pointer and an
    # .s32 coordinate vector that have no single-register encoding.
    bracket: str | None = None


CheckFn = Callable[[dict], str | None]


@dataclass(frozen=True)
class InstructionEntry:
    """One PTX instruction family (one syntax shape).
    ``orders_memory`` marks an instruction that constrains the order of *other*
    threads' memory accesses without naming any address itself -- fences,
    barriers, async-group waits. The ``"memory"`` clobber is otherwise derived
    from "does this entry have an ``addr`` operand", which is the right rule for
    instructions that touch memory but cannot see a fence: ``asm volatile``
    alone only pins the asm block, it does not stop the compiler moving ordinary
    loads and stores across it.
    """

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
    orders_memory: bool = False
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

    @functools.cached_property
    def typed_operands(self) -> tuple[OperandSlot, ...]:
        """The operands carrying a dtype, in order: a dtype tuple aligns with these."""
        return tuple(s for s in self.operands if s.role in ("value", "dst", "acc"))

    @property
    def has_dst(self) -> bool:
        """Whether the instruction writes a destination operand.

        Gates ``@p``: a false predicate leaves destinations unwritten, and the
        ``"="`` output constraint tells nvcc the prior value is dead, so a
        predicated destination silently loses it. An accumulator (``role="acc"``)
        binds "+" instead, which keeps the old value live -- so it does not
        count here and @p remains available on it. A pred_dst is written
        through "=" the same way a dst is, so it counts.
        """
        return any(slot.role in ("dst", "pred_dst") for slot in self.operands)


def mods(entry: InstructionEntry, tokens) -> dict:
    """The canonical modifier map: every slot name -> token, ``""`` = omitted."""
    return {slot.name: tok or "" for slot, tok in zip(entry.slots, tokens)}


def lanes_of(slot: OperandSlot, mod_map: dict) -> int:
    """How many registers this operand's group occupies under these modifiers."""
    return slot.lanes(mod_map) if callable(slot.lanes) else slot.lanes


def operand_layout(entry, mod_map: dict) -> tuple[tuple[OperandSlot, int, int], ...]:
    """``(slot, first_arg_index, lanes)`` per operand the caller supplies.

    One row per *operand*, not per argument: a register group is one operand
    occupying N registers, so its dtype and its coercion are decided once for
    the whole group. Depends on the modifiers because a group's length may --
    the modifier set is closed, so there is one layout per token combination,
    memoized below.
    """
    return _operand_layout(entry, tuple(mod_map.values()))


@functools.cache
def _operand_layout(entry, mod_values):
    mod_map = dict(zip((s.name for s in entry.slots), mod_values))
    rows, i = [], 0
    for slot in entry.operands:
        if slot.role == "imm" and slot.choices is None:
            continue
        n = lanes_of(slot, mod_map)
        rows.append((slot, i, n))
        i += n
    return tuple(rows)


def tokens_for(entry: InstructionEntry, **by_name) -> tuple[str, ...]:
    """Modifier tokens in slot order, named instead of positional.

    The inverse of :func:`mods`. A positional tuple silently shifts every token
    when a slot is inserted -- the exact edit this table invites -- so anything
    naming a specific variant (tests, tools) should name its slots.
    """
    unknown = set(by_name) - {slot.name for slot in entry.slots}
    if unknown:
        raise ValueError(f"{entry.name}: no modifier slot named {sorted(unknown)}")
    out = []
    for slot in entry.slots:
        token = by_name.get(slot.name, "")
        if token and token not in slot.choices:
            raise ValueError(f"{entry.name}.{slot.name}: {token!r} not in {slot.choices}")
        if not token and not slot.optional:
            raise ValueError(f"{entry.name}.{slot.name} is required")
        out.append(token)
    out = tuple(out)
    # Slot-level legality is not the whole rule: `check` rejects combinations
    # whose tokens are individually fine (ld.volatile takes no scope). Without
    # this a golden test could pin a variant the dialect refuses to emit.
    if entry.check is not None:
        error = entry.check(mods(entry, out))
        if error:
            raise ValueError(f"{entry.name}: {error}")
    return out


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


def operand_dtypes(slot: OperandSlot, mod_map: dict) -> tuple[str, ...]:
    """The TVM dtypes one operand accepts, canonical first (see PTX_TYPE_DTYPES)."""
    return PTX_TYPE_DTYPES[operand_type(slot, mod_map)]


def canonical_dtypes(entry: InstructionEntry, tokens) -> tuple[str, ...]:
    """The canonical dtype of each typed operand, in operand order.

    This is `dtype_combos(...)[0]`, but as the per-operand fact it actually is:
    "canonical" belongs to an operand, not to the product of all of them.
    """
    mod_map = mods(entry, tokens)
    return tuple(operand_dtypes(s, mod_map)[0] for s in entry.typed_operands)


def dtype_combos(entry: InstructionEntry, tokens) -> tuple[tuple[str, ...], ...]:
    """Every dtype assignment for one modifier combination, canonical first.

    One dtype per *operand*, never per lane: a register group is one operand
    that occupies N registers, and ISA 6.4.3 calls a brace list "similarly typed
    scalars". Operands multiply, so an instruction with two bit-typed operands
    has the product of their choices.
    """
    mod_map = mods(entry, tokens)
    axes = [operand_dtypes(s, mod_map) for s in entry.typed_operands]
    return tuple(itertools.product(*axes)) if axes else ((),)


def renderings(entry: InstructionEntry):
    """Every ``(tokens, dtypes, predicated)`` this entry renders to.

    The product of the four independent axes -- modifiers, operand dtypes,
    caller immediates, predication. Defined once so a new axis lands in one
    place instead of in every consumer (the certification tiers, the helper
    dump, the stub writer).
    """
    for tokens in variants(entry):
        for dtypes in dtype_combos(entry, tokens):
            for imms in imm_combos(entry):
                for predicated in pred_forms(entry):
                    yield tokens, dtypes, predicated, imms


def imm_slots(entry: InstructionEntry) -> tuple[OperandSlot, ...]:
    """The caller-chosen immediates, in operand order; an imm tuple aligns with these."""
    return tuple(s for s in entry.operands if s.role == "imm" and s.choices is not None)


def imm_combos(entry: InstructionEntry) -> tuple[tuple[str, ...], ...]:
    """Every caller-immediate assignment. The domains are closed on purpose:
    that is what makes each generated helper certifiable."""
    axes = [s.choices for s in imm_slots(entry)]
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


def _scalar_view(m):
    """The scalar grammar reads slots the vector entries do not declare."""
    return {**m, "mmio": "", "scope": "", "l2ev": m.get("l2ev", "")}


def _check_ld_vec(m):
    return _check_vec128(m) or _check_ld(_scalar_view(m))


def _check_st_vec(m):
    return _check_vec128(m) or _check_st(_scalar_view(m))


def _check_ld_vec256(m):
    return _check_vec256(m) or _check_ld(_scalar_view(m))


def _check_st_vec256(m):
    return _check_vec256(m) or _check_st(_scalar_view(m))


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


# The .type tokens of the integer min/max lines (PTX ISA 9.7.1.13/9.7.1.14).
# `.relu` is only on the signed line (type2); the unsigned/other line takes none.
# NOT REGISTERED: `.u8x4` and `{.relu}.s8x4`, which the ISA supports only on
# "sm_120f or higher in the same family" -- outside the architectures this
# dialect certifies, and ptxas rejects them at sm_100.
_MINMAX_INT_PLAIN = ("u16", "u32", "u64", "u16x2", "s16", "s64")
_MINMAX_INT_RELU = ("s16x2", "s32")
# The floating-point and half-precision lines (9.7.3.12/9.7.3.13, 9.7.4.8/9.7.4.9).
_MINMAX_FP = ("f32", "f64", "f16", "f16x2", "bf16", "bf16x2")


def _check_mbarrier_sem_scope(m):
    """`.sem` and `.scope` are one qualifier pair: both or neither (ISA 9.7.14.9)."""
    if bool(m.get("sem", "")) != bool(m.get("scope", "")):
        return ".sem and .scope go together: write both or neither"
    return None


def _check_fence_proxy(m):
    """`.proxykind = { .alias, .async, .async.global, .async.shared::{cta,cluster} }`.

    Modelled as proxykind + an optional space so the surface can walk it one
    attribute at a time; only `.async` carries a state space (ISA 9.7.14.4).
    """
    if m["space"] and m["proxykind"] != "async":
        return f"fence.proxy.{m['proxykind']} takes no state space"
    return None


def _check_minmax(m):
    """Which qualifiers each min/max syntax line allows.

    Integer lines (9.7.1.13/14):  op.type1 d,a,b   |  op{.relu}.type2 d,a,b
    Floating lines (9.7.3.12/13): op{.ftz}{.NaN}{.xorsign.abs}.f32 d,a,b
                                  op.f64 d,a,b
    Half lines (9.7.4.8/9):       op{.ftz}{.NaN}{.xorsign.abs}.f16{x2} d,a,b
                                  op{.NaN}{.xorsign.abs}.bf16{x2} d,a,b
    `.xorsign` and `.abs` are one paired qualifier `{.xorsign.abs}` -- two slots
    only because the surface reaches them one attribute at a time.
    """
    ty = m["type"]
    ftz, nan, xorsign, abs_, relu = (
        m.get("ftz", ""),
        m.get("nan", ""),
        m.get("xorsign", ""),
        m.get("abs", ""),
        m.get("relu", ""),
    )
    is_int = ty in _MINMAX_INT_PLAIN or ty in _MINMAX_INT_RELU
    if is_int:
        if ftz or nan or xorsign or abs_:
            return f".{ty} is an integer line and takes no .ftz/.NaN/.xorsign.abs"
        if relu and ty not in _MINMAX_INT_RELU:
            return f".relu is only on {'/'.join(_MINMAX_INT_RELU)}, not .{ty}"
        return None
    if relu:
        return f".relu is an integer-line qualifier, not valid on .{ty}"
    if ty == "f64" and (ftz or nan or xorsign or abs_):
        return "the .f64 line is the bare form: no .ftz/.NaN/.xorsign.abs"
    if ty.startswith("bf16") and ftz:
        return f".{ty} takes no .ftz"
    if bool(xorsign) != bool(abs_):
        return ".xorsign and .abs are one paired qualifier: write both or neither"
    return None


def _check_minmax3(m):
    """The three-source line: `op{.ftz}{.NaN}{.abs}.f32 d, a, b, c` (9.7.3.12).

    `.abs` here is standalone -- unlike the two-source line, which pairs it with
    `.xorsign`.
    """
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
    "s32", "b64", "u64", "s64", "f32", "f64", "b128",
)  # fmt: skip
# ISA 5.4.2 caps a vector register at 128 bits, so .v2.b128 does not exist.
_LD_VEC_TYPES = tuple(t for t in _LD_TYPES if t != "b128")
_L1_EVICT = (
    "L1::evict_normal",
    "L1::evict_unchanged",
    "L1::evict_first",
    "L1::evict_last",
    "L1::no_allocate",
)
# ptxas rejects ".L2::evict_unchanged" on ld/st ("Illegal modifier"), though
# the ISA lists it among the eviction priorities.
_L2_EVICT = ("L2::evict_first", "L2::evict_last", "L2::evict_normal")
_BITS32 = ("b32", "u32", "s32", "f32")
_BITS64 = ("b64", "u64", "s64", "f64")


def _vec_lanes(m):
    # ISA 9.7.9.8/9.7.9.11: the destination/source is a brace-enclosed vector
    # of `.vec` registers.
    return int(m["vec"][1:])


def _check_vec128(m):
    """The .vec lines up to 128 bits wide."""
    if m["vec"] == "v4" and m["type"] in _BITS64:
        # 256 bits wide: a separate entry, with its own sm_100 floor.
        return "a 64-bit .v4 is a 256-bit access -- use the 256-bit entry"
    return None


def _check_vec256(m):
    """The two 256-bit .vec lines, which the ISA spells out individually."""
    vec, ty, ss = m["vec"], m["type"], m["space"]
    if not ((vec == "v8" and ty in _BITS32) or (vec == "v4" and ty in _BITS64)):
        return "the 256-bit lines are .v8 with a 32-bit type or .v4 with a 64-bit type"
    if ss not in ("", "global"):
        # Both lines: "State space is .global or generic addressing where the
        # address points to .global".
        return "a 256-bit access takes .global or generic addressing"
    return None


_FRND = ("rn", "rz", "rm", "rp")  # .rnd on the floating-point arithmetic lines


def _matrix_num_lanes(m):
    # ISA 9.7.15.5.16 (stmatrix): "a brace-enclosed vector expression consisting
    # of 1, 2, or 4 32-bit registers as per the value of .num" -- no shape term,
    # unlike ldmatrix's .m16n16 doubling.
    return int(m["num"][1:])


def _ldmatrix_lanes(m):
    # ISA 9.7.15.5.15: "a brace-enclosed vector expression consisting of 1, 2,
    # or 4 32-bit registers as per the value of .num" -- and, for shape 16x16,
    # "two destination registers r0 and r1 of type .b32 must be specified" per
    # matrix, so .m16n16 doubles the count (ptxas: "Vector of size 2 is
    # expected"). That doubling is also why .m16n16 caps .num at .x2.
    return int(m["num"][1:]) * (2 if m["shape"] == "m16n16" else 1)


def _check_ldmatrix_b8fmt(m):
    # Line 2 (`.m8n16`) spells no `.trans`; line 3 (`.m16n16`) spells it
    # mandatorily. "When .shape is .m16n16, only .x1 and .x2 are valid".
    if m["shape"] == "m8n16" and m["trans"]:
        return "the .m8n16 line takes no .trans"
    if m["shape"] == "m16n16":
        if not m["trans"]:
            return "the .m16n16 decompression line requires .trans"
        if m["num"] == "x4":
            return "shape m16n16 supports only .x1 and .x2"
    return None


def _tcgen05_ldst_lanes(m):
    # ISA 9.7.17.8.3 Table 52 / 9.7.17.8.4 Table 53: the register vector holds
    # `.num` x (shape width / 32b) registers, capped at 128.
    per_num = {"16x64b": 1, "32x32b": 1, "16x128b": 2, "16x256b": 4}
    return int(m["num"][1:]) * per_num[m["shape"]]


def _check_tcgen05_ldst(m):
    """The Table 52/53 rows marked NA -- the products that exceed 128 registers."""
    if _tcgen05_ldst_lanes(m) > 128:
        return f"shape {m['shape']} caps .num where the vector would exceed 128 registers"
    return None


# mma fragment sizes, per the Matrix Fragments tables of ISA 9.7.15.5.1-13.
# Each is `rows * cols * bits / threads / 32`, the register count a thread holds
# of an MxN (or MxK / KxN) tile -- the ISA states the tables, this states the
# rule they follow. .m8n8k4 with .f16 multiplicands is the one shape a warp
# runs as four independent 8-thread MMAs, so its A/B fragments divide by 8.
_MMA_BITS = {
    "f16": 16, "bf16": 16, "tf32": 32, "f32": 32, "f64": 64, "s32": 32,
    "u8": 8, "s8": 8, "u4": 4, "s4": 4, "b1": 1,
    "e4m3": 8, "e5m2": 8, "e3m2": 6, "e2m3": 6, "e2m1": 4,
}  # fmt: skip


def _mma_shape(m):
    mm, nn, kk = re.match(r"m(\d+)n(\d+)k(\d+)", m["shape"]).groups()
    return int(mm), int(nn), int(kk)


def _mma_regs(dtype, rows, cols, threads, reg_bits=32):
    return max(1, rows * cols * _MMA_BITS[dtype] // threads // reg_bits)


def _mma_threads(m):
    # ISA: "A warp executing mma.sync.m8n8k4 computes 4 matrix multiply and
    # accumulate operations". Each is over 8 threads, so a thread's fragment is
    # an eighth of the tile rather than a thirty-second -- ptxas agrees
    # (d=4, a=2, b=2 for .f16; anything else is "Arguments mismatch").
    return 8 if m["shape"] == "m8n8k4" else 32


def _mma_lanes(which):
    """Registers in one of the four operand groups, as a function of the modifiers."""

    def lanes(m):
        mm, nn, kk = _mma_shape(m)
        threads = _mma_threads(m)
        # f64 fragments live in .f64 registers, everything else in .b32.
        reg_bits = 64 if m["dtype"] == "f64" else 32
        if which == "d":
            return _mma_regs(m["dtype"], mm, nn, threads, reg_bits)
        if which == "c":
            return _mma_regs(m["ctype"], mm, nn, threads, reg_bits)
        if which == "a":
            return _mma_regs(m["atype"], mm, kk, threads, reg_bits)
        return _mma_regs(m["btype"], kk, nn, threads, reg_bits)

    return lanes


def _check_mma_fp_f16(m):
    """The .f16-accumulator forms: the ISA's f16 and f8 lines."""
    if m["atype"] != m["btype"]:
        return "the multiplicand types must match"
    shape, a = m["shape"], m["atype"]
    if shape != "m8n8k4" and (m["alayout"], m["blayout"]) != ("row", "col"):
        return f"{shape} is spelled .row.col"
    if a == "f16" and shape not in ("m8n8k4", "m16n8k8", "m16n8k16"):
        return f"{shape} has no .f16 multiplicand line"
    if a in ("e4m3", "e5m2") and shape not in ("m16n8k16", "m16n8k32"):
        return f"{shape} has no .f8 multiplicand line"
    return None


def _check_mma_fp_f32(m):
    """One check per floating-point syntax line of ISA 9.7.15.5.14.

    The lines differ in which shapes pair with which operand types, and only
    the .m8n8k4 line leaves the layouts free -- every other line spells
    `.row.col`.
    """
    shape, d, c = m["shape"], m["dtype"], m["ctype"]
    a, b = m["atype"], m["btype"]
    if a != b:
        # Every floating-point line spells the two operand types identically.
        return "the multiplicand types must match"
    if shape != "m8n8k4" and (m["alayout"], m["blayout"]) != ("row", "col"):
        return f"{shape} is spelled .row.col"
    if a == "f16":
        # "mma.m8n8k4 / m16n8k8 / m16n8k16 .dtype.f16.f16.ctype"
        if shape not in ("m8n8k4", "m16n8k8", "m16n8k16"):
            return f"{shape} has no .f16 multiplicand line"
        if shape == "m8n8k4" and c == "f32" and d != "f32":
            return "m8n8k4 with a .f32 accumulator must produce .f32"
    elif a == "tf32":
        # "m16n8k4 .f32.tf32.tf32.f32" and the m16n8k8 atype/btype line.
        if shape not in ("m16n8k4", "m16n8k8") or d != "f32" or c != "f32":
            return "the .tf32 lines are m16n8k4 / m16n8k8, .f32 in and out"
    elif a == "bf16":
        # "m16n8k16 .f32.bf16.bf16.f32" and the m16n8k8 atype/btype line.
        if shape not in ("m16n8k8", "m16n8k16") or d != "f32" or c != "f32":
            return "the .bf16 lines are m16n8k8 / m16n8k16, .f32 in and out"
    else:  # e4m3 / e5m2
        # "mma.shape.row.col.dtype.f8type.f8type.ctype", shape in k16/k32.
        if shape not in ("m16n8k16", "m16n8k32"):
            return f"{shape} has no .f8 multiplicand line"
    if shape in ("m16n8k8", "m16n8k16", "m16n8k32") and d != c:
        # ".dtype must be the same as .ctype" on these shapes.
        return f"{shape} requires .dtype == .ctype"
    return None


def _check_mma_int(m):
    """The integer / sub-byte / single-bit lines of ISA 9.7.15.5.14."""
    shape, a, b = m["shape"], m["atype"], m["btype"]
    if a != b:
        # "the values for .atype and .btype must be the same"
        return "the multiplicand types must match"
    if a in ("u8", "s8"):
        if shape not in ("m8n8k16", "m16n8k16", "m16n8k32"):
            return f"{shape} has no 8-bit integer line"
    elif a in ("u4", "s4"):
        if shape not in ("m8n8k32", "m16n8k32", "m16n8k64"):
            return f"{shape} has no 4-bit integer line"
    else:  # b1
        if shape not in ("m8n8k128", "m16n8k128", "m16n8k256"):
            return f"{shape} has no single-bit line"
        if not m["bitop"]:
            # "mma.sync...s32.b1.b1.s32.bitOp.popc" -- bitOp is not optional.
            return "the single-bit line requires .xor or .and"
    if a != "b1" and (m["bitop"] or m["popc"]):
        # `.bitOp.popc` belongs to the single-bit line alone; ptxas otherwise
        # reports "Unexpected instruction types specified for 'mma'".
        return "only the single-bit line takes .bitOp.popc"
    if a == "b1":
        if m["satfinite"]:
            return "the single-bit line takes no .satfinite"
        if not m["popc"]:
            return "the single-bit line spells .popc"
    return None


# wgmma.mma_async register fragments, per ISA 9.7.16.4 (Warpgroup matrix
# fragments): across the 128-thread warpgroup the accumulator D holds
# M*N/128 = N/2 registers per thread (.f32 and .s32), and M*N/256 = N/4
# when .dtype is .f16 (two halves per register). The A fragment of the rs
# form works out to M*K/128/(32/bits) = 4 registers for every (K, type)
# pairing the ISA defines, so it is a plain `lanes=4`, not a function.
#
# The N domains, straight from each syntax line's `.shape =` set: the
# floating-point and fp8 lines take every multiple of 8 up to 256; the
# s8/u8 line drops 40, 56, ... (the odd multiples of 8 above 32) and stops
# at 224; the single-bit line adds 240 and 256 back.
_WGMMA_N_FULL = tuple(str(8 * i) for i in range(1, 33))
_WGMMA_N_S8 = ("8", "16", "24", "32", "48", "64", "80", "96", "112", "128",
               "144", "160", "176", "192", "208", "224")  # fmt: skip
_WGMMA_N_B1 = (*_WGMMA_N_S8, "240", "256")


def _wgmma_acc_lanes(m):
    n = int(m["shape"].split("n")[1].split("k")[0])
    return n // 4 if m["dtype"] == "f16" else n // 2


# cp.async.bulk.tensor coordinate vectors, per ISA 9.7.9.26.5.2: "Vector of
# n elements where n = .dim" -- except the gather4/scatter4 load modes, whose
# tensorCoords is a "fixed length vector of size 5" (one column index plus
# four row indices) whatever the dimension.
def _tma_coords_lanes(m):
    if m["load_mode"] in ("tile::gather4", "tile::scatter4"):
        return 5
    return int(m["dim"][0])


def _tma_mask_lanes(m):
    # `{, ctaMask}` exists exactly when `.multicast::cluster` is written.
    return 1 if m["multicast"] else 0


def _tma_cache_lanes(m):
    # `{, cache_policy}` exists exactly when `.L2::cache_hint` is written.
    return 1 if m["cache"] else 0


def _check_tma_gather4(m):
    """gather4/scatter4 reindex four rows of a 2D tensor (ISA: "the four rows
    in the 2-dimensional tensor"), so ptxas rejects every other .dim."""
    if m["load_mode"] in ("tile::gather4", "tile::scatter4") and m["dim"] != "2d":
        return f"{m['load_mode']} is a 2d-only load mode"
    return None


# tcgen05.mma disable-output-lane, per ISA 9.7.17.10.9.1: "The size of the
# vector is as follows: .cta_group::1 -> 4, .cta_group::2 -> 8".
def _tcgen05_mma_mask_lanes(m):
    return 8 if m["cta_group"] == "cta_group::2" else 4


def _check_tcgen05_mma_block_scale(m):
    """Valid .scale_vec sizes per kind (ISA "Scale factor" table): mxf8f6f4
    scales in 1X, mxf4 in 2X, mxf4nvf4 in 2X or 4X."""
    valid = {
        "kind::mxf8f6f4": ("scale_vec::1X",),
        "kind::mxf4": ("scale_vec::2X",),
        "kind::mxf4nvf4": ("scale_vec::2X", "scale_vec::4X"),
    }[m["kind"]]
    if m["scale_vec"] not in valid:
        return f"{m['kind']} scales in {'/'.join(valid)}"
    return None


def _check_cp_async_bulk_s2g(m):
    """ptxas: .cp_mask without .L2::cache_hint is rejected (the legacy helper
    raised on the same pairing)."""
    if m["cp_mask"] and not m["cache"]:
        return ".cp_mask requires .L2::cache_hint"
    return None


def _cp_mask_lanes(m):
    # `{, byteMask}` exists exactly when `.cp_mask` is written.
    return 1 if m["cp_mask"] else 0


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
    # - .level::cache_hint + the cache_policy operand (optional operand)
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
            ModifierSlot("l1ev", _L1_EVICT, optional=True),
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
    # - .level::cache_hint + its trailing cache_policy operand (optional operand)
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
            ModifierSlot("l1ev", _L1_EVICT, optional=True),
            ModifierSlot("type", _LD_TYPES),
        ),
        check=_check_st,
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("value", role="value"),
        ),
    ),
    # The `.vec` lines of ld / st (PTX ISA 9.7.9.8 / 9.7.9.11). Separate
    # entries rather than an optional slot on the scalar ones: a vector operand
    # is brace-enclosed even at one register, so vector-ness has to be a
    # property of the entry, and the .v8/.v4-64bit lines carry a
    # .level2::eviction_priority the scalar lines do not have.
    #
    # NOT REGISTERED: the sink symbol `_` in the two 256-bit lines (it needs an
    # operand role for "this lane is discarded"), and .level::cache_hint with
    # its trailing cache_policy operand.
    InstructionEntry(
        name="ld_vec",
        mnemonic="ld",
        slots=(
            ModifierSlot("sem", ("weak", "volatile"), optional=True),
            ModifierSlot(
                "space",
                ("global", "shared", "shared::cta", "shared::cluster", "local"),
                optional=True,
            ),
            ModifierSlot("cop", ("ca", "cg", "cs", "lu", "cv"), optional=True),
            ModifierSlot("nc", ("nc",), optional=True),
            ModifierSlot("l1ev", _L1_EVICT, optional=True),
            ModifierSlot("prefetch", ("L2::64B", "L2::128B", "L2::256B"), optional=True),
            ModifierSlot("vec", ("v2", "v4")),
            ModifierSlot("type", _LD_VEC_TYPES),
        ),
        check=_check_ld_vec,
        operands=(
            OperandSlot("d", role="dst", lanes=_vec_lanes),
            OperandSlot("addr", role="addr"),
        ),
    ),
    InstructionEntry(
        # The two 256-bit lines. ptxas: "Feature '256 bit wide load/store'
        # requires .target sm_100 or higher", and .level2::eviction_priority is
        # spelled only here.
        name="ld_vec256",
        mnemonic="ld",
        slots=(
            ModifierSlot("sem", ("weak", "volatile"), optional=True),
            ModifierSlot("space", ("global",), optional=True),
            ModifierSlot("cop", ("ca", "cg", "cs", "lu", "cv"), optional=True),
            ModifierSlot("nc", ("nc",), optional=True),
            ModifierSlot("l1ev", _L1_EVICT, optional=True),
            ModifierSlot("l2ev", _L2_EVICT, optional=True),
            ModifierSlot("prefetch", ("L2::64B", "L2::128B", "L2::256B"), optional=True),
            ModifierSlot("vec", ("v4", "v8")),
            ModifierSlot("type", _BITS32 + _BITS64),
        ),
        cert_arch="sm_100",
        check=_check_ld_vec256,
        operands=(
            OperandSlot("d", role="dst", lanes=_vec_lanes),
            OperandSlot("addr", role="addr"),
        ),
    ),
    InstructionEntry(
        name="st_vec",
        mnemonic="st",
        slots=(
            ModifierSlot("sem", ("weak", "volatile"), optional=True),
            ModifierSlot(
                "space",
                ("global", "shared", "shared::cta", "shared::cluster", "local"),
                optional=True,
            ),
            ModifierSlot("cop", ("wb", "cg", "cs", "wt"), optional=True),
            ModifierSlot("l1ev", _L1_EVICT, optional=True),
            ModifierSlot("vec", ("v2", "v4")),
            ModifierSlot("type", _LD_VEC_TYPES),
        ),
        check=_check_st_vec,
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("value", role="value", lanes=_vec_lanes),
        ),
    ),
    InstructionEntry(
        name="st_vec256",
        mnemonic="st",
        slots=(
            ModifierSlot("sem", ("weak", "volatile"), optional=True),
            ModifierSlot("space", ("global",), optional=True),
            ModifierSlot("cop", ("wb", "cg", "cs", "wt"), optional=True),
            ModifierSlot("l1ev", _L1_EVICT, optional=True),
            ModifierSlot("l2ev", _L2_EVICT, optional=True),
            ModifierSlot("vec", ("v4", "v8")),
            ModifierSlot("type", _BITS32 + _BITS64),
        ),
        cert_arch="sm_100",
        check=_check_st_vec256,
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("value", role="value", lanes=_vec_lanes),
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
    # min / max, complete per PTX ISA 9.7.1.13, 9.7.1.14 (integer),
    # 9.7.3.12, 9.7.3.13 (single/double), 9.7.4.8, 9.7.4.9 (half).
    # Each mnemonic gets two entries because the ISA gives it two operand
    # shapes: the usual `d, a, b` and the three-source `d, a, b, c` line.
    # NOT REGISTERED: nothing -- every syntax line of both families is here.
    *[
        InstructionEntry(
            name=name,
            slots=(
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("nan", ("NaN",), optional=True),
                ModifierSlot("xorsign", ("xorsign",), optional=True),
                ModifierSlot("abs", ("abs",), optional=True),
                ModifierSlot("relu", ("relu",), optional=True),
                ModifierSlot("type", (*_MINMAX_FP, *_MINMAX_INT_PLAIN, *_MINMAX_INT_RELU)),
            ),
            check=_check_minmax,
            # `.u16x2`, `{.relu}.s16x2` and `.relu.s32` need sm_90 (ISA Target
            # Notes); cert_arch is the max over the entry's variants.
            cert_arch="sm_90",
            operands=(
                OperandSlot("d", role="dst"),
                OperandSlot("a", role="value"),
                OperandSlot("b", role="value"),
            ),
        )
        for name in ("max", "min")
    ],
    *[
        InstructionEntry(
            name=f"{name}3",
            mnemonic=name,
            slots=(
                ModifierSlot("ftz", ("ftz",), optional=True),
                ModifierSlot("nan", ("NaN",), optional=True),
                ModifierSlot("abs", ("abs",), optional=True),
                ModifierSlot("type", ("f32",)),
            ),
            check=_check_minmax3,
            cert_arch="sm_100",  # "max.f32 with 3 input operands" requires sm_100
            operands=(
                OperandSlot("d", role="dst"),
                OperandSlot("a", role="value"),
                OperandSlot("b", role="value"),
                OperandSlot("c", role="value"),
            ),
        )
        for name in ("max", "min")
    ],
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
    # mapa per PTX ISA 9.7.9.6: map a shared address into another CTA of the
    # cluster. All four syntax lines differ only in how `a` is spelled at the
    # PTX level (register / variable / variable+imm); through a C helper the
    # operand is always a register, so they collapse to one entry.
    # `.type` fixes the width of BOTH d and a, so the two type tokens are two
    # entries: `.u64` maps a generic address (a pointer), `.u32` maps a 32-bit
    # shared-window address (a plain register, not bracketed).
    InstructionEntry(
        name="mapa",
        slots=(
            ModifierSlot("space", ("shared::cluster",), optional=True),
            ModifierSlot("type", ("u64",)),
        ),
        asm_volatile=False,  # a pure address computation
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("a", role="ptr"),
            OperandSlot("b", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(
        name="mapa_u32",
        mnemonic="mapa",
        slots=(
            ModifierSlot("space", ("shared::cluster",), optional=True),
            ModifierSlot("type", ("u32",)),
        ),
        asm_volatile=False,
        operands=(
            OperandSlot("d", role="dst"),
            OperandSlot("a", role="value", dtype="u32"),
            OperandSlot("b", role="value", dtype="u32"),
        ),
    ),
    # clusterlaunchcontrol.try_cancel per PTX ISA 9.7.14.15.
    # NOT REGISTERED: the `query_cancel` lines (9.7.14.16) -- they decode the
    # b128 response into a `.pred` or a `.v4.b32` group, which needs both a
    # predicate result and a variable register group.
    InstructionEntry(
        name="clusterlaunchcontrol_try_cancel",
        mnemonic="clusterlaunchcontrol",
        slots=(
            ModifierSlot("action", ("try_cancel",)),
            ModifierSlot("async_", ("async",)),
            ModifierSlot("space", ("shared::cta",), optional=True),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
            ModifierSlot("multicast", ("multicast::cluster::all",), optional=True),
            ModifierSlot("type", ("b128",)),
        ),
        # The instruction itself needs sm_100, but `.multicast::cluster::all`
        # is only on the arch-specific targets (ISA: sm_100a / sm_101a / sm_120a).
        cert_arch="sm_100a",
        operands=(
            OperandSlot("addr", role="addr", space="shared::cta"),
            OperandSlot("mbar", role="addr", space="shared::cta"),
        ),
    ),
    # ------------------------------------------------------------------
    # fence / membar per PTX ISA 9.7.14.4, griddepcontrol per 9.7.14.14.
    #
    # These name no address yet constrain everyone else's memory order, so they
    # set `orders_memory=True` -- see InstructionEntry for why `asm volatile`
    # alone is not enough. Each syntax line whose token sequence differs is its
    # own entry, all sharing the `fence` mnemonic, so the surface stays
    # `T.ptxd.fence...` and the chain narrows on the tokens themselves.
    # NOT REGISTERED: the `.sync_restrict` lines and the fabric-proxy line
    # (fixed token sequences with no users yet), and the deprecated `membar`
    # spellings, which the ISA itself marks as the old style for `fence`.
    InstructionEntry(  # fence{.sem}.scope;
        name="fence",
        slots=(
            ModifierSlot("sem", ("sc", "acq_rel", "acquire", "release"), optional=True),
            ModifierSlot("scope", ("cta", "cluster", "gpu", "sys")),
        ),
        orders_memory=True,
        operands=(),
    ),
    InstructionEntry(  # fence.mbarrier_init.release.cluster;
        name="fence_mbarrier_init",
        mnemonic="fence",
        slots=(
            ModifierSlot("op_restrict", ("mbarrier_init",)),
            ModifierSlot("sem", ("release",)),
            ModifierSlot("scope", ("cluster",)),
        ),
        orders_memory=True,
        operands=(),
    ),
    InstructionEntry(  # fence.proxy.proxykind;
        name="fence_proxy",
        mnemonic="fence",
        slots=(
            ModifierSlot("proxy", ("proxy",)),
            ModifierSlot("proxykind", ("alias", "async")),
            ModifierSlot("space", ("global", "shared::cta", "shared::cluster"), optional=True),
        ),
        check=_check_fence_proxy,
        orders_memory=True,
        operands=(),
    ),
    InstructionEntry(  # fence.proxy.tensormap::generic.release.scope;
        name="fence_proxy_tensormap_release",
        mnemonic="fence",
        slots=(
            ModifierSlot("proxy", ("proxy",)),
            ModifierSlot("proxykind", ("tensormap::generic",)),
            ModifierSlot("sem", ("release",)),
            ModifierSlot("scope", ("cta", "cluster", "gpu", "sys")),
        ),
        orders_memory=True,
        operands=(),
    ),
    InstructionEntry(  # fence.proxy.tensormap::generic.acquire.scope [addr], 128;
        name="fence_proxy_tensormap_acquire",
        mnemonic="fence",
        slots=(
            ModifierSlot("proxy", ("proxy",)),
            ModifierSlot("proxykind", ("tensormap::generic",)),
            ModifierSlot("sem", ("acquire",)),
            ModifierSlot("scope", ("cta", "cluster", "gpu", "sys")),
        ),
        orders_memory=True,
        operands=(
            OperandSlot("addr", role="addr"),
            # "The only supported value for the size operand is 128, which must
            # be a constant integer literal" -- ISA 9.7.14.4.
            OperandSlot("size", role="imm", literal="128"),
        ),
    ),
    InstructionEntry(  # griddepcontrol.action;
        name="griddepcontrol",
        slots=(ModifierSlot("action", ("launch_dependents", "wait")),),
        orders_memory=True,
        operands=(),
    ),
    # ------------------------------------------------------------------
    # bar / barrier per PTX ISA 9.7.14.1, barrier.cluster per 9.7.14.2.
    #
    #   barrier{.cta}.sync{.aligned}   a{, b};    bar{.cta}.sync   a{, b};
    #   barrier{.cta}.arrive{.aligned} a,  b;     bar{.cta}.arrive a,  b;
    #
    # The optional thread count is its own syntax line, so `sync` is two
    # entries sharing a mnemonic, told apart by arity (as `mov`'s shapes are;
    # `pred` is keyword-only, so arity is unambiguous).
    # NOT REGISTERED: the `.red.popc` / `.red.op` lines, which take a `{!}c`
    # predicate source and produce a `.pred` result.
    *[
        InstructionEntry(  # bar{.cta}.sync a;  /  barrier{.cta}.sync{.aligned} a;
            name=f"{mnem}_sync",
            mnemonic=mnem,
            slots=(
                ModifierSlot("cta", ("cta",), optional=True),
                ModifierSlot("action", ("sync",)),
            )
            + (
                (ModifierSlot("aligned", ("aligned",), optional=True),) if mnem == "barrier" else ()
            ),
            orders_memory=True,
            operands=(OperandSlot("a", role="value", dtype="u32"),),
        )
        for mnem in ("bar", "barrier")
    ],
    InstructionEntry(
        name="bar_sync_count",
        mnemonic="bar",
        slots=(
            ModifierSlot("cta", ("cta",), optional=True),
            ModifierSlot("action", ("sync",)),
        ),
        orders_memory=True,
        operands=(
            OperandSlot("a", role="value", dtype="u32"),
            OperandSlot("b", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(
        name="bar_arrive",
        mnemonic="bar",
        slots=(
            ModifierSlot("cta", ("cta",), optional=True),
            ModifierSlot("action", ("arrive",)),
        ),
        orders_memory=True,
        operands=(
            OperandSlot("a", role="value", dtype="u32"),
            OperandSlot("b", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(
        name="barrier_sync_count",
        mnemonic="barrier",
        slots=(
            ModifierSlot("cta", ("cta",), optional=True),
            ModifierSlot("action", ("sync",)),
            ModifierSlot("aligned", ("aligned",), optional=True),
        ),
        orders_memory=True,
        operands=(
            OperandSlot("a", role="value", dtype="u32"),
            OperandSlot("b", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(
        name="barrier_arrive",
        mnemonic="barrier",
        slots=(
            ModifierSlot("cta", ("cta",), optional=True),
            ModifierSlot("action", ("arrive",)),
            ModifierSlot("aligned", ("aligned",), optional=True),
        ),
        orders_memory=True,
        operands=(
            OperandSlot("a", role="value", dtype="u32"),
            OperandSlot("b", role="value", dtype="u32"),
        ),
    ),
    *[
        InstructionEntry(  # barrier.cluster.arrive{.sem}{.aligned} / .wait{.acquire}{.aligned}
            name=f"barrier_cluster_{act}",
            # Shares the `barrier` mnemonic with 9.7.14.1 so the surface reads
            # `T.ptxd.barrier.cluster.arrive(...)`; `cluster` is the token that
            # tells the two ISA sections apart.
            mnemonic="barrier",
            slots=(
                ModifierSlot("cluster", ("cluster",)),
                ModifierSlot("action", (act,)),
                ModifierSlot(
                    "sem",
                    ("release", "relaxed") if act == "arrive" else ("acquire",),
                    optional=True,
                ),
                ModifierSlot("aligned", ("aligned",), optional=True),
            ),
            orders_memory=True,
            operands=(),
        )
        for act in ("arrive", "wait")
    ],
    # ------------------------------------------------------------------
    # tcgen05 memory-allocation and synchronisation, per PTX ISA 9.7.16.
    #
    # Every one of these carries `orders_memory=True`: the legacy helpers all
    # had `::: "memory"`, and the waits/fences name no address at all.
    #
    # NOT REGISTERED:
    # - `tcgen05.shift.cta_group.down [taddr]` -- the operand is bracketed, so
    #   it needs the `addr` role, but `addr` only picks the 32-bit carrier when
    #   the state space starts with "shared". A tmem address is neither shared
    #   nor generic, and labelling it shared to get the right carrier would be
    #   a lie in the table; it needs a tmem address space first.
    # - `tcgen05.ld` / `.st` / `.mma` / `.cp`, which need register groups whose
    #   length is a function of the modifiers.
    InstructionEntry(  # tcgen05.alloc.cta_group.sync.aligned{.shared::cta}.b32 [dst], nCols;
        name="tcgen05_alloc",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("alloc",)),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("space", ("shared::cta",), optional=True),
            ModifierSlot("type", ("b32",)),
        ),
        cert_arch="sm_100a",
        orders_memory=True,
        operands=(
            OperandSlot("dst", role="addr", space="shared::cta"),
            OperandSlot("ncols", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(  # tcgen05.dealloc.cta_group.sync.aligned.b32 taddr, nCols;
        name="tcgen05_dealloc",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("dealloc",)),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("type", ("b32",)),
        ),
        cert_arch="sm_100a",
        orders_memory=True,
        operands=(
            # `taddr` is a plain register operand here, not bracketed.
            OperandSlot("taddr", role="value", dtype="u32"),
            OperandSlot("ncols", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(  # tcgen05.relinquish_alloc_permit.cta_group.sync.aligned;
        name="tcgen05_relinquish_alloc_permit",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("relinquish_alloc_permit",)),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
        ),
        cert_arch="sm_100a",
        orders_memory=True,
        operands=(),
    ),
    InstructionEntry(  # tcgen05.wait::{ld,st}.sync.aligned;
        name="tcgen05_wait",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("wait::ld", "wait::st")),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
        ),
        cert_arch="sm_100a",
        orders_memory=True,
        operands=(),
    ),
    InstructionEntry(  # tcgen05.fence::{before,after}_thread_sync;
        name="tcgen05_fence",
        mnemonic="tcgen05",
        slots=(ModifierSlot("action", ("fence::before_thread_sync", "fence::after_thread_sync")),),
        cert_arch="sm_100a",
        orders_memory=True,
        operands=(),
    ),
    # tcgen05.commit.cta_group.mbarrier::arrive::one{.shared::cluster}.b64 [mbar];
    InstructionEntry(
        name="tcgen05_commit",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("commit",)),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
            ModifierSlot("completion", ("mbarrier::arrive::one",)),
            ModifierSlot("space", ("shared::cluster",), optional=True),
            ModifierSlot("type", ("b64",)),
        ),
        cert_arch="sm_100a",
        orders_memory=True,
        operands=(OperandSlot("mbar", role="addr", space="shared::cluster"),),
    ),
    # tcgen05.commit...{.shared::cluster}.multicast::cluster.b64 [mbar], ctaMask;
    # The multicast form: `pred` is keyword-only, so the trailing mask
    # dispatches by arity against the unicast entry. The mask is `.b16`
    # (the legacy helper bound it "h").
    InstructionEntry(
        name="tcgen05_commit_multicast",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("commit",)),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
            ModifierSlot("completion", ("mbarrier::arrive::one",)),
            ModifierSlot("space", ("shared::cluster",), optional=True),
            ModifierSlot("multicast", ("multicast::cluster",)),
            ModifierSlot("type", ("b64",)),
        ),
        cert_arch="sm_100a",
        orders_memory=True,
        operands=(
            OperandSlot("mbar", role="addr", space="shared::cluster"),
            OperandSlot("mask", role="value", dtype="u16"),
        ),
    ),
    # cp.async completion tracking, per PTX ISA 9.7.9.13 / 9.7.9.15.
    #
    # The wait_group counts are caller-chosen immediates: the ISA gives N no
    # register form, so each value is its own helper, and the closed `choices`
    # set is what makes every one of them certifiable. 0..7 covers every call
    # site (pipeline depths); widen the tuple if a deeper pipeline appears.
    #
    # NOT REGISTERED: the `cp.async` ca/cg copy lines, which carry an optional
    # ignore-src operand.
    InstructionEntry(  # cp.async.mbarrier.arrive{.noinc}{.shared{::cta}}.b64 [addr];
        name="cp_async_mbarrier_arrive",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("target", ("mbarrier",)),
            ModifierSlot("action", ("arrive",)),
            ModifierSlot("noinc", ("noinc",), optional=True),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b64",)),
        ),
        operands=(OperandSlot("addr", role="addr", space="shared::cta"),),
    ),
    InstructionEntry(  # cp.async.commit_group;
        name="cp_async_commit_group",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("action", ("commit_group",)),
        ),
        operands=(),
    ),
    *[
        InstructionEntry(  # cp.async{.bulk}.wait_group{.read} N;
            name=f"cp_async{'_bulk' if bulk else ''}_wait_group",
            mnemonic="cp",
            slots=(
                ModifierSlot("api", ("async",)),
                *((ModifierSlot("kind", ("bulk",)),) if bulk else ()),
                ModifierSlot("action", ("wait_group",)),
                *((ModifierSlot("read", ("read",), optional=True),) if bulk else ()),
            ),
            orders_memory=True,
            operands=(OperandSlot("group", role="imm", choices=tuple(str(n) for n in range(8))),),
        )
        for bulk in (False, True)
    ],
    # setmaxnreg per PTX ISA 9.7.17.2: the register count is an immediate the
    # ISA bounds to [24, 256] in steps of 8 -- exactly a closed choices set.
    InstructionEntry(
        name="setmaxnreg",
        slots=(
            ModifierSlot("action", ("inc", "dec")),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("type", ("u32",)),
        ),
        cert_arch="sm_90a",
        orders_memory=True,
        operands=(
            OperandSlot("nreg", role="imm", choices=tuple(str(n) for n in range(24, 257, 8))),
        ),
    ),
    # wgmma.wait_group per PTX ISA 9.7.15.4, same caller-immediate shape.
    InstructionEntry(
        name="wgmma_wait_group",
        mnemonic="wgmma",
        slots=(
            ModifierSlot("action", ("wait_group",)),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
        ),
        cert_arch="sm_90a",
        orders_memory=True,
        operands=(OperandSlot("group", role="imm", choices=tuple(str(n) for n in range(8))),),
    ),
    InstructionEntry(  # cp.async.bulk.commit_group;
        name="cp_async_bulk_commit_group",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("action", ("commit_group",)),
        ),
        operands=(),
    ),
    # cp.async per PTX ISA 9.7.9.26.3.1: the non-bulk asynchronous copy.
    # cp-size is an integer constant the ISA closes to {4, 8, 16} (and to 16
    # alone under .cg) -- a choices immediate. The src-size arity zero-fills
    # the destination tail; it is a separate entry told apart by arity, the
    # mbarrier.arrive precedent.
    #
    # NOT REGISTERED: the `{, ignore-src}` lines. No call site ever used
    # them (the legacy intrinsics for those forms were unreachable from the
    # dispatcher), and their arity collides with the src-size lines -- the
    # operand-shape dispatch could not tell a src-size u32 from an
    # ignore-src predicate.
    *[
        InstructionEntry(
            name=f"cp_async_{cop}{'_src_size' if src_size else ''}",
            mnemonic="cp",
            slots=(
                ModifierSlot("api", ("async",)),
                ModifierSlot("cop", (cop,)),
                ModifierSlot("dst", ("shared", "shared::cta")),
                ModifierSlot("src", ("global",)),
                ModifierSlot("cache", ("L2::cache_hint",), optional=True),
                ModifierSlot("prefetch", ("L2::64B", "L2::128B", "L2::256B"), optional=True),
            ),
            cert_arch="sm_90",
            operands=(
                OperandSlot("dst_mem", role="addr", space="shared"),
                OperandSlot("src_mem", role="addr", space="global"),
                OperandSlot(
                    "cp_size", role="imm", choices=("4", "8", "16") if cop == "ca" else ("16",)
                ),
                *((OperandSlot("src_size", role="value", dtype="u32"),) if src_size else ()),
                OperandSlot(
                    "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
                ),
            ),
        )
        for cop in ("ca", "cg")
        for src_size in (False, True)
    ],
    # cp.async.bulk (non-tensor), per PTX ISA 9.7.9.26.4.1: four directions.
    #
    # NOT REGISTERED:
    # - the `.sem.scope`/`.type` lines and `.ignore_oob` with its
    #   `{, ignoreBytesLeft, ignoreBytesRight}` operands: PTX ISA 9.2 in this
    #   toolchain cannot assemble them, and no call site uses them.
    InstructionEntry(  # global -> shared::cta
        name="cp_async_bulk_g2s_cta",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("dst", ("shared::cta",)),
            ModifierSlot("src", ("global",)),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
        ),
        cert_arch="sm_90",
        operands=(
            OperandSlot("dst_mem", role="addr", space="shared::cta"),
            OperandSlot("src_mem", role="addr", space="global"),
            OperandSlot("size", role="value", dtype="u32"),
            OperandSlot("mbar", role="addr", space="shared"),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
        ),
    ),
    InstructionEntry(  # global -> shared::cluster
        name="cp_async_bulk_g2s_cluster",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("dst", ("shared::cluster",)),
            ModifierSlot("src", ("global",)),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
            ModifierSlot("multicast", ("multicast::cluster",), optional=True),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
        ),
        cert_arch="sm_90",
        operands=(
            OperandSlot("dst_mem", role="addr", space="shared::cluster"),
            OperandSlot("src_mem", role="addr", space="global"),
            OperandSlot("size", role="value", dtype="u32"),
            OperandSlot("mbar", role="addr", space="shared"),
            OperandSlot("cta_mask", role="value", dtype="u16", lanes=_tma_mask_lanes, vector=False),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
        ),
    ),
    InstructionEntry(  # shared::cta -> shared::cluster (peer-CTA push)
        name="cp_async_bulk_s2c",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("dst", ("shared::cluster",)),
            ModifierSlot("src", ("shared::cta",)),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
        ),
        cert_arch="sm_90",
        operands=(
            OperandSlot("dst_mem", role="addr", space="shared::cluster"),
            OperandSlot("src_mem", role="addr", space="shared::cta"),
            OperandSlot("size", role="value", dtype="u32"),
            OperandSlot("mbar", role="addr", space="shared"),
        ),
    ),
    InstructionEntry(  # shared::cta -> global
        name="cp_async_bulk_s2g",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("dst", ("global",)),
            ModifierSlot("src", ("shared::cta",)),
            ModifierSlot("completion", ("bulk_group",)),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
            # .cp_mask (sm_100) masks bytes within each 16-byte word; ptxas
            # requires it to ride with .L2::cache_hint, which the legacy
            # helper enforced too.
            ModifierSlot("cp_mask", ("cp_mask",), optional=True),
        ),
        cert_arch="sm_100a",
        check=_check_cp_async_bulk_s2g,
        operands=(
            OperandSlot("dst_mem", role="addr", space="global"),
            OperandSlot("src_mem", role="addr", space="shared::cta"),
            OperandSlot("size", role="value", dtype="u32"),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
            # "the 16-bit wide byteMask operand" -- the legacy helper bound it
            # "r", but that form was unreachable and ptxas rejects the 32-bit
            # register here.
            OperandSlot("byte_mask", role="value", dtype="u16", lanes=_cp_mask_lanes, vector=False),
        ),
    ),
    # cp.async.bulk.tensor (TMA), per ISA 9.7.9.26.5.2-4. The tensor address
    # is the composite `[tensorMap, tensorCoords]` -- one PTX operand holding
    # a 64-bit tensor-map pointer plus an .s32 coordinate vector -- which is
    # what OperandSlot.bracket transcribes. The coordinate count follows .dim
    # except in the gather4/scatter4 modes (fixed 5); ctaMask and cache_policy
    # are trailing operands that exist exactly when their modifier is written,
    # a zero-lanes function each.
    #
    # NOT REGISTERED:
    # - the .im2col family of load modes (im2col/im2col::w/im2col::w::128 and
    #   s2g's im2col_no_offs) with their `{, im2colInfo}` operand: no call
    #   site uses im2col, and the legacy helpers never supported it.
    # - `.tile::scatter4` under cp.reduce: absent from that syntax line.
    InstructionEntry(  # global -> shared::cluster (the TMA load every kernel uses)
        name="cp_async_bulk_tensor_g2s_cluster",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("unit", ("tensor",)),
            ModifierSlot("dim", ("1d", "2d", "3d", "4d", "5d")),
            ModifierSlot("dst", ("shared::cluster",)),
            ModifierSlot("src", ("global",)),
            # ptxas accepts both spellings of the default load mode: written
            # `.tile` or omitted entirely (the legacy g2s helpers omitted it,
            # the s2g/prefetch ones wrote it).
            ModifierSlot("load_mode", ("tile", "tile::gather4"), optional=True),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
            ModifierSlot("multicast", ("multicast::cluster",), optional=True),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2"), optional=True),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
        ),
        cert_arch="sm_100a",
        check=_check_tma_gather4,
        operands=(
            OperandSlot("dst_mem", role="addr", space="shared::cluster"),
            OperandSlot("tmap", role="addr", space="global", bracket="src"),
            OperandSlot(
                "coords", role="value", dtype="s32", lanes=_tma_coords_lanes, bracket="src"
            ),
            OperandSlot("mbar", role="addr", space="shared"),
            OperandSlot("cta_mask", role="value", dtype="u16", lanes=_tma_mask_lanes, vector=False),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
        ),
    ),
    InstructionEntry(  # global -> shared::cta (no multicast operand)
        name="cp_async_bulk_tensor_g2s_cta",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("unit", ("tensor",)),
            ModifierSlot("dim", ("1d", "2d", "3d", "4d", "5d")),
            ModifierSlot("dst", ("shared::cta",)),
            ModifierSlot("src", ("global",)),
            ModifierSlot("load_mode", ("tile", "tile::gather4"), optional=True),
            ModifierSlot("completion", ("mbarrier::complete_tx::bytes",)),
            ModifierSlot("cta_group", ("cta_group::1", "cta_group::2"), optional=True),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
        ),
        cert_arch="sm_100a",
        check=_check_tma_gather4,
        operands=(
            OperandSlot("dst_mem", role="addr", space="shared::cta"),
            OperandSlot("tmap", role="addr", space="global", bracket="src"),
            OperandSlot(
                "coords", role="value", dtype="s32", lanes=_tma_coords_lanes, bracket="src"
            ),
            OperandSlot("mbar", role="addr", space="shared"),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
        ),
    ),
    InstructionEntry(  # shared::cta -> global
        name="cp_async_bulk_tensor_s2g",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("unit", ("tensor",)),
            ModifierSlot("dim", ("1d", "2d", "3d", "4d", "5d")),
            ModifierSlot("dst", ("global",)),
            ModifierSlot("src", ("shared::cta",)),
            ModifierSlot("load_mode", ("tile", "tile::scatter4"), optional=True),
            ModifierSlot("completion", ("bulk_group",)),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
        ),
        cert_arch="sm_100a",
        check=_check_tma_gather4,
        operands=(
            OperandSlot("tmap", role="addr", space="global", bracket="dst"),
            OperandSlot(
                "coords", role="value", dtype="s32", lanes=_tma_coords_lanes, bracket="dst"
            ),
            OperandSlot("src_mem", role="addr", space="shared::cta"),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
        ),
    ),
    InstructionEntry(  # global -> L2 prefetch
        name="cp_async_bulk_tensor_prefetch",
        mnemonic="cp",
        slots=(
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("action", ("prefetch",)),
            ModifierSlot("unit", ("tensor",)),
            ModifierSlot("dim", ("1d", "2d", "3d", "4d", "5d")),
            ModifierSlot("level", ("L2",)),
            ModifierSlot("src", ("global",)),
            ModifierSlot("load_mode", ("tile", "tile::gather4"), optional=True),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
        ),
        cert_arch="sm_100a",
        check=_check_tma_gather4,
        operands=(
            OperandSlot("tmap", role="addr", space="global", bracket="src"),
            OperandSlot(
                "coords", role="value", dtype="s32", lanes=_tma_coords_lanes, bracket="src"
            ),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
        ),
    ),
    InstructionEntry(  # cp.reduce.async.bulk.tensor: shared::cta -> global, in place
        name="cp_reduce_async_bulk_tensor",
        mnemonic="cp",
        slots=(
            ModifierSlot("op", ("reduce",)),
            ModifierSlot("api", ("async",)),
            ModifierSlot("kind", ("bulk",)),
            ModifierSlot("unit", ("tensor",)),
            ModifierSlot("dim", ("1d", "2d", "3d", "4d", "5d")),
            ModifierSlot("dst", ("global",)),
            ModifierSlot("src", ("shared::cta",)),
            ModifierSlot("redop", ("add", "min", "max", "inc", "dec", "and", "or", "xor")),
            ModifierSlot("load_mode", ("tile",), optional=True),
            ModifierSlot("completion", ("bulk_group",)),
            ModifierSlot("cache", ("L2::cache_hint",), optional=True),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("tmap", role="addr", space="global", bracket="dst"),
            OperandSlot(
                "coords", role="value", dtype="s32", lanes=_tma_coords_lanes, bracket="dst"
            ),
            OperandSlot("src_mem", role="addr", space="shared::cta"),
            OperandSlot(
                "cache_policy", role="value", dtype="u64", lanes=_tma_cache_lanes, vector=False
            ),
        ),
    ),
    # ------------------------------------------------------------------
    # mbarrier, per PTX ISA 9.7.14.5-9.7.14.12.
    #
    # The arrive lines come in a `state`-returning form and a sink form
    # (`_, [addr]`); the sink is what a void helper can express, so `_` is an
    # ISA-fixed immediate the way `st.bulk`'s initval is. That also leaves the
    # instruction without a destination, so `pred=` works.
    #
    # NOT REGISTERED:
    # - the `state, [addr]` forms: the destination is the barrier's pre-arrival
    #   state, which nothing reads today. (`arrive.noComplete` has no sink form
    #   at all, so it is registered below with a real state result.)
    # - the non-parity try_wait / test_wait lines (their `state` operand is the
    #   arrive-returned token nothing reads today), the `.phase_type` qualifiers
    #   (no call site), and try_wait's timeHint-less arity (every caller passes
    #   the tick budget).
    #
    # The addr slot takes no fixed space: with `.space` omitted the ISA means
    # a generic address, which is 64-bit on sm_90+, so pinning the operand to
    # shared would bind a 32-bit register there and ptxas rejects the 32-bit
    # ABI. Letting `operand_space` read the modifier picks the right carrier
    # for each variant, exactly as ld/st do.
    InstructionEntry(  # mbarrier.init{.shared{::cta}}.b64 [addr], count;
        name="mbarrier_init",
        mnemonic="mbarrier",
        slots=(
            ModifierSlot("action", ("init",)),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b64",)),
        ),
        operands=(
            OperandSlot("addr", role="addr"),
            OperandSlot("count", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(  # mbarrier.inval{.shared{::cta}}.b64 [addr];
        name="mbarrier_inval",
        mnemonic="mbarrier",
        slots=(
            ModifierSlot("action", ("inval",)),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b64",)),
        ),
        operands=(OperandSlot("addr", role="addr"),),
    ),
    InstructionEntry(  # mbarrier.arrive{.sem.scope}{.space}.b64 _, [addr];
        # The `{, count}` optionality is one ISA line; here it is two entries
        # told apart by arity, and the ISA defines the omitted count as 1.
        name="mbarrier_arrive_nocount",
        mnemonic="mbarrier",
        slots=(
            ModifierSlot("action", ("arrive",)),
            ModifierSlot("sem", ("release", "relaxed"), optional=True),
            ModifierSlot("scope", ("cta", "cluster"), optional=True),
            ModifierSlot("space", ("shared", "shared::cta", "shared::cluster"), optional=True),
            ModifierSlot("type", ("b64",)),
        ),
        check=_check_mbarrier_sem_scope,
        operands=(
            OperandSlot("state", role="imm", literal="_"),
            OperandSlot("addr", role="addr"),
        ),
    ),
    InstructionEntry(  # mbarrier.arrive{.sem.scope}{.space}.b64 _, [addr], count;
        name="mbarrier_arrive",
        mnemonic="mbarrier",
        slots=(
            ModifierSlot("action", ("arrive",)),
            ModifierSlot("sem", ("release", "relaxed"), optional=True),
            ModifierSlot("scope", ("cta", "cluster"), optional=True),
            ModifierSlot("space", ("shared", "shared::cta", "shared::cluster"), optional=True),
            ModifierSlot("type", ("b64",)),
        ),
        check=_check_mbarrier_sem_scope,
        operands=(
            OperandSlot("state", role="imm", literal="_"),
            OperandSlot("addr", role="addr"),
            OperandSlot("count", role="value", dtype="u32"),
        ),
    ),
    InstructionEntry(  # mbarrier.arrive.expect_tx{.sem.scope}{.space}.b64 _, [addr], txCount;
        name="mbarrier_arrive_expect_tx",
        mnemonic="mbarrier",
        slots=(
            ModifierSlot("action", ("arrive",)),
            ModifierSlot("expect_tx", ("expect_tx",)),
            ModifierSlot("sem", ("release", "relaxed"), optional=True),
            ModifierSlot("scope", ("cta", "cluster"), optional=True),
            ModifierSlot("space", ("shared", "shared::cta", "shared::cluster"), optional=True),
            ModifierSlot("type", ("b64",)),
        ),
        check=_check_mbarrier_sem_scope,
        operands=(
            OperandSlot("state", role="imm", literal="_"),
            OperandSlot("addr", role="addr"),
            OperandSlot("tx_count", role="value", dtype="u32"),
        ),
    ),
    # mbarrier.arrive.noComplete{.release.cta}{.shared{::cta}}.b64 state, [addr], count;
    InstructionEntry(
        # This line has no sink form, so `state` is a real register result.
        # It rides role="acc" rather than dst: "+" keeps the old value live
        # under a false predicate, which is what lets `pred=` remain legal on
        # an instruction that writes a register. Its qualifier pair is fixed
        # (.release.cta only) and the space domain has no ::cluster.
        name="mbarrier_arrive_no_complete",
        mnemonic="mbarrier",
        slots=(
            ModifierSlot("action", ("arrive",)),
            ModifierSlot("nocomplete", ("noComplete",)),
            ModifierSlot("sem", ("release",), optional=True),
            ModifierSlot("scope", ("cta",), optional=True),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b64",)),
        ),
        check=_check_mbarrier_sem_scope,
        operands=(
            # Pinned u64, not b64: the bit-type dtype axis would offer an f64
            # carrier, and ptxas rejects an .f64 register as the state operand
            # ("Arguments mismatch for instruction 'mbarrier.arrive'").
            OperandSlot("state", role="acc", dtype="u64"),
            OperandSlot("addr", role="addr"),
            OperandSlot("count", role="value", dtype="u32"),
        ),
    ),
    # mbarrier.test_wait.parity{.sem.scope}{.ss}.b64 waitComplete, [addr], phaseParity;
    # mbarrier.try_wait.parity{.sem.scope}{.ss}.b64 waitComplete, [addr], phaseParity, timeHint;
    #
    # waitComplete is a `.pred` result -- role="pred_dst", the in-block selp
    # materialization. try_wait is registered in its timeHint arity only (the
    # hint is a nanosecond budget the callers always pass).
    *[
        InstructionEntry(
            name=f"mbarrier_{act}_parity",
            mnemonic="mbarrier",
            slots=(
                ModifierSlot("action", (act,)),
                ModifierSlot("parity", ("parity",)),
                ModifierSlot("sem", ("acquire", "relaxed"), optional=True),
                ModifierSlot("scope", ("cta", "cluster"), optional=True),
                ModifierSlot("space", ("shared", "shared::cta"), optional=True),
                ModifierSlot("type", ("b64",)),
            ),
            check=_check_mbarrier_sem_scope,
            operands=(
                OperandSlot("wait_complete", role="pred_dst"),
                OperandSlot("addr", role="addr"),
                OperandSlot("phase", role="value", dtype="u32"),
                *(
                    (OperandSlot("time_hint", role="value", dtype="u32"),)
                    if act == "try_wait"
                    else ()
                ),
            ),
        )
        for act in ("test_wait", "try_wait")
    ],
    *[
        InstructionEntry(  # mbarrier.{expect_tx,complete_tx}{.sem.scope}{.space}.b64 [addr], tx;
            name=f"mbarrier_{act}",
            mnemonic="mbarrier",
            slots=(
                ModifierSlot("action", (act,)),
                ModifierSlot("sem", ("relaxed",), optional=True),
                ModifierSlot("scope", ("cta", "cluster"), optional=True),
                ModifierSlot("space", ("shared", "shared::cta", "shared::cluster"), optional=True),
                ModifierSlot("type", ("b64",)),
            ),
            check=_check_mbarrier_sem_scope,
            operands=(
                OperandSlot("addr", role="addr"),
                OperandSlot("tx_count", role="value", dtype="u32"),
            ),
        )
        for act in ("expect_tx", "complete_tx")
    ],
    # wgmma group synchronisation, per PTX ISA 9.7.15.4. (wait_group's textual
    # group count is the `choices` immediate registered above; the mma_async
    # lines are the acc-role entries further down.)
    *[
        InstructionEntry(
            name=f"wgmma_{act}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", (act,)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
            ),
            cert_arch="sm_90a",
            orders_memory=True,
            operands=(),
        )
        for act in ("fence", "commit_group")
    ],
    # ldmatrix per PTX ISA 9.7.15.5.15 -- warp-level matrix load. Three syntax
    # lines; the destination is a brace-enclosed vector of 1/2/4 32-bit
    # registers "as per the value of .num", which is what a callable `lanes`
    # transcribes. The first line's two shapes split into two entries because
    # their target floors differ (m8n8 is sm_75+; m16n16/.b8 are sm_100a) and
    # `cert_arch` is per entry.
    #
    #   ldmatrix.sync.aligned.shape.num{.trans}{.ss}.type            r, [p];
    #   ldmatrix.sync.aligned.m8n16.num{.ss}.dst_fmt.src_fmt        r, [p];
    #   ldmatrix.sync.aligned.m16n16.num.trans{.ss}.dst_fmt.src_fmt r, [p];
    #
    # NOT REGISTERED: nothing -- every syntax line of the family is here.
    InstructionEntry(  # line 1, .m8n8.b16: "Each matrix element holds 16-bit data"
        name="ldmatrix",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m8n8",)),
            ModifierSlot("num", ("x1", "x2", "x4")),
            ModifierSlot("trans", ("trans",), optional=True),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b16",)),
        ),
        operands=(
            OperandSlot("r", role="dst", dtype="b32", lanes=_ldmatrix_lanes),
            OperandSlot("p", role="addr"),
        ),
    ),
    InstructionEntry(  # line 1, .m16n16.b8: "only .x1 and .x2 are valid"
        name="ldmatrix_m16n16_b8",
        mnemonic="ldmatrix",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m16n16",)),
            ModifierSlot("num", ("x1", "x2")),
            # The syntax line spells {.trans} optional, but ptxas requires it
            # whenever the shape is .m16n16 ("Modifier '.trans' require for
            # instruction ldmatrix with shape '.m16n16'") -- the 16x16 8-bit
            # load only exists in the transposed layout.
            ModifierSlot("trans", ("trans",)),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b8",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("r", role="dst", dtype="b32", lanes=_ldmatrix_lanes),
            OperandSlot("p", role="addr"),
        ),
    ),
    InstructionEntry(  # lines 2+3: the 6/4-bit decompression loads
        name="ldmatrix_b8fmt",
        mnemonic="ldmatrix",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m8n16", "m16n16")),
            ModifierSlot("num", ("x1", "x2", "x4")),
            ModifierSlot("trans", ("trans",), optional=True),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("dst_fmt", ("b8x16",)),
            ModifierSlot("src_fmt", ("b6x16_p32", "b4x16_p64")),
        ),
        cert_arch="sm_100a",
        check=_check_ldmatrix_b8fmt,
        operands=(
            OperandSlot("r", role="dst", dtype="b32", lanes=_ldmatrix_lanes),
            OperandSlot("p", role="addr"),
        ),
    ),
    # stmatrix per PTX ISA 9.7.15.5.16 -- the store mirror of ldmatrix. One
    # syntax line; the two shapes split into two entries because their type and
    # target floors differ.
    #
    #   stmatrix.sync.aligned.shape.num{.trans}{.ss}.type [p], r;
    #
    # Note the operand order is the reverse of ldmatrix's: the address comes
    # first, the register group second.
    #
    # NOT REGISTERED: nothing -- every syntax line of the family is here.
    InstructionEntry(  # .m8n8.b16
        name="stmatrix",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m8n8",)),
            ModifierSlot("num", ("x1", "x2", "x4")),
            ModifierSlot("trans", ("trans",), optional=True),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b16",)),
        ),
        cert_arch="sm_90",  # ISA: "Requires sm_90 or higher."
        operands=(
            OperandSlot("p", role="addr"),
            OperandSlot("r", role="value", dtype="b32", lanes=_matrix_num_lanes),
        ),
    ),
    InstructionEntry(  # .m16n8.b8: ".m16n8 shape is valid only for .b8 type"
        name="stmatrix_m16n8_b8",
        mnemonic="stmatrix",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m16n8",)),
            ModifierSlot("num", ("x1", "x2", "x4")),
            # ISA: "for 16x8 matrices, .trans is mandatory".
            ModifierSlot("trans", ("trans",)),
            ModifierSlot("space", ("shared", "shared::cta"), optional=True),
            ModifierSlot("type", ("b8",)),
        ),
        cert_arch="sm_100a",
        operands=(
            OperandSlot("p", role="addr"),
            OperandSlot("r", role="value", dtype="b32", lanes=_matrix_num_lanes),
        ),
    ),
    # tcgen05.ld / .st per PTX ISA 9.7.17.8.3 / 9.7.17.8.4. The register vector
    # length is `.num` scaled by the shape width, which is what a callable
    # `lanes` transcribes; `taddr` is a tmem address, a packed 32-bit
    # (row << 16 | col) value rather than a pointer.
    #
    #   tcgen05.ld.sync.aligned.shape.num{.pack::16b}.b32   r, [taddr];
    #   tcgen05.st.sync.aligned.shape.num{.unpack::16b}.b32 [taddr], r;
    #
    # Note the operand orders mirror each other.
    #
    # NOT REGISTERED:
    # - The `.16x32bx2` shape, whose extra `immHalfSplitoff` operand is an
    #   instruction-text immediate the ISA gives no value domain for -- the
    #   `choices` mechanism needs a closed set, and no call site uses it.
    # - `tcgen05.ld.red`, which is sm_101a-only (not sm_100a, so it cannot be
    #   certified here) and has no call sites.
    InstructionEntry(
        name="tcgen05_ld",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("ld",)),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("16x64b", "16x128b", "16x256b", "32x32b")),
            ModifierSlot("num", ("x1", "x2", "x4", "x8", "x16", "x32", "x64", "x128")),
            ModifierSlot("pack", ("pack::16b",), optional=True),
            ModifierSlot("type", ("b32",)),
        ),
        cert_arch="sm_100a",
        check=_check_tcgen05_ldst,
        operands=(
            OperandSlot("r", role="dst", dtype="b32", lanes=_tcgen05_ldst_lanes),
            OperandSlot("taddr", role="addr", space="tmem"),
        ),
    ),
    InstructionEntry(
        name="tcgen05_st",
        mnemonic="tcgen05",
        slots=(
            ModifierSlot("action", ("st",)),
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("16x64b", "16x128b", "16x256b", "32x32b")),
            ModifierSlot("num", ("x1", "x2", "x4", "x8", "x16", "x32", "x64", "x128")),
            ModifierSlot("unpack", ("unpack::16b",), optional=True),
            ModifierSlot("type", ("b32",)),
        ),
        cert_arch="sm_100a",
        check=_check_tcgen05_ldst,
        operands=(
            OperandSlot("taddr", role="addr", space="tmem"),
            OperandSlot("r", role="value", dtype="b32", lanes=_tcgen05_ldst_lanes),
        ),
    ),
    # tcgen05.mma per PTX ISA 9.7.17.10.9.1 (sm_100a): D = A*B + D where D
    # lives in Tensor Memory (an address, not registers -- so the instruction
    # has no dst and @p stays available). Two entries per family split on the
    # A operand's home: a 64-bit shared-memory descriptor ("ss") or a tmem
    # address ("ts"), the same split the syntax lines draw. enable-input-d is
    # a runtime .pred argument -- role="pred_src", the in-block setp
    # conversion -- and disable-output-lane is a register vector whose length
    # follows .cta_group (4 or 8).
    #
    # NOT REGISTERED:
    # - `{, scale-input-d}`: an instruction-text immediate in [0, 15], no
    #   call site uses it.
    # - `tcgen05.mma.sp` / `.ws.sp` (sparse metadata operand), the
    #   .collector::/.ashift qualifiers, and the i8 convolution lines that
    #   only differ by those qualifiers: no call sites.
    # - .ws without the zero-column-mask-desc operand: every caller passes
    #   the mask (as literal zero).
    # - block_scale's .block16/.block32 vector sizes and its
    #   scale_vec-omitted spelling: the library always writes .scale_vec::NX.
    *[
        InstructionEntry(
            name=f"tcgen05_mma_{form}",
            mnemonic="tcgen05",
            slots=(
                ModifierSlot("action", ("mma",)),
                ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
                ModifierSlot("kind", ("kind::f16", "kind::tf32", "kind::f8f6f4", "kind::i8")),
            ),
            cert_arch="sm_100a",
            operands=(
                OperandSlot("d_tmem", role="addr", space="tmem"),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a_tmem", role="addr", space="tmem"),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("idesc", role="value", dtype="u32"),
                OperandSlot(
                    "disable_output_lane",
                    role="value",
                    dtype="u32",
                    lanes=_tcgen05_mma_mask_lanes,
                ),
                OperandSlot("enable_input_d", role="pred_src"),
            ),
        )
        for form in ("ss", "ts")
    ],
    *[
        InstructionEntry(  # weight-stationary: no mask vector, a zero-column desc
            name=f"tcgen05_mma_ws_{form}",
            mnemonic="tcgen05",
            slots=(
                ModifierSlot("action", ("mma",)),
                ModifierSlot("ws", ("ws",)),
                ModifierSlot("cta_group", ("cta_group::1",)),
                ModifierSlot("kind", ("kind::f16", "kind::tf32", "kind::f8f6f4", "kind::i8")),
            ),
            cert_arch="sm_100a",
            operands=(
                OperandSlot("d_tmem", role="addr", space="tmem"),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a_tmem", role="addr", space="tmem"),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("idesc", role="value", dtype="u32"),
                OperandSlot("enable_input_d", role="pred_src"),
                OperandSlot("zero_col_mask", role="value", dtype="u64"),
            ),
        )
        for form in ("ss", "ts")
    ],
    *[
        InstructionEntry(  # block-scaled: A/B scale factors live in tmem
            name=f"tcgen05_mma_block_scale_{form}",
            mnemonic="tcgen05",
            slots=(
                ModifierSlot("action", ("mma",)),
                ModifierSlot("cta_group", ("cta_group::1", "cta_group::2")),
                ModifierSlot("kind", ("kind::mxf8f6f4", "kind::mxf4", "kind::mxf4nvf4")),
                ModifierSlot("block_scale", ("block_scale",)),
                ModifierSlot("scale_vec", ("scale_vec::1X", "scale_vec::2X", "scale_vec::4X")),
            ),
            cert_arch="sm_100a",
            check=_check_tcgen05_mma_block_scale,
            operands=(
                OperandSlot("d_tmem", role="addr", space="tmem"),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a_tmem", role="addr", space="tmem"),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("idesc", role="value", dtype="u32"),
                OperandSlot("sfa_tmem", role="addr", space="tmem"),
                OperandSlot("sfb_tmem", role="addr", space="tmem"),
                OperandSlot("enable_input_d", role="pred_src"),
            ),
        )
        for form in ("ss", "ts")
    ],
    # mma per PTX ISA 9.7.15.5.14. Four operand groups (d, a, b, c), each a
    # register vector whose length follows the Matrix Fragments tables -- four
    # callable `lanes`, one per group. `d` and `c` are separate operands: the
    # ISA lists them separately and the legacy helper bound them to separate
    # "=" and "r" constraints, so no read-modify-write constraint is involved
    # even when a caller passes the same registers for both.
    #
    # NOT REGISTERED:
    # - The `.kind::`/`.block_scale` lines and the .e3m2/.e2m3/.e2m1 types,
    #   which require sm_120a -- outside the architectures this table certifies.
    # - `mma.sp`, a separate instruction with a metadata operand.
    InstructionEntry(  # half precision, and the alternate formats that share it
        name="mma",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m8n8k4", "m16n8k4", "m16n8k8", "m16n8k16", "m16n8k32")),
            ModifierSlot("alayout", ("row", "col")),
            ModifierSlot("blayout", ("row", "col")),
            ModifierSlot("dtype", ("f32",)),
            ModifierSlot("atype", ("f16", "bf16", "tf32", "e4m3", "e5m2")),
            ModifierSlot("btype", ("f16", "bf16", "tf32", "e4m3", "e5m2")),
            ModifierSlot("ctype", ("f32",)),
        ),
        check=_check_mma_fp_f32,
        # Register carriers, not element types: A and B fragments are packed
        # into .b32 registers whatever the element format, so they bind "r"
        # through a uint32; an .f32 accumulator holds one float per register
        # and binds "f". Pinning each to a single-dtype PTX type keeps the
        # dtype axis from offering combinations ptxas refuses.
        operands=(
            OperandSlot("d", role="dst", dtype="f32", lanes=_mma_lanes("d")),
            OperandSlot("a", role="value", dtype="u32", lanes=_mma_lanes("a")),
            OperandSlot("b", role="value", dtype="u32", lanes=_mma_lanes("b")),
            OperandSlot("c", role="value", dtype="f32", lanes=_mma_lanes("c")),
        ),
    ),
    InstructionEntry(  # the same lines with an .f16 accumulator
        name="mma_f16acc",
        mnemonic="mma",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot("shape", ("m8n8k4", "m16n8k8", "m16n8k16", "m16n8k32")),
            ModifierSlot("alayout", ("row", "col")),
            ModifierSlot("blayout", ("row", "col")),
            ModifierSlot("dtype", ("f16",)),
            ModifierSlot("atype", ("f16", "e4m3", "e5m2")),
            ModifierSlot("btype", ("f16", "e4m3", "e5m2")),
            ModifierSlot("ctype", ("f16",)),
        ),
        check=_check_mma_fp_f16,
        operands=(
            OperandSlot("d", role="dst", dtype="u32", lanes=_mma_lanes("d")),
            OperandSlot("a", role="value", dtype="u32", lanes=_mma_lanes("a")),
            OperandSlot("b", role="value", dtype="u32", lanes=_mma_lanes("b")),
            OperandSlot("c", role="value", dtype="u32", lanes=_mma_lanes("c")),
        ),
    ),
    InstructionEntry(  # integer, sub-byte and single-bit lines
        name="mma_int",
        mnemonic="mma",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            ModifierSlot(
                "shape",
                (
                    "m8n8k16",
                    "m16n8k16",
                    "m16n8k32",
                    "m8n8k32",
                    "m16n8k64",
                    "m8n8k128",
                    "m16n8k128",
                    "m16n8k256",
                ),
            ),
            ModifierSlot("alayout", ("row",)),
            ModifierSlot("blayout", ("col",)),
            ModifierSlot("satfinite", ("satfinite",), optional=True),
            ModifierSlot("dtype", ("s32",)),
            ModifierSlot("atype", ("u8", "s8", "u4", "s4", "b1")),
            ModifierSlot("btype", ("u8", "s8", "u4", "s4", "b1")),
            ModifierSlot("ctype", ("s32",)),
            ModifierSlot("bitop", ("xor", "and"), optional=True),
            ModifierSlot("popc", ("popc",), optional=True),
        ),
        check=_check_mma_int,
        operands=(
            OperandSlot("d", role="dst", dtype="u32", lanes=_mma_lanes("d")),
            OperandSlot("a", role="value", dtype="u32", lanes=_mma_lanes("a")),
            OperandSlot("b", role="value", dtype="u32", lanes=_mma_lanes("b")),
            OperandSlot("c", role="value", dtype="u32", lanes=_mma_lanes("c")),
        ),
    ),
    InstructionEntry(  # double precision
        name="mma_f64",
        mnemonic="mma",
        slots=(
            ModifierSlot("sync", ("sync",)),
            ModifierSlot("aligned", ("aligned",)),
            # The ISA writes ".m8n84" in this line's shape list -- a typo. It is
            # not .m8n8k4: ptxas rejects that ("Argument vector size mismatch"),
            # so the double-precision shapes are the three m16n8 ones.
            ModifierSlot("shape", ("m16n8k4", "m16n8k8", "m16n8k16")),
            ModifierSlot("alayout", ("row",)),
            ModifierSlot("blayout", ("col",)),
            ModifierSlot("dtype", ("f64",)),
            ModifierSlot("atype", ("f64",)),
            ModifierSlot("btype", ("f64",)),
            ModifierSlot("ctype", ("f64",)),
        ),
        operands=(
            OperandSlot("d", role="dst", dtype="f64", lanes=_mma_lanes("d")),
            OperandSlot("a", role="value", dtype="f64", lanes=_mma_lanes("a")),
            OperandSlot("b", role="value", dtype="f64", lanes=_mma_lanes("b")),
            OperandSlot("c", role="value", dtype="f64", lanes=_mma_lanes("c")),
        ),
    ),
    # wgmma.mma_async per PTX ISA 9.7.16.5.2 (sm_90a). Six type groups, each
    # with an ss line (both A and B from shared memory, named by 64-bit matrix
    # descriptors) and an rs line (A from registers -- always four .b32, see
    # the fragment note above -- which also drops imm-trans-a). Like mma, one
    # entry per accumulator register type so every operand dtype is pinned.
    #
    # The accumulator is read and written in place: D = A*B + D, so `d` is
    # role="acc", the "+" constraint. The trailing arguments all live in the
    # instruction text as caller immediates:
    #   - scale-d: "the operation of the form D = A*B is issued when the input
    #     predicate argument scale-d is false". The syntax calls it a
    #     predicate, but ptxas accepts the literals 0 and 1 in that position
    #     (certified), and every call site passes a compile-time constant --
    #     so it is a choices immediate, not a runtime operand. (The legacy
    #     helper burned a setp + predicate register on it per call.)
    #   - imm-scale-a/b: "the valid values ... are -1 and 1". The legacy
    #     helper emitted 0 for "no negate", outside the ISA's domain; ptxd
    #     transcribes the documented set.
    #   - imm-trans-a/b: {0, 1}, f16/bf16 lines only (k-major inputs cannot
    #     be transposed); the rs line has no imm-trans-a.
    #
    # NOT REGISTERED: `wgmma.mma_async.sp` (sparse A, a separate instruction
    # with a metadata operand) and the `wgmma.fence`/`commit_group`/
    # `wait_group` companions (registered above).
    *[
        InstructionEntry(  # .f16 inputs, .f32 accumulator
            name=f"wgmma_f16_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k16" for n in _WGMMA_N_FULL)),
                ModifierSlot("dtype", ("f32",)),
                ModifierSlot("atype", ("f16",)),
                ModifierSlot("btype", ("f16",)),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("d", role="acc", dtype="f32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
                OperandSlot("scale_a", role="imm", choices=("-1", "1")),
                OperandSlot("scale_b", role="imm", choices=("-1", "1")),
                *(
                    (OperandSlot("trans_a", role="imm", choices=("0", "1")),)
                    if form == "ss"
                    else ()
                ),
                OperandSlot("trans_b", role="imm", choices=("0", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
    *[
        InstructionEntry(  # the same lines with an .f16 accumulator (f16x2 pairs)
            name=f"wgmma_f16_f16acc_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k16" for n in _WGMMA_N_FULL)),
                ModifierSlot("dtype", ("f16",)),
                ModifierSlot("atype", ("f16",)),
                ModifierSlot("btype", ("f16",)),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("d", role="acc", dtype="u32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
                OperandSlot("scale_a", role="imm", choices=("-1", "1")),
                OperandSlot("scale_b", role="imm", choices=("-1", "1")),
                *(
                    (OperandSlot("trans_a", role="imm", choices=("0", "1")),)
                    if form == "ss"
                    else ()
                ),
                OperandSlot("trans_b", role="imm", choices=("0", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
    *[
        InstructionEntry(  # .bf16 inputs (.f32 accumulator only)
            name=f"wgmma_bf16_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k16" for n in _WGMMA_N_FULL)),
                ModifierSlot("dtype", ("f32",)),
                ModifierSlot("atype", ("bf16",)),
                ModifierSlot("btype", ("bf16",)),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("d", role="acc", dtype="f32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
                OperandSlot("scale_a", role="imm", choices=("-1", "1")),
                OperandSlot("scale_b", role="imm", choices=("-1", "1")),
                *(
                    (OperandSlot("trans_a", role="imm", choices=("0", "1")),)
                    if form == "ss"
                    else ()
                ),
                OperandSlot("trans_b", role="imm", choices=("0", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
    *[
        InstructionEntry(  # .tf32 inputs: k8, no transpose immediates
            name=f"wgmma_tf32_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k8" for n in _WGMMA_N_FULL)),
                ModifierSlot("dtype", ("f32",)),
                ModifierSlot("atype", ("tf32",)),
                ModifierSlot("btype", ("tf32",)),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("d", role="acc", dtype="f32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
                OperandSlot("scale_a", role="imm", choices=("-1", "1")),
                OperandSlot("scale_b", role="imm", choices=("-1", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
    *[
        InstructionEntry(  # fp8 inputs, all four e4m3/e5m2 pairings, .f32 acc
            name=f"wgmma_fp8_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k32" for n in _WGMMA_N_FULL)),
                ModifierSlot("dtype", ("f32",)),
                ModifierSlot("atype", ("e4m3", "e5m2")),
                ModifierSlot("btype", ("e4m3", "e5m2")),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("d", role="acc", dtype="f32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
                OperandSlot("scale_a", role="imm", choices=("-1", "1")),
                OperandSlot("scale_b", role="imm", choices=("-1", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
    *[
        InstructionEntry(  # fp8 inputs with an .f16 accumulator
            name=f"wgmma_fp8_f16acc_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k32" for n in _WGMMA_N_FULL)),
                ModifierSlot("dtype", ("f16",)),
                ModifierSlot("atype", ("e4m3", "e5m2")),
                ModifierSlot("btype", ("e4m3", "e5m2")),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("d", role="acc", dtype="u32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
                OperandSlot("scale_a", role="imm", choices=("-1", "1")),
                OperandSlot("scale_b", role="imm", choices=("-1", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
    *[
        InstructionEntry(  # s8/u8 inputs: {.satfinite}, no scale-a/b
            name=f"wgmma_int_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k32" for n in _WGMMA_N_S8)),
                ModifierSlot("satfinite", ("satfinite",), optional=True),
                ModifierSlot("dtype", ("s32",)),
                ModifierSlot("atype", ("s8", "u8")),
                ModifierSlot("btype", ("s8", "u8")),
            ),
            cert_arch="sm_90a",
            operands=(
                # An .s32 accumulator rides a u32 carrier register, the same
                # bit-identical pinning as mma_int's d/c operands.
                OperandSlot("d", role="acc", dtype="u32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
    *[
        InstructionEntry(  # single-bit inputs: .op = {.and}, .popc accumulate
            name=f"wgmma_b1_{form}",
            mnemonic="wgmma",
            slots=(
                ModifierSlot("action", ("mma_async",)),
                ModifierSlot("sync", ("sync",)),
                ModifierSlot("aligned", ("aligned",)),
                ModifierSlot("shape", tuple(f"m64n{n}k256" for n in _WGMMA_N_B1)),
                ModifierSlot("dtype", ("s32",)),
                ModifierSlot("atype", ("b1",)),
                ModifierSlot("btype", ("b1",)),
                ModifierSlot("bitop", ("and",)),
                ModifierSlot("popc", ("popc",)),
            ),
            cert_arch="sm_90a",
            operands=(
                OperandSlot("d", role="acc", dtype="u32", lanes=_wgmma_acc_lanes),
                *(
                    (OperandSlot("a_desc", role="value", dtype="u64"),)
                    if form == "ss"
                    else (OperandSlot("a", role="value", dtype="u32", lanes=4),)
                ),
                OperandSlot("b_desc", role="value", dtype="u64"),
                OperandSlot("scale_d", role="imm", choices=("0", "1")),
            ),
        )
        for form in ("ss", "rs")
    ],
]

TABLE: dict[str, InstructionEntry] = {e.name: e for e in _ENTRIES}
