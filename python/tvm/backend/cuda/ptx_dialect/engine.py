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
"""Generic engine for the ``T.ptxd`` table-driven PTX dialect prototype.

One engine interprets every :class:`~.table.InstructionEntry` — there is no
per-instruction generated or hand-written code:

- :func:`register_table` registers each family as a TVM Op
  (``tirx.ptxd.<name>``) with effect/printer attrs, plus one generic codegen
  closure that renders the ``asm volatile`` helper from the table.
- :class:`PTXDNamespace` (surfaced as ``T.ptxd``) resolves attribute chains
  such as ``T.ptxd.ld.global_.acquire.gpu.b32(addr)`` against the table:
  the first token names the family, every further token fills a modifier
  slot (order-free). Python keywords are escaped with a trailing underscore
  (``global_``); ``::`` is written as a double underscore (``shared__cta``).
  The string form ``T.ptxd["st.weak.shared::cta.b32"]`` preserves exact PTX
  text.
- Modifiers travel as trailing positional string args of the traced Call
  (never ``Call.attrs`` — that would break TVMScript pretty-printing). Call
  arg layout: ``[operands..., pred?] [slot tokens ("" = omitted)]``; the
  codegen derives predication from the arg count. Destinations are ordinary
  leading operands, so a call is always a statement:
  ``T.ptxd.ld.acquire.gpu.global_.b32(val, ptr)``.
"""

from tvm.backend.cuda.intrinsics.registry import register_codegen
from tvm.backend.cuda.intrinsics.utils import parse_str
from tvm.backend.cuda.op import cuda_cvta_generic_to_shared, cuda_func_call
from tvm.ir.op import register_op_attr
from tvm.ir.type import PointerType, PrimType
from tvm.runtime import const
from tvm.tirx.expr import BufferLoad, CallEffectKind
from tvm.tirx.op import call_intrin, reinterpret

from .render import render_variant
from .table import (
    PTX_TYPES,
    InstructionEntry,
    call_slots,
    escape_token,
    mods,
    operand_space,
    operand_type,
    unescape_token,
)

# Every ptxd call is a void statement, and RemoveNoOp deletes any Evaluate()
# whose value is <= kReadState, so kOpaque is what keeps the instruction alive.
# It is also the honest answer: "do not touch my instruction" is exactly the
# contract a hand-written PTX call wants.
_EFFECT_OPAQUE = CallEffectKind.Opaque.value

# ---------------------------------------------------------------------------
# Registration (import time)
# ---------------------------------------------------------------------------


def register_table(table: dict[str, InstructionEntry]) -> None:
    """Register every table entry as a TVM Op + generic codegen."""
    for entry in table.values():
        # First attr call implicitly creates the Op registry entry. Effect
        # kind must exist before any side-effect analysis sees the op.
        register_op_attr(entry.op_name, "TCallEffectKind", _EFFECT_OPAQUE)
        # The printer name is the *surface* path a user can type, which is the
        # mnemonic (several `mov_*` entries all answer to `T.ptxd.mov`), not the
        # table key. Reparsing re-dispatches on the operand shape.
        family = entry.family
        register_op_attr(entry.op_name, "TScriptPrinterName", f"ptxd.{family}", level=20)
        register_op_attr(entry.op_name, "TIRxOpCategory", "device_intrin")
        register_op_attr(entry.op_name, "TDeviceIntrinsicNamespace", "ptxd")
        register_codegen(f"ptxd.{entry.name}")(_make_codegen(entry))


# ---------------------------------------------------------------------------
# Codegen (compile time): table -> asm volatile helper
# ---------------------------------------------------------------------------


def _make_codegen(entry: InstructionEntry):
    n_slots = len(entry.slots)
    n_operands = len(call_slots(entry))

    def codegen(*args):
        tokens = [parse_str(a) for a in args[len(args) - n_slots :]]
        rest = args[: len(args) - n_slots]  # operands, plus pred when present
        predicated = len(rest) > n_operands
        _, helper, source = render_variant(entry, tokens, predicated)
        # Every helper is void; a destination is an ordinary argument, printed
        # by the C codegen as the lvalue it binds the reference parameter to.
        return cuda_func_call(helper, *rest, source_code=source)

    return codegen


# ---------------------------------------------------------------------------
# Trace time: operand coercion + Call emission
# ---------------------------------------------------------------------------


def _coerce_operand(entry, slot, value, mod_map):
    # T.local_scalar / `x: T.float32` hand back a wrapper around the BufferLoad;
    # unwrap once here so every role sees the node itself.
    value = getattr(value, "scalar", value)
    ty = getattr(value, "ty", None)
    if slot.role == "dst":
        # A PTX destination is a register the caller declared, so the argument
        # has to be a writable lvalue: a scalar (`x: T.float32`) or a buffer
        # element. Both are BufferLoad nodes, which the C codegen prints as the
        # lvalue bound to the helper's reference parameter.
        want = PTX_TYPES[operand_type(slot, mod_map)][0]
        if not isinstance(value, BufferLoad):
            raise ValueError(
                f"{entry.name}: destination '{slot.name}' must be a writable scalar or "
                f"buffer element (declare it first, e.g. `d: T.{want}`), got "
                f"{type(value).__name__} — a T.let binding is immutable and cannot be one"
            )
        got = value.buffer.dtype
        if got != want:
            raise ValueError(
                f"{entry.name}: destination '{slot.name}' must have dtype {want} "
                f"(from .{operand_type(slot, mod_map)}), got {got}"
            )
        return value
    if slot.role == "value":
        type_token = operand_type(slot, mod_map)
        want = PTX_TYPES[type_token][0]
        if isinstance(value, int | float):
            return const(value, want)
        if isinstance(ty, PrimType) and ty.dtype == want:
            return value
        got = ty.dtype if isinstance(ty, PrimType) else type(value).__name__
        raise ValueError(
            f"{entry.name}: operand '{slot.name}' must have dtype {want} "
            f"(from .{type_token}), got {got}"
        )
    if slot.role == "ptr":
        if isinstance(ty, PointerType):
            return value
        raise ValueError(f"{entry.name}: operand '{slot.name}' must be a pointer")
    # role == "addr"
    space = operand_space(slot, mod_map)
    if space.startswith("shared"):
        if isinstance(ty, PointerType):
            # Any pointer is accepted and converted, which is what the legacy
            # helpers did. The pointer's storage_scope is not a reliable
            # discriminator here: a shared buffer's ptr_to() reports 'global',
            # so gating on it rejects correct code.
            return cuda_cvta_generic_to_shared(value)
        if isinstance(ty, PrimType) and ty.dtype == "uint32":
            return value  # trusted raw shared-window address
        raise ValueError(
            f"{entry.name}: operand '{slot.name}' must be a shared-scope pointer "
            f"or a uint32 shared address"
        )
    if isinstance(ty, PointerType):
        if ty.storage_scope.startswith("shared"):
            raise ValueError(
                f"{entry.name}: operand '{slot.name}' is a {space or 'generic'} address "
                f"but got a shared-scope pointer"
            )
        return value
    if isinstance(ty, PrimType) and ty.dtype == "uint64":
        # A 64-bit address handle, e.g. T.address_of(tensormap). The helper
        # parameter is `const void*` and PTX binds it to the same "l" register
        # either way, so make the conversion an explicit, visible IR node
        # rather than a silent type pun.
        return reinterpret("handle", value)
    if isinstance(ty, PrimType) and ty.dtype == "uint32":
        raise ValueError(f"{entry.name}: uint32 address requires shared state space")
    raise ValueError(f"{entry.name}: operand '{slot.name}' must be a pointer or uint64 handle")


def _coerce_pred(entry, pred):
    ty = getattr(pred, "ty", None)
    if isinstance(ty, PrimType) and ty.dtype in ("bool", "uint32", "int32"):
        return pred
    raise ValueError(f"{entry.name}: pred must be a bool/uint32/int32 expression")


def _emit(entry, filled, operands, pred=None):
    supplied = call_slots(entry)
    if len(operands) != len(supplied):
        names = ", ".join(
            f"{s.name}[{s.lanes}]" if s.lanes > 1 else s.name
            for s in entry.operands
            if s.role != "imm"
        )
        raise ValueError(
            f"{entry.name} expects {len(supplied)} operand(s) ({names}), got {len(operands)}"
        )
    missing = [
        slot.name for slot, tok in zip(entry.slots, filled) if tok is None and not slot.optional
    ]
    if missing:
        raise ValueError(f"{entry.name}: missing required modifier(s): {', '.join(missing)}")
    mod_map = mods(entry, filled)
    if entry.check is not None:
        error = entry.check(mod_map)
        if error:
            raise ValueError(f"{entry.name}: {error}")
    if pred is not None:
        if entry.has_dst:
            raise ValueError(
                f"{entry.name}: @p predication is only supported on instructions without a "
                f"destination (a false predicate leaves the destination unwritten, and the "
                f'"=" output constraint would discard its prior value)'
            )
        pred = _coerce_pred(entry, pred)
    coerced = [
        _coerce_operand(entry, slot, value, mod_map) for slot, value in zip(supplied, operands)
    ]
    # Call arg layout: [operands..., pred?] [slot tokens ("" = omitted)].
    return call_intrin(
        "",  # every ptxd call is a void statement; destinations are operands
        entry.op_name,
        *coerced,
        *((pred,) if pred is not None else ()),
        *mod_map.values(),
    )


# ---------------------------------------------------------------------------
# Namespace surface: T.ptxd attribute chains + string form
# ---------------------------------------------------------------------------


def _fill(entry, filled, token):
    """Assign ``token`` to the first open modifier slot listing it, or None.

    Slot membership only; whether the final combination is legal is decided
    once at call time by the entry's ``check`` function. Returning None rather
    than raising keeps the one error message in :func:`_narrow`, which is the
    only caller that knows the full candidate set.
    """
    for i, slot in enumerate(entry.slots):
        if filled[i] is None and token in slot.choices:
            return (*filled[:i], token, *filled[i + 1 :])
    return None


class _InstrChain:
    """A PTX instruction family with a partially-filled modifier tuple.

    Holds *candidate* ``(entry, filled)`` pairs rather than a single entry,
    because PTX puts some of an instruction's shape in the operand list rather
    than in the dotted modifier text: ``mov.b64 d, {lo, hi}`` and
    ``mov.b64 {lo, hi}, a`` are the same opcode with different operand shapes.
    Each modifier token narrows the candidates; the call's argument count and
    operand dtypes -- the same information the assembler resolves them by --
    select the final one.
    """

    __slots__ = ("_cands",)

    def __init__(self, cands):
        self._cands = tuple(cands)

    def __getattr__(self, name):
        if name.startswith("_"):  # keep copy/pickle/IPython dunder probes out
            raise AttributeError(name)
        return _InstrChain(_narrow(self._cands, unescape_token(name)))

    def __call__(self, *args, pred=None):
        # Also accepts the printed round-trip form: trailing modifier-token
        # strings in slot order ("" = omitted slot) and, for @p calls, the
        # predicate as the last expression operand.
        split = len(args)
        while split > 0 and isinstance(args[split - 1], str):
            split -= 1
        cands = self._cands
        for token in args[split:]:
            if token:
                cands = _narrow(cands, token)
        operands = args[:split]

        hits, errors = [], []
        for entry, filled in cands:
            ops, p = operands, pred
            if len(ops) == len(call_slots(entry)) + 1 and p is None:
                ops, p = ops[:-1], ops[-1]
            try:
                hits.append(_emit(entry, filled, ops, pred=p))
            except ValueError as err:
                # Keep the exception; a lone candidate re-raises it untouched,
                # and only the aggregate view needs entry names in front.
                errors.append((entry, err))
        if len(hits) == 1:
            return hits[0]
        if not hits:
            if len(errors) == 1:
                raise errors[0][1]
            raise ValueError(
                "no ptxd instruction matches these operands; candidates rejected it as:\n  "
                + "\n  ".join(f"{e.name}: {err}" for e, err in errors)
            )
        raise AssertionError(  # a table bug, not a user error
            f"ambiguous ptxd table: {len(hits)} entries accept the same call "
            f"({', '.join(e.name for e, _ in cands)})"
        )

    def __dir__(self):
        """Valid next tokens — drives tab completion in IPython/Jupyter."""
        return sorted(
            {
                escape_token(tok)
                for entry, filled in self._cands
                for i, slot in enumerate(entry.slots)
                if filled[i] is None
                for tok in slot.choices
            }
        )

    def __repr__(self):
        entry, filled = self._cands[0]
        mods_str = [tok for tok in filled if tok]
        return f"<T.ptxd.{'.'.join([entry.ptx_name, *mods_str])}>"


def _narrow(cands, token):
    """Keep the candidates that can still take ``token`` in an open slot."""
    out = [(e, filled) for e, f in cands if (filled := _fill(e, f, token)) is not None]
    if not out:
        entry, filled = cands[0]
        open_choices = [
            f"{slot.name}∈{{{','.join(slot.choices)}}}"
            for i, slot in enumerate(entry.slots)
            if filled[i] is None
        ]
        raise AttributeError(
            f"'{token}' is not a valid modifier for '{entry.ptx_name}'; "
            f"open slots: {'; '.join(open_choices) or '(none)'}"
        )
    return out


class PTXDNamespace:
    """``T.ptxd`` — table-driven PTX instruction namespace (prototype)."""

    def __init__(self, table=None):
        if table is None:
            from .table import TABLE as table  # pylint: disable=import-outside-toplevel
        self._table = table

    def _family(self, token):
        # The attribute name is the mnemonic with dots folded to underscores:
        # identical to the table key for every single-shape family (st.bulk ->
        # st_bulk), and the shared surface name for families whose shapes are
        # separate entries (all `mov_*` entries answer to `mov`).
        cands = [(e, (None,) * len(e.slots)) for e in self._table.values() if e.family == token]
        return _InstrChain(cands) if cands else None

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        chain = self._family(unescape_token(name))
        if chain is None:
            raise AttributeError(
                f"'{name}' is not a ptxd instruction; known families: "
                f"{', '.join(sorted(self._family_names()))}"
            )
        return chain

    def __getitem__(self, text):
        """Exact-PTX-text form, e.g. ``T.ptxd["st.weak.shared::cta.b32"]``."""
        first, _, rest = text.partition(".")
        chain = self._family(first)
        if chain is None:
            raise KeyError(
                f"'{text}' does not start with a ptxd instruction family; "
                f"known: {', '.join(sorted(self._family_names()))}"
            )
        for token in rest.split(".") if rest else []:
            try:
                chain = getattr(chain, escape_token(token))
            except AttributeError as err:
                raise KeyError(str(err)) from None
        return chain

    def _family_names(self):
        return {e.family for e in self._table.values()}

    def __dir__(self):
        """Family names — drives tab completion."""
        return sorted(self._family_names() | set(super().__dir__()))

    def __repr__(self):
        return f"<T.ptxd: {len(self._family_names())} instruction families>"
