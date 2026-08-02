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
  codegen derives predication from the arg count.
"""

from tvm.backend.cuda.intrinsics.registry import register_codegen
from tvm.backend.cuda.intrinsics.utils import parse_str
from tvm.backend.cuda.op import cuda_cvta_generic_to_shared, cuda_func_call
from tvm.ir.op import register_op_attr
from tvm.ir.type import PointerType, PrimType
from tvm.runtime import const
from tvm.tirx.op import call_intrin, reinterpret

from .render import render_variant
from .table import PTX_TYPES, InstructionEntry, escape_token, mods, unescape_token

# ---------------------------------------------------------------------------
# Registration (import time)
# ---------------------------------------------------------------------------


def register_table(table: dict[str, InstructionEntry]) -> None:
    """Register every table entry as a TVM Op + generic codegen."""
    for entry in table.values():
        # First attr call implicitly creates the Op registry entry. Effect
        # kind must exist before any side-effect analysis sees the op.
        register_op_attr(entry.op_name, "TCallEffectKind", entry.effect)
        register_op_attr(entry.op_name, "TScriptPrinterName", f"ptxd.{entry.name}", level=20)
        register_op_attr(entry.op_name, "TIRxOpCategory", "device_intrin")
        register_op_attr(entry.op_name, "TDeviceIntrinsicNamespace", "ptxd")
        register_codegen(f"ptxd.{entry.name}")(_make_codegen(entry))


# ---------------------------------------------------------------------------
# Codegen (compile time): table -> asm volatile helper
# ---------------------------------------------------------------------------


def _make_codegen(entry: InstructionEntry):
    n_slots = len(entry.slots)
    n_operands = len([s for s in entry.operands if s.role != "imm"])

    def codegen(*args):
        tokens = [parse_str(a) for a in args[len(args) - n_slots :]]
        rest = args[: len(args) - n_slots]  # operands, plus pred when present
        predicated = len(rest) > n_operands
        _, helper, source = render_variant(entry, tokens, predicated)
        if entry.returns is None:
            return cuda_func_call(helper, *rest, source_code=source)
        # return_type is the TVM dtype of the traced Call; the C return type
        # only appears inside the helper source.
        tvm_ret = PTX_TYPES[mods(entry, tokens)[entry.returns]][0]
        return cuda_func_call(helper, *rest, source_code=source, return_type=tvm_ret)

    return codegen


# ---------------------------------------------------------------------------
# Trace time: operand coercion + Call emission
# ---------------------------------------------------------------------------


def _coerce_operand(entry, slot, value, mod_map):
    ty = getattr(value, "ty", None)
    if slot.role == "value":
        type_token = slot.dtype or mod_map["type"]
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
    space = slot.space or mod_map.get("space", "")
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
    # ISA-fixed immediates are part of the instruction text, not arguments.
    supplied = [s for s in entry.operands if s.role != "imm"]
    if len(operands) != len(supplied):
        raise ValueError(
            f"{entry.name} expects {len(supplied)} operand(s) "
            f"({', '.join(s.name for s in supplied)}), got {len(operands)}"
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
        if entry.returns is not None:
            raise ValueError(
                f"{entry.name}: @p predication is only supported on void instructions "
                f"(a false predicate would leave the result undefined)"
            )
        pred = _coerce_pred(entry, pred)
    ret_dtype = PTX_TYPES[mod_map[entry.returns]][0] if entry.returns is not None else ""
    coerced = [
        _coerce_operand(entry, slot, value, mod_map) for slot, value in zip(supplied, operands)
    ]
    # Call arg layout: [operands..., pred?] [slot tokens ("" = omitted)].
    return call_intrin(
        ret_dtype,
        entry.op_name,
        *coerced,
        *((pred,) if pred is not None else ()),
        *mod_map.values(),
    )


# ---------------------------------------------------------------------------
# Namespace surface: T.ptxd attribute chains + string form
# ---------------------------------------------------------------------------


def _fill(entry, filled, token):
    """Assign ``token`` to the first open modifier slot listing it. Order-free.

    Slot membership only; whether the final combination is legal is decided
    once at call time by the entry's ``check`` function.
    """
    for i, slot in enumerate(entry.slots):
        if filled[i] is None and token in slot.choices:
            return (*filled[:i], token, *filled[i + 1 :])
    open_choices = [
        f"{slot.name}∈{{{','.join(slot.choices)}}}"
        for i, slot in enumerate(entry.slots)
        if filled[i] is None
    ]
    raise AttributeError(
        f"'{token}' is not a valid modifier for '{entry.name}'; "
        f"open slots: {'; '.join(open_choices) or '(none)'}"
    )


class _InstrChain:
    """One instruction family with a partially-filled modifier tuple."""

    __slots__ = ("_entry", "_filled")

    def __init__(self, entry, filled):
        self._entry = entry
        self._filled = filled

    def __getattr__(self, name):
        if name.startswith("_"):  # keep copy/pickle/IPython dunder probes out
            raise AttributeError(name)
        return _InstrChain(self._entry, _fill(self._entry, self._filled, unescape_token(name)))

    def __call__(self, *args, pred=None):
        # Also accepts the printed round-trip form: trailing modifier-token
        # strings in slot order ("" = omitted slot) and, for @p calls, the
        # predicate as the last expression operand.
        entry = self._entry
        split = len(args)
        while split > 0 and isinstance(args[split - 1], str):
            split -= 1
        filled = self._filled
        for token in args[split:]:
            if token:
                filled = _fill(entry, filled, token)
        operands = args[:split]
        if len(operands) == len(entry.operands) + 1 and pred is None:
            operands, pred = operands[:-1], operands[-1]
        return _emit(entry, filled, operands, pred=pred)

    def __dir__(self):
        """Valid next tokens — drives tab completion in IPython/Jupyter."""
        return sorted(
            {
                escape_token(tok)
                for i, slot in enumerate(self._entry.slots)
                if self._filled[i] is None
                for tok in slot.choices
            }
        )

    def __repr__(self):
        mods_str = [tok for tok in self._filled if tok]
        return f"<T.ptxd.{'.'.join([self._entry.name, *mods_str])}>"


class PTXDNamespace:
    """``T.ptxd`` — table-driven PTX instruction namespace (prototype)."""

    def __init__(self, table=None):
        if table is None:
            from .table import TABLE as table  # pylint: disable=import-outside-toplevel
        self._table = table

    def _family(self, token):
        entry = self._table.get(token)
        if entry is None:
            return None
        return _InstrChain(entry, (None,) * len(entry.slots))

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        chain = self._family(unescape_token(name))
        if chain is None:
            raise AttributeError(
                f"'{name}' is not a ptxd instruction; known families: "
                f"{', '.join(sorted(self._table))}"
            )
        return chain

    def __getitem__(self, text):
        """Exact-PTX-text form, e.g. ``T.ptxd["st.weak.shared::cta.b32"]``."""
        first, _, rest = text.partition(".")
        chain = self._family(first)
        if chain is None:
            raise KeyError(
                f"'{text}' does not start with a ptxd instruction family; "
                f"known: {', '.join(sorted(self._table))}"
            )
        filled = chain._filled  # pylint: disable=protected-access
        entry = chain._entry  # pylint: disable=protected-access
        for token in rest.split(".") if rest else []:
            try:
                filled = _fill(entry, filled, token)
            except AttributeError as err:
                raise KeyError(str(err)) from None
        return _InstrChain(entry, filled)

    def __dir__(self):
        """Family names — drives tab completion."""
        return sorted(set(self._table) | set(super().__dir__()))

    def __repr__(self):
        return f"<T.ptxd: {len(self._table)} instruction families>"
