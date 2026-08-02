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
"""Tests for the table-driven PTX dialect prototype (``T.ptxd``)."""

import os
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

import tvm
from tvm.ir import Op
from tvm.script import tirx as T
from tvm.testing import env

TARGET = tvm.target.Target("cuda")

requires_nvcc = pytest.mark.skipif(shutil.which("nvcc") is None, reason="nvcc not available")


def _cuda_source(func) -> str:
    with TARGET:
        mod = tvm.compile(tvm.IRModule({"main": func}), target=TARGET, tir_pipeline="tirx")
    return mod.mod.imports[0].inspect_source("cuda")


# Architecture the certification assembles at. Families whose ISA floor is
# higher (tcgen05, clusterlaunchcontrol, several cp.async.bulk forms) MUST be
# certified at their own floor: assembling them below it makes ptxas report
# legal variants as illegal, which would then get baked into a check().
PTXD_ARCH = os.environ.get("PTXD_ARCH", "sm_90")


def _assert_ptxas_ok(src: str, rdc: bool = False, arch: str = PTXD_ARCH) -> None:
    """Assemble through ptxas (cubin) — `-ptx` alone never validates inline asm."""
    from tvm.support import nvcc

    options = ["-rdc=true"] if rdc else None
    nvcc.compile_cuda(src, target_format="cubin", arch=arch, options=options, compiler="nvcc")


def test_ptxd_registration():
    from tvm.backend.cuda.intrinsics.registry import CODEGEN_REGISTRY
    from tvm.backend.cuda.ptx_dialect.table import TABLE

    assert hasattr(T, "ptxd")
    for entry in TABLE.values():
        op = Op.get(entry.op_name)  # raises if unregistered
        assert op.get_attr("TCallEffectKind") is not None, entry.name
        assert op.get_attr("TScriptPrinterName") == f"ptxd.{entry.name}", entry.name
        assert entry.op_name in CODEGEN_REGISTRY, entry.name


def test_ptxd_prefetch_codegen():
    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "float32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        T.ptxd.prefetch.global_.L2(A.ptr_to([0]))
        A[tx] = T.float32(0)  # keep-alive store so the buffer is not elided

    src = _cuda_source(kernel)
    assert "prefetch.global.L2 [%0];" in src
    assert "tvm_builtin_ptxd_prefetch_global_L2" in src


def test_ptxd_ld_st_codegen():
    @T.prim_func
    def kernel(a_ptr: T.handle, b_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        B = T.match_buffer(b_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            val: T.uint32 = T.ptxd.ld.global_.acquire.gpu.b32(A.ptr_to([0]))
            T.ptxd.st.release.gpu.global_.b32(B.ptr_to([0]), val)
        B[tx] = B[tx]

    src = _cuda_source(kernel)
    # Modifier tokens render in table slot order (sem, scope, space, type)
    # regardless of the order they were written in the chain.
    assert "ld.acquire.gpu.global.b32 %0, [%1];" in src
    assert "st.release.gpu.global.b32 [%0], %1;" in src
    assert "tvm_builtin_ptxd_ld_acquire_gpu_global_b32" in src
    assert "tvm_builtin_ptxd_st_release_gpu_global_b32" in src


def test_ptxd_st_shared_coercion():
    @T.prim_func
    def kernel(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (1,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        smem = T.alloc_buffer((4,), "uint32", scope="shared")
        if tx == 0:
            # Shared-space slot fed a shared-scope pointer: engine must
            # auto-wrap with cvta_generic_to_shared.
            T.ptxd.st.shared__cta.b32(smem.ptr_to([0]), T.uint32(7))
        T.cuda.cta_sync()
        out[0] = smem[0]

    src = _cuda_source(kernel)
    assert "st.shared::cta.b32 [%0], %1;" in src
    assert "cvta_generic_to_shared" in src


def test_ptxd_explicit_cvta():
    @T.prim_func
    def kernel(out_ptr: T.handle):
        out = T.match_buffer(out_ptr, (1,), "uint64")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        smem = T.alloc_buffer((4,), "uint32", scope="shared")
        smem[tx % 4] = T.uint32(0)
        if tx == 0:
            out[0] = T.ptxd.cvta.to.shared.u64(smem.data)

    src = _cuda_source(kernel)
    assert "cvta.to.shared.u64 %0, %1;" in src
    assert "tvm_builtin_ptxd_cvta_to_shared_u64" in src


def test_ptxd_red_codegen():
    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (1,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            T.ptxd.red.relaxed.gpu.global_.add.u32(A.ptr_to([0]), T.uint32(1))
        A[0] = A[0]

    src = _cuda_source(kernel)
    assert "red.relaxed.gpu.global.add.u32 [%0], %1;" in src


def test_ptxd_predication_codegen():
    @T.prim_func
    def kernel(a_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        flag: T.uint32 = T.uint32(0)
        if tx == 0:
            flag = T.uint32(1)
        T.ptxd.red.relaxed.gpu.global_.add.u32(A.ptr_to([0]), T.uint32(1), pred=flag)
        A[tx] = A[tx]

    src = _cuda_source(kernel)
    assert "tvm_builtin_ptxd_red_relaxed_gpu_global_add_u32_pred" in src
    assert "setp.ne.b32 p, %2, 0; @p red.relaxed.gpu.global.add.u32 [%0], %1;" in src


def test_ptxd_string_form_matches_chain():
    chain_call = T.ptxd.ld.global_.acquire.gpu.b32
    string_call = T.ptxd["ld.acquire.gpu.global.b32"]

    def make(fn):
        @T.prim_func
        def kernel(a_ptr: T.handle, b_ptr: T.handle):
            A = T.match_buffer(a_ptr, (32,), "uint32")
            B = T.match_buffer(b_ptr, (32,), "uint32")
            T.device_entry()
            T.cta_id([1])
            tx = T.thread_id([32])
            if tx == 0:
                B[0] = fn(A.ptr_to([0]))
            B[tx] = B[tx]

        return kernel

    tvm.ir.assert_structural_equal(make(chain_call), make(string_call))


def test_ptxd_trace_time_errors():
    # Global ld fed a raw uint32 address.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="shared state space"):

        @T.prim_func
        def bad_global_addr(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            out[0] = T.ptxd.ld.global_.b32(T.uint32(0))

    # Bogus modifier token.
    with pytest.raises((AttributeError, tvm.error.DiagnosticError), match="not a valid modifier"):

        @T.prim_func
        def bad_modifier(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            out[0] = T.ptxd.ld.global_.bogus.b32(T.uint32(0))

    # Value dtype mismatching the type modifier.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="must have dtype"):

        @T.prim_func
        def bad_value_dtype(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            T.ptxd.st.global_.b32(A.ptr_to([0]), T.float32(1.0))

    # Missing required modifier (no type token).
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="missing required modifier"):

        @T.prim_func
        def missing_type(a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            T.ptxd.st.global_(A.ptr_to([0]), T.uint32(0))

    # Per-slot legal tokens whose combination is illegal PTX: acquire
    # requires a scope — rejected by the entry's check function.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="requires a scope"):

        @T.prim_func
        def acquire_without_scope(out: T.Buffer((1,), "uint32"), a_ptr: T.handle):
            A = T.match_buffer(a_ptr, (1,), "uint32")
            T.device_entry()
            out[0] = T.ptxd.ld.global_.acquire.b32(A.ptr_to([0]))

    # Raw uint64 generic address: the helper parameter is `const void*`,
    # so integer addresses are only accepted as uint32 in shared space.
    with pytest.raises((ValueError, tvm.error.DiagnosticError), match="must be a pointer"):

        @T.prim_func
        def bad_u64_addr(out: T.Buffer((1,), "uint32")):
            T.device_entry()
            out[0] = T.ptxd.ld.global_.b32(T.uint64(0))


def test_ptxd_parser_roundtrip():
    """script() output re-parses to a structurally equal PrimFunc."""

    @T.prim_func
    def kernel(a_ptr: T.handle, b_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        B = T.match_buffer(b_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        smem = T.alloc_buffer((4,), "uint32", scope="shared")
        if tx == 0:
            val: T.uint32 = T.ptxd.ld.global_.acquire.gpu.b32(A.ptr_to([0]))
            T.ptxd.st.shared__cta.b32(smem.ptr_to([0]), val)
            T.ptxd.red.relaxed.gpu.global_.add.u32(B.ptr_to([0]), T.uint32(1), pred=val)
            T.ptxd.prefetch.global_.L2(A.ptr_to([16]))
            T.evaluate(T.ptxd.cvta.to.shared.u64(smem.data))
        T.cuda.cta_sync()
        B[tx] = smem[tx % 4]

    reparsed = tvm.script.from_source(kernel.script())
    tvm.ir.assert_structural_equal(kernel, reparsed)


def test_ptxd_printer_form():
    @T.prim_func
    def kernel(a_ptr: T.handle, b_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        B = T.match_buffer(b_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        if tx == 0:
            B[0] = T.ptxd.ld.global_.acquire.gpu.b32(A.ptr_to([0]))
        B[tx] = B[tx]

    script = kernel.script()
    assert "T.ptxd.ld(" in script
    assert '"acquire"' in script


# ---------------------------------------------------------------------------
# Registered-instruction unit tests: pin down the engine's generated helpers
# and its trace-time coercion so the behavior is readable here, not implicit.
# ---------------------------------------------------------------------------


def test_ptxd_helper_source_golden():
    """Exact generated helper source, one per family (executable documentation)."""
    from tvm.backend.cuda.ptx_dialect.render import render_variant
    from tvm.backend.cuda.ptx_dialect.table import TABLE

    def render(family, tokens, predicated=False):
        return render_variant(TABLE[family], tokens, predicated)[2]

    assert render("prefetch", ("global", "L2")) == (
        "__forceinline__ __device__ void tvm_builtin_ptxd_prefetch_global_L2"
        "(const void* __addr) {\n"
        '  asm volatile("prefetch.global.L2 [%0];" :  : "l"(__addr) : "memory");\n'
        "}\n"
    )
    assert render("ld", ("", "acquire", "gpu", "global", "", "", "", "", "b32")) == (
        "__forceinline__ __device__ uint32_t tvm_builtin_ptxd_ld_acquire_gpu_global_b32"
        "(const void* __addr) {\n"
        "  uint32_t __ret;\n"
        '  asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(__ret) : "l"(__addr)'
        ' : "memory");\n'
        "  return __ret;\n"
        "}\n"
    )
    # Shared-space addr slot: helper takes uint32_t (post-coercion form), not void*.
    assert render("st", ("", "", "shared::cta", "b32")) == (
        "__forceinline__ __device__ void tvm_builtin_ptxd_st_shared__cta_b32"
        "(uint32_t __addr, uint32_t __value) {\n"
        '  asm volatile("st.shared::cta.b32 [%0], %1;" :  : "r"(__addr), "r"(__value)'
        ' : "memory");\n'
        "}\n"
    )
    # Pure op: plain asm, no volatile, no memory clobber.
    assert render("cvta", ("to", "shared", "u64")) == (
        "__forceinline__ __device__ uint64_t tvm_builtin_ptxd_cvta_to_shared_u64"
        "(const void* __ptr) {\n"
        "  uint64_t __ret;\n"
        '  asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(__ret) : "l"(__ptr));\n'
        "  return __ret;\n"
        "}\n"
    )
    # Predicated twin (framework-level @p): extra uint32 pred param,
    # setp + @p guard wrapping the same instruction.
    assert render("red", ("relaxed", "gpu", "global", "add", "u32"), predicated=True) == (
        "__forceinline__ __device__ void tvm_builtin_ptxd_red_relaxed_gpu_global_add_u32_pred"
        "(const void* __addr, uint32_t __value, uint32_t __pred) {\n"
        '  asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0; '
        '@p red.relaxed.gpu.global.add.u32 [%0], %1; }"'
        ' :  : "l"(__addr), "r"(__value), "r"(__pred) : "memory");\n'
        "}\n"
    )
    # Mixed-space operands: per-operand space/dtype pick each carrier —
    # shared addrs are uint32, the global addr is a pointer.
    cp_tokens = ("async", "bulk", "shared::cta", "global", "mbarrier::complete_tx::bytes")
    assert render("cp", cp_tokens) == (
        "__forceinline__ __device__ void tvm_builtin_ptxd_cp_async_bulk_shared__cta_global"
        "_mbarrier__complete_tx__bytes"
        "(uint32_t __dst, const void* __src, uint32_t __size, uint32_t __mbar) {\n"
        '  asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes '
        '[%0], [%1], %2, [%3];" :  : "r"(__dst), "l"(__src), "r"(__size), "r"(__mbar)'
        ' : "memory");\n'
        "}\n"
    )


def test_ptxd_coercion_ir_forms():
    """The trace-time addr coercion, written down as IR-level assertions."""
    from tvm.ir.type import PointerType, PrimType

    shared_ptr = tvm.tirx.Var("p", PointerType(PrimType("uint32"), "shared"))
    global_ptr = tvm.tirx.Var("g", PointerType(PrimType("uint32"), "global"))
    raw_u32 = tvm.tirx.Var("a", "uint32")
    val = tvm.tirx.Var("v", "uint32")

    # Shared slot + shared pointer: engine wraps with an explicit cvta call node.
    call = T.ptxd.st.shared__cta.b32(shared_ptr, val)
    addr = call.args[0]
    assert addr.op.name == "tirx.cuda.cvta_generic_to_shared"
    assert addr.args[0].same_as(shared_ptr)

    # Shared slot + raw uint32: passthrough, no conversion inserted.
    call = T.ptxd.st.shared__cta.b32(raw_u32, val)
    assert call.args[0].same_as(raw_u32)

    # Global slot + global pointer: passthrough.
    call = T.ptxd.st.release.gpu.global_.b32(global_ptr, val)
    assert call.args[0].same_as(global_ptr)

    # Modifiers ride as trailing positional string args in slot order.
    assert [str(a).strip('"') for a in call.args[2:]] == ["release", "gpu", "global", "b32"]

    # Scope mismatches are trace-time errors.
    with pytest.raises(ValueError, match="shared-space address"):
        T.ptxd.st.shared__cta.b32(global_ptr, val)
    with pytest.raises(ValueError, match="shared-scope pointer"):
        T.ptxd.st.release.gpu.global_.b32(shared_ptr, val)

    # Predication: pred rides after the operands (codegen derives the
    # predicated form from the arg count).
    flag = tvm.tirx.Var("f", "uint32")
    call = T.ptxd.st.release.gpu.global_.b32(global_ptr, val, pred=flag)
    assert call.args[2].same_as(flag)
    assert len(call.args) == 2 + 1 + 4  # operands + pred + slot tokens
    # @p on a value-returning instruction is rejected.
    with pytest.raises(ValueError, match="only supported on void"):
        T.ptxd.ld.global_.b32(global_ptr, pred=flag)


_ASM_RE = re.compile(r'asm(?: volatile)?\("(.*?)"\s*:', re.S)
_PRED_RE = re.compile(r"^\{ \.reg \.pred p; setp\.ne\.b32 p, %\d+, 0; @p (?P<instr>[^;]+;) \}$")


def _sole_instruction(asm_text):
    """The single PTX statement in ``asm_text``, or None if it is not exactly one.

    The sanctioned ``@p`` wrapper is unwrapped first: it guards one instruction
    rather than adding one.
    """
    m = _PRED_RE.match(asm_text)
    body = m.group("instr") if m else asm_text
    if body.count(";") != 1 or not body.endswith(";"):
        return None
    return body


def test_ptxd_single_instruction_invariant():
    """Every ptxd variant emits exactly ONE native PTX instruction.

    This is the dialect's defining constraint, enforced mechanically so it
    cannot erode as the table grows: the only multi-statement form allowed is
    the framework-level ``@p`` wrapper, which guards a single instruction
    rather than adding one. cvta coercion is a separate IR node and must never
    appear inside a helper body.
    """
    from tvm.backend.cuda.ptx_dialect.render import render_variant
    from tvm.backend.cuda.ptx_dialect.table import TABLE, variants

    asm_re, sole_instruction = _ASM_RE, _sole_instruction
    checked = 0
    for entry in TABLE.values():
        for tokens in variants(entry):
            for predicated in (False, True) if entry.returns is None else (False,):
                opcode, _, source = render_variant(entry, tokens, predicated)
                asm_blocks = asm_re.findall(source)
                assert len(asm_blocks) == 1, f"{opcode}: {len(asm_blocks)} asm blocks, expected 1"
                instr = sole_instruction(asm_blocks[0])
                assert instr is not None, f"{opcode}: not a single statement: {asm_blocks[0]!r}"
                assert instr.startswith(opcode + " ") or instr == opcode + ";", (
                    f"{opcode}: emitted instruction does not match the opcode: {instr!r}"
                )
                assert "cvta" not in source or entry.name == "cvta", (
                    f"{opcode}: cvta must be a separate IR node, never inside a helper"
                )
                checked += 1
    assert checked > 0


def test_ptxd_single_instruction_invariant_detects_violations():
    """Falsify the probe: it must reject every shape the invariant forbids.

    A guard that has never been shown to fail is worth nothing, so exercise
    the real helper the invariant test uses.
    """
    forbidden = {
        # two chained instructions
        "bundle": 'asm volatile("mov.u32 %0, 1; add.u32 %0, %0, 2;" : "=r"(x));',
        # spin loop with a label and a branch (the mbarrier.try_wait shape)
        "spin": 'asm volatile("{ LAB: mbarrier.try_wait.b64 p, [%0]; @!p bra LAB; }" :: "r"(a));',
        # prologue + instruction (the cvt e2m1x2 shape)
        "prologue": 'asm volatile("{ .reg .b8 t; cvt.u8.u16 t, %1; cvt.f32 %0, t; }" : "=r"(d));',
    }
    for shape, src in forbidden.items():
        assert _sole_instruction(_ASM_RE.findall(src)[0]) is None, f"{shape} slipped through"

    # The sanctioned @p wrapper is NOT a violation — it guards one instruction.
    guarded = "{ .reg .pred p; setp.ne.b32 p, %2, 0; @p red.relaxed.gpu.global.add.u32 [%0], %1; }"
    assert _sole_instruction(guarded) == "red.relaxed.gpu.global.add.u32 [%0], %1;"


def test_ptxd_all_variants_render_unique():
    """Every legal variant renders; helper names (incl. @p twins) are unique."""
    from tvm.backend.cuda.ptx_dialect.render import render_variant
    from tvm.backend.cuda.ptx_dialect.table import TABLE, variants

    names = set()
    total = 0
    for entry in TABLE.values():
        combos = list(variants(entry))
        assert combos, f"{entry.name}: check() filtered out every combination"
        for tokens in combos:
            total += 1
            opcode, helper, source = render_variant(entry, tokens)
            assert helper not in names, f"helper name collision: {helper}"
            names.add(helper)
            assert f'"{opcode} ' in source or f'"{opcode};"' in source
            if entry.returns is None:  # framework-level @p twin
                _, pred_helper, pred_source = render_variant(entry, tokens, predicated=True)
                assert pred_helper not in names
                names.add(pred_helper)
                assert f"@p {opcode} " in pred_source
    assert total == 9663  # update when the table grows


def test_ptxd_stub_up_to_date():
    """The checked-in tvm.script.tirx stub must match the generator."""
    from tvm.backend.cuda.ptx_dialect import gen_stubs

    stub = Path(tvm.__file__).parent / "script" / "tirx.pyi"
    assert stub.read_text(encoding="utf-8") == gen_stubs.generate(), (
        "python/tvm/script/tirx.pyi is stale; regenerate with "
        "`python -m tvm.backend.cuda.ptx_dialect.gen_stubs -o python/tvm/script/tirx.pyi`"
    )


@requires_nvcc
def test_ptxas_gate_rejects_invalid():
    """Honesty check: the gate path must actually reject bad instructions."""
    bogus = '__device__ void f(unsigned x) { asm volatile("totally.bogus.instr %0;" : : "r"(x)); }'
    with pytest.raises(Exception, match="bogus|error"):
        _assert_ptxas_ok(bogus, rdc=True)


@requires_nvcc
def test_ptxd_sampled_helpers_assemble():
    """Fast tier: a seeded sample of every family's variants assembles."""
    import random

    from tvm.backend.cuda.ptx_dialect.render import render_variant
    from tvm.backend.cuda.ptx_dialect.table import TABLE, variants

    rng = random.Random(20260802)
    chunks = ["#include <cstdint>"]
    for entry in TABLE.values():
        combos = variants(entry)
        for i in rng.sample(range(len(combos)), min(16, len(combos))):
            for predicated in (False, True) if entry.returns is None else (False,):
                _, _, source = render_variant(entry, combos[i], predicated)
                chunks.append(source.replace("__forceinline__ ", ""))
    _assert_ptxas_ok("\n".join(chunks), rdc=True)


_CERT_SHARDS = 32


@pytest.mark.skipif(
    not os.environ.get("PTXD_CERT"),
    reason="full-table ptxas certification; run with PTXD_CERT=1 after changing the table",
)
@requires_nvcc
@pytest.mark.parametrize("shard", range(_CERT_SHARDS))
def test_ptxd_all_helpers_certify(shard):
    """Certification tier: EVERY legal variant assembles under ptxas (sm_90).

    Sharded so pytest-xdist can spread the nvcc work::

        PTXD_CERT=1 pytest -n 16 -k certify tests/python/tirx/codegen/test_ptxd_dialect.py

    Stride slicing keeps shards balanced (ld dominates the variant count).
    __forceinline__ must be stripped and -rdc used: unreferenced inline
    device functions are silently dropped before ptxas ever sees them.
    """
    from tvm.backend.cuda.ptx_dialect.render import render_variant
    from tvm.backend.cuda.ptx_dialect.table import TABLE, variants

    chunks = ["#include <cstdint>"]
    covered = 0
    index = 0
    for name in sorted(TABLE):
        entry = TABLE[name]
        for combo in variants(entry):
            if index % _CERT_SHARDS == shard:
                covered += 1
                for predicated in (False, True) if entry.returns is None else (False,):
                    _, _, source = render_variant(entry, combo, predicated)
                    chunks.append(source.replace("__forceinline__ ", ""))
            index += 1
    assert covered, "empty shard: lower _CERT_SHARDS"
    _assert_ptxas_ok("\n".join(chunks), rdc=True)


@requires_nvcc
def test_ptxd_nvcc_smoke():
    @T.prim_func
    def kernel(a_ptr: T.handle, b_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        B = T.match_buffer(b_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        smem = T.alloc_buffer((4,), "uint32", scope="shared")
        if tx == 0:
            val: T.uint32 = T.ptxd.ld.global_.acquire.gpu.b32(A.ptr_to([0]))
            T.ptxd.st.shared__cta.b32(smem.ptr_to([0]), val)
            T.ptxd.red.relaxed.gpu.global_.add.u32(B.ptr_to([0]), T.uint32(1))
            T.ptxd.prefetch.global_.L2(A.ptr_to([16]))
        T.cuda.cta_sync()
        B[tx] = smem[tx % 4]

    _assert_ptxas_ok(_cuda_source(kernel))


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="CUDA GPU not available")
def test_ptxd_ld_st_gpu_roundtrip():
    @T.prim_func
    def kernel(a_ptr: T.handle, b_ptr: T.handle):
        A = T.match_buffer(a_ptr, (32,), "uint32")
        B = T.match_buffer(b_ptr, (32,), "uint32")
        T.device_entry()
        T.cta_id([1])
        tx = T.thread_id([32])
        val: T.uint32 = T.ptxd.ld.global_.acquire.gpu.b32(A.ptr_to([tx]))
        T.ptxd.st.release.gpu.global_.b32(B.ptr_to([tx]), val)

    with TARGET:
        mod = tvm.compile(tvm.IRModule({"main": kernel}), target=TARGET, tir_pipeline="tirx")

    def run_and_check():
        dev = tvm.cuda(0)
        a_np = np.arange(32, dtype=np.uint32) + 100
        b_np = np.zeros(32, dtype=np.uint32)
        a = tvm.runtime.tensor(a_np, device=dev)
        b = tvm.runtime.tensor(b_np, device=dev)
        mod(a, b)
        np.testing.assert_array_equal(b.numpy(), a_np)

    tvm.testing.run_with_gpu_lock(run_and_check)


if __name__ == "__main__":
    tvm.testing.main()
