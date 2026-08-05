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
"""The ptxd cvt entries: one case per registered syntax line of ISA 9.7.9.22."""

import pytest

from tvm.backend.cuda.ptx_dialect.render import render_variant
from tvm.backend.cuda.ptx_dialect.table import TABLE, renderings

# (entry name, one modifier combination, the instruction that combination emits)
_FORM_CASES = [
    ("cvt_ue8m0x2_f32", ("rz", "", "ue8m0x2", "f32"), "cvt.rz.ue8m0x2.f32"),
    ("cvt_ue8m0x2_f32", ("rp", "satfinite", "ue8m0x2", "f32"), "cvt.rp.satfinite.ue8m0x2.f32"),
    ("cvt_ue8m0x2_bf16x2", ("rz", "", "ue8m0x2", "bf16x2"), "cvt.rz.ue8m0x2.bf16x2"),
    ("cvt_bf16x2_ue8m0x2", ("rn", "bf16x2", "ue8m0x2"), "cvt.rn.bf16x2.ue8m0x2"),
    ("cvt_f8x2_f32", ("rn", "satfinite", "", "e4m3x2", "f32"), "cvt.rn.satfinite.e4m3x2.f32"),
    (
        "cvt_f8x2_f32",
        ("rn", "satfinite", "relu", "e5m2x2", "f32"),
        "cvt.rn.satfinite.relu.e5m2x2.f32",
    ),
    (
        "cvt_f8x2_fp16x2",
        ("rn", "satfinite", "", "e4m3x2", "f16x2"),
        "cvt.rn.satfinite.e4m3x2.f16x2",
    ),
    ("cvt_f16x2_f8x2", ("rn", "", "f16x2", "e4m3x2"), "cvt.rn.f16x2.e4m3x2"),
    ("cvt_f16x2_f8x2", ("rn", "relu", "f16x2", "e5m2x2"), "cvt.rn.relu.f16x2.e5m2x2"),
    ("cvt_bf16x2_f8x2", ("rn", "", "", "bf16x2", "e4m3x2"), "cvt.rn.bf16x2.e4m3x2"),
    (
        "cvt_bf16x2_f8x2",
        ("rn", "relu", "satfinite", "bf16x2", "e5m2x2"),
        "cvt.rn.relu.satfinite.bf16x2.e5m2x2",
    ),
]

_CVT_ENTRIES = {name for name in TABLE if name.startswith("cvt_")}


@pytest.mark.parametrize("entry_name,tokens,instruction", _FORM_CASES)
def test_cvt_form_renders_its_instruction(entry_name, tokens, instruction):
    entry = TABLE[entry_name]
    opcode, helper, source = render_variant(entry, tokens)
    assert opcode == instruction
    assert f'"{instruction} ' in source
    assert helper.startswith("tvm_builtin_ptxd_cvt_")
    # Every ptxd helper is void: the destination is an operand, not a return.
    assert source.startswith("__forceinline__ __device__ void ")


def test_cvt_cases_cover_every_registered_entry():
    """Every cvt entry has a case here, so a newly transcribed syntax line has
    to be given one rather than sliding in untested."""
    assert {case[0] for case in _FORM_CASES} == _CVT_ENTRIES


def test_cvt_packed_operands_bind_their_carrier():
    """A packed format names a lane layout; the register it binds is the
    carrier, not one register per lane."""
    _, _, source = render_variant(TABLE["cvt_ue8m0x2_f32"], ("rz", "", "ue8m0x2", "f32"))
    # .ue8m0x2 is two 8-bit exponents in one 16-bit register, from two floats.
    assert "uint16_t& __d" in source
    assert source.count("float __") == 2

    _, _, source = render_variant(TABLE["cvt_bf16x2_ue8m0x2"], ("rn", "bf16x2", "ue8m0x2"))
    assert "uint32_t& __d" in source
    assert "uint16_t __a" in source


def test_cvt_blackwell_lines_carry_their_arch_floor():
    """bf16x2 and ue8m0x2 conversions are Blackwell lines; certifying them at
    the sm_90 default would report legal forms as illegal."""
    for name in _CVT_ENTRIES:
        entry = TABLE[name]
        touches_blackwell = any(
            tok in ("bf16x2", "ue8m0x2") for tokens, _, _, _ in renderings(entry) for tok in tokens
        )
        if touches_blackwell:
            assert entry.cert_arch == "sm_100a", name


if __name__ == "__main__":
    pytest.main([__file__])
