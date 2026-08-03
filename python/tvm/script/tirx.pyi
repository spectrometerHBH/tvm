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
"""Generated stub for T.ptxd — do not edit.

Regenerate:
  python -m tvm.backend.cuda.ptx_dialect.gen_stubs -o python/tvm/script/tirx.pyi
"""

from typing import Any

class _Chain_add:
    """`add` — rnd∈{rn,rz,rm,rp} (opt); ftz∈{ftz} (opt); sat∈{sat} (opt); type∈{f32,f64,f32x2};
    srctype∈{f16,bf16} (opt) — Which qualifiers each add/sub/mul/fma syntax line allows (PTX
    ISA 9.7.3.{3,4,5,6}, 9.7.5).      Same-precision lines:  op{.rnd}{.ftz}{.sat}.f32 |
    op{.rnd}{.ftz}.f32x2 | op{.rnd}.f64     Mixed-precision lines: op{.rnd}{.sat}.f32.atype
    (.atype = .f16 | .bf16)
    """

    bf16: _Chain_add
    f16: _Chain_add
    f32: _Chain_add
    f32x2: _Chain_add
    f64: _Chain_add
    ftz: _Chain_add
    rm: _Chain_add
    rn: _Chain_add
    rp: _Chain_add
    rz: _Chain_add
    sat: _Chain_add
    def __call__(self, d: Any, a: Any, b: Any, *args: Any) -> None: ...

class _Chain_atom:
    """`atom` — sem∈{relaxed,acquire,release,acq_rel} (opt); scope∈{cta,cluster,gpu,sys} (opt);
    space∈{global,shared,shared::cta,shared::cluster} (opt);
    op∈{and,or,xor,add,inc,dec,min,max}; type∈{b32,b64,u32,u64,s32,s64,f32,f64} — op x type
    pairings for atom/red (PTX ISA 9.7.14.5 / 9.7.14.6).      Normative source: ISA Table 35
    (atom) and Table 36 (red), which give the     pairing cell by cell. The `.type = {...}`
    line in the Syntax block is only     the union across ops, which is why it cannot be
    transcribed directly. Half-precision     types appear in ptxas' message but are excluded
    from this entry (they need     .noftz and a half carrier type).
    """

    acq_rel: _Chain_atom
    acquire: _Chain_atom
    add: _Chain_atom
    and_: _Chain_atom
    b32: _Chain_atom
    b64: _Chain_atom
    cluster: _Chain_atom
    cta: _Chain_atom
    dec: _Chain_atom
    f32: _Chain_atom
    f64: _Chain_atom
    global_: _Chain_atom
    gpu: _Chain_atom
    inc: _Chain_atom
    max: _Chain_atom
    min: _Chain_atom
    or_: _Chain_atom
    relaxed: _Chain_atom
    release: _Chain_atom
    s32: _Chain_atom
    s64: _Chain_atom
    shared: _Chain_atom
    shared__cluster: _Chain_atom
    shared__cta: _Chain_atom
    sys: _Chain_atom
    u32: _Chain_atom
    u64: _Chain_atom
    xor: _Chain_atom
    def __call__(self, d: Any, addr: Any, value: Any, *args: Any) -> None: ...

class _Chain_cp:
    """`cp` — api∈{async}; kind∈{bulk}; dst_space∈{shared::cta}; src_space∈{global};
    completion∈{mbarrier::complete_tx::bytes}
    """

    async_: _Chain_cp
    bulk: _Chain_cp
    global_: _Chain_cp
    mbarrier__complete_tx__bytes: _Chain_cp
    shared__cta: _Chain_cp
    def __call__(
        self,
        dst: Any,
        src: Any,
        size: Any,
        mbar: Any,
        *args: Any,
        pred: Any = None,
    ) -> None: ...

class _Chain_cvta:
    """`cvta` — dir∈{to}; space∈{shared}; type∈{u64}"""

    shared: _Chain_cvta
    to: _Chain_cvta
    u64: _Chain_cvta
    def __call__(self, d: Any, ptr: Any, *args: Any) -> None: ...

class _Chain_ex2:
    """`ex2` — mode∈{approx}; ftz∈{ftz} (opt); type∈{f32}"""

    approx: _Chain_ex2
    f32: _Chain_ex2
    ftz: _Chain_ex2
    def __call__(self, d: Any, value: Any, *args: Any) -> None: ...

class _Chain_fma:
    """`fma` — rnd∈{rn,rz,rm,rp}; ftz∈{ftz} (opt); sat∈{sat} (opt); type∈{f32,f64,f32x2};
    srctype∈{f16,bf16} (opt) — Which qualifiers each add/sub/mul/fma syntax line allows (PTX
    ISA 9.7.3.{3,4,5,6}, 9.7.5).      Same-precision lines:  op{.rnd}{.ftz}{.sat}.f32 |
    op{.rnd}{.ftz}.f32x2 | op{.rnd}.f64     Mixed-precision lines: op{.rnd}{.sat}.f32.atype
    (.atype = .f16 | .bf16)
    """

    bf16: _Chain_fma
    f16: _Chain_fma
    f32: _Chain_fma
    f32x2: _Chain_fma
    f64: _Chain_fma
    ftz: _Chain_fma
    rm: _Chain_fma
    rn: _Chain_fma
    rp: _Chain_fma
    rz: _Chain_fma
    sat: _Chain_fma
    def __call__(self, d: Any, a: Any, b: Any, c: Any, *args: Any) -> None: ...

class _Chain_fns:
    """`fns` — type∈{b32}"""

    b32: _Chain_fns
    def __call__(self, d: Any, mask: Any, base: Any, offset: Any, *args: Any) -> None: ...

class _Chain_ld:
    """`ld` — mmio∈{mmio} (opt); sem∈{weak,acquire,relaxed,volatile} (opt);
    scope∈{cta,cluster,gpu,sys} (opt);
    space∈{global,shared,shared::cta,shared::cluster,local} (opt); cop∈{ca,cg,cs,lu,cv}
    (opt); nc∈{nc} (opt); l1ev∈{L1::evict_normal,L1::evict_unchanged,L1::evict_first,L1::evi
    ct_last,L1::no_allocate} (opt); prefetch∈{L2::64B,L2::128B,L2::256B} (opt);
    type∈{b8,u8,s8,b16,u16,s16,b32,u32,s32,b64,u64,s64,f32,f64} — Scalar ld grammar per PTX
    ISA 9.7.9.8 (ld) and 9.7.9.9 (ld.global.nc).
    """

    L1__evict_first: _Chain_ld
    L1__evict_last: _Chain_ld
    L1__evict_normal: _Chain_ld
    L1__evict_unchanged: _Chain_ld
    L1__no_allocate: _Chain_ld
    L2__128B: _Chain_ld
    L2__256B: _Chain_ld
    L2__64B: _Chain_ld
    acquire: _Chain_ld
    b16: _Chain_ld
    b32: _Chain_ld
    b64: _Chain_ld
    b8: _Chain_ld
    ca: _Chain_ld
    cg: _Chain_ld
    cluster: _Chain_ld
    cs: _Chain_ld
    cta: _Chain_ld
    cv: _Chain_ld
    f32: _Chain_ld
    f64: _Chain_ld
    global_: _Chain_ld
    gpu: _Chain_ld
    local: _Chain_ld
    lu: _Chain_ld
    mmio: _Chain_ld
    nc: _Chain_ld
    relaxed: _Chain_ld
    s16: _Chain_ld
    s32: _Chain_ld
    s64: _Chain_ld
    s8: _Chain_ld
    shared: _Chain_ld
    shared__cluster: _Chain_ld
    shared__cta: _Chain_ld
    sys: _Chain_ld
    u16: _Chain_ld
    u32: _Chain_ld
    u64: _Chain_ld
    u8: _Chain_ld
    volatile: _Chain_ld
    weak: _Chain_ld
    def __call__(self, d: Any, addr: Any, *args: Any) -> None: ...

class _Chain_max:
    """`max` — ftz∈{ftz} (opt); nan∈{NaN} (opt); type∈{f32,f64} — `max.f64` is the bare form;
    .ftz/.NaN belong to the .f32 line (PTX ISA 9.7.3.12).
    """

    NaN: _Chain_max
    f32: _Chain_max
    f64: _Chain_max
    ftz: _Chain_max
    def __call__(self, d: Any, a: Any, b: Any, *args: Any) -> None: ...

class _Chain_mov:
    """`mov` — 16 entries sharing this mnemonic; PTX puts their difference in the operand list,
    so the call selects one. Shapes: (d, a0, a1); (d0, d1, a); (d, a0, a1, a2, a3); (d0, d1,
    d2, d3, a)
    """

    b128: _Chain_mov
    b32: _Chain_mov
    b64: _Chain_mov
    def __call__(self, *args: Any) -> None: ...

class _Chain_mul:
    """`mul` — rnd∈{rn,rz,rm,rp} (opt); ftz∈{ftz} (opt); sat∈{sat} (opt); type∈{f32,f64,f32x2}
    — Which qualifiers each add/sub/mul/fma syntax line allows (PTX ISA 9.7.3.{3,4,5,6},
    9.7.5).      Same-precision lines:  op{.rnd}{.ftz}{.sat}.f32 | op{.rnd}{.ftz}.f32x2 |
    op{.rnd}.f64     Mixed-precision lines: op{.rnd}{.sat}.f32.atype  (.atype = .f16 |
    .bf16)
    """

    f32: _Chain_mul
    f32x2: _Chain_mul
    f64: _Chain_mul
    ftz: _Chain_mul
    rm: _Chain_mul
    rn: _Chain_mul
    rp: _Chain_mul
    rz: _Chain_mul
    sat: _Chain_mul
    def __call__(self, d: Any, a: Any, b: Any, *args: Any) -> None: ...

class _Chain_prefetch:
    """`prefetch` — space∈{global,local,const,param} (opt); level∈{L1,L2} (opt);
    evict∈{L2::evict_last,L2::evict_normal} (opt); tensormap∈{tensormap} (opt) — Each
    prefetch syntax line names exactly one target (PTX ISA 9.7.9.16).
    `.level::eviction_priority` stays bound to `.global` on purpose: its syntax     line is
    `prefetch.global.level::eviction_priority`, with `.global` written     in rather than
    the `{.ss}` that the `ld` lines carry. Generic addressing is     not offered there, so
    neither is it here.
    """

    L1: _Chain_prefetch
    L2: _Chain_prefetch
    L2__evict_last: _Chain_prefetch
    L2__evict_normal: _Chain_prefetch
    const: _Chain_prefetch
    global_: _Chain_prefetch
    local: _Chain_prefetch
    param: _Chain_prefetch
    tensormap: _Chain_prefetch
    def __call__(self, addr: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_rcp:
    """`rcp` — mode∈{approx,rn,rz,rm,rp}; ftz∈{ftz} (opt); type∈{f32,f64} — This entry's
    rcp.approx is f32-only; .f64 is IEEE-rounded, no .ftz (PTX ISA 9.7.3.13).
    """

    approx: _Chain_rcp
    f32: _Chain_rcp
    f64: _Chain_rcp
    ftz: _Chain_rcp
    rm: _Chain_rcp
    rn: _Chain_rcp
    rp: _Chain_rcp
    rz: _Chain_rcp
    def __call__(self, d: Any, value: Any, *args: Any) -> None: ...

class _Chain_red:
    """`red` — sem∈{relaxed,release} (opt); scope∈{cta,cluster,gpu,sys} (opt);
    space∈{global,shared,shared::cta,shared::cluster} (opt);
    op∈{and,or,xor,add,inc,dec,min,max}; type∈{b32,b64,u32,u64,s32,s64,f32,f64} — op x type
    pairings for atom/red (PTX ISA 9.7.14.5 / 9.7.14.6).      Normative source: ISA Table 35
    (atom) and Table 36 (red), which give the     pairing cell by cell. The `.type = {...}`
    line in the Syntax block is only     the union across ops, which is why it cannot be
    transcribed directly. Half-precision     types appear in ptxas' message but are excluded
    from this entry (they need     .noftz and a half carrier type).
    """

    add: _Chain_red
    and_: _Chain_red
    b32: _Chain_red
    b64: _Chain_red
    cluster: _Chain_red
    cta: _Chain_red
    dec: _Chain_red
    f32: _Chain_red
    f64: _Chain_red
    global_: _Chain_red
    gpu: _Chain_red
    inc: _Chain_red
    max: _Chain_red
    min: _Chain_red
    or_: _Chain_red
    relaxed: _Chain_red
    release: _Chain_red
    s32: _Chain_red
    s64: _Chain_red
    shared: _Chain_red
    shared__cluster: _Chain_red
    shared__cta: _Chain_red
    sys: _Chain_red
    u32: _Chain_red
    u64: _Chain_red
    xor: _Chain_red
    def __call__(self, addr: Any, value: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_st:
    """`st` — mmio∈{mmio} (opt); sem∈{weak,release,relaxed,volatile} (opt);
    scope∈{cta,cluster,gpu,sys} (opt);
    space∈{global,shared,shared::cta,shared::cluster,local} (opt); cop∈{wb,cg,cs,wt} (opt);
    l1ev∈{L1::evict_normal,L1::evict_unchanged,L1::evict_first,L1::evict_last,L1::no_allocat
    e} (opt); type∈{b8,u8,s8,b16,u16,s16,b32,u32,s32,b64,u64,s64,f32,f64} — Scalar st
    grammar per PTX ISA 9.7.9.11 (the mirror of _check_ld).
    """

    L1__evict_first: _Chain_st
    L1__evict_last: _Chain_st
    L1__evict_normal: _Chain_st
    L1__evict_unchanged: _Chain_st
    L1__no_allocate: _Chain_st
    b16: _Chain_st
    b32: _Chain_st
    b64: _Chain_st
    b8: _Chain_st
    cg: _Chain_st
    cluster: _Chain_st
    cs: _Chain_st
    cta: _Chain_st
    f32: _Chain_st
    f64: _Chain_st
    global_: _Chain_st
    gpu: _Chain_st
    local: _Chain_st
    mmio: _Chain_st
    relaxed: _Chain_st
    release: _Chain_st
    s16: _Chain_st
    s32: _Chain_st
    s64: _Chain_st
    s8: _Chain_st
    shared: _Chain_st
    shared__cluster: _Chain_st
    shared__cta: _Chain_st
    sys: _Chain_st
    u16: _Chain_st
    u32: _Chain_st
    u64: _Chain_st
    u8: _Chain_st
    volatile: _Chain_st
    wb: _Chain_st
    weak: _Chain_st
    wt: _Chain_st
    def __call__(self, addr: Any, value: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_st_bulk:
    """`st_bulk` — weak∈{weak} (opt); space∈{shared::cta} (opt)"""

    shared__cta: _Chain_st_bulk
    weak: _Chain_st_bulk
    def __call__(self, addr: Any, size: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_sub:
    """`sub` — rnd∈{rn,rz,rm,rp} (opt); ftz∈{ftz} (opt); sat∈{sat} (opt); type∈{f32,f64,f32x2};
    srctype∈{f16,bf16} (opt) — Which qualifiers each add/sub/mul/fma syntax line allows (PTX
    ISA 9.7.3.{3,4,5,6}, 9.7.5).      Same-precision lines:  op{.rnd}{.ftz}{.sat}.f32 |
    op{.rnd}{.ftz}.f32x2 | op{.rnd}.f64     Mixed-precision lines: op{.rnd}{.sat}.f32.atype
    (.atype = .f16 | .bf16)
    """

    bf16: _Chain_sub
    f16: _Chain_sub
    f32: _Chain_sub
    f32x2: _Chain_sub
    f64: _Chain_sub
    ftz: _Chain_sub
    rm: _Chain_sub
    rn: _Chain_sub
    rp: _Chain_sub
    rz: _Chain_sub
    sat: _Chain_sub
    def __call__(self, d: Any, a: Any, b: Any, *args: Any) -> None: ...

class _PTXD:
    add: _Chain_add
    atom: _Chain_atom
    cp: _Chain_cp
    cvta: _Chain_cvta
    ex2: _Chain_ex2
    fma: _Chain_fma
    fns: _Chain_fns
    ld: _Chain_ld
    max: _Chain_max
    mov: _Chain_mov
    mul: _Chain_mul
    prefetch: _Chain_prefetch
    rcp: _Chain_rcp
    red: _Chain_red
    st: _Chain_st
    st_bulk: _Chain_st_bulk
    sub: _Chain_sub
    def __getitem__(self, text: str) -> Any: ...

ptxd: _PTXD

# Every other tvm.script.tirx member stays dynamically typed, as before.
def __getattr__(name: str) -> Any: ...
