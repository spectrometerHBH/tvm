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

class _Chain_prefetch:
    """`prefetch` — space∈{global}; level∈{L2}"""

    L2: _Chain_prefetch
    global_: _Chain_prefetch
    def __call__(self, addr: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_ld:
    """`ld` — mmio∈{mmio} (opt); sem∈{weak,acquire,relaxed,volatile} (opt);
    scope∈{cta,cluster,gpu,sys} (opt); ss∈{global,shared,shared::cta,shared::cluster,local}
    (opt); cop∈{ca,cg,cs,lu,cv} (opt); nc∈{nc} (opt); l1ev∈{L1::evict_normal,L1::evict_uncha
    nged,L1::evict_first,L1::evict_last,L1::no_allocate} (opt);
    prefetch∈{L2::64B,L2::128B,L2::256B} (opt);
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
    def __call__(self, addr: Any, *args: Any) -> Any: ...

class _Chain_st:
    """`st` — sem∈{weak,release,relaxed,volatile} (opt); scope∈{cta,gpu,sys} (opt);
    space∈{global,shared::cta}; type∈{b32,b64,u32,u64,s32,f32} — release/relaxed require a
    scope; weak/volatile take none; shared::cta caps scope at cta.
    """

    b32: _Chain_st
    b64: _Chain_st
    cta: _Chain_st
    f32: _Chain_st
    global_: _Chain_st
    gpu: _Chain_st
    relaxed: _Chain_st
    release: _Chain_st
    s32: _Chain_st
    shared__cta: _Chain_st
    sys: _Chain_st
    u32: _Chain_st
    u64: _Chain_st
    volatile: _Chain_st
    weak: _Chain_st
    def __call__(self, addr: Any, value: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_red:
    """`red` — sem∈{relaxed,release}; scope∈{gpu,sys}; space∈{global}; op∈{add};
    type∈{u32,s32,f32}
    """

    add: _Chain_red
    f32: _Chain_red
    global_: _Chain_red
    gpu: _Chain_red
    relaxed: _Chain_red
    release: _Chain_red
    s32: _Chain_red
    sys: _Chain_red
    u32: _Chain_red
    def __call__(self, addr: Any, value: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_cvta:
    """`cvta` — dir∈{to}; space∈{shared}; type∈{u64}"""

    shared: _Chain_cvta
    to: _Chain_cvta
    u64: _Chain_cvta
    def __call__(self, ptr: Any, *args: Any) -> Any: ...

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

class _PTXD:
    cp: _Chain_cp
    cvta: _Chain_cvta
    ld: _Chain_ld
    prefetch: _Chain_prefetch
    red: _Chain_red
    st: _Chain_st
    def __getitem__(self, text: str) -> Any: ...

ptxd: _PTXD

# Every other tvm.script.tirx member stays dynamically typed, as before.
def __getattr__(name: str) -> Any: ...
