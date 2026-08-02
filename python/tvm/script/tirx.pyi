"""Generated stub for T.ptxd — do not edit.

Regenerate:
  python -m tvm.backend.cuda.ptx_dialect.gen_stubs -o python/tvm/script/tirx.pyi
"""

from typing import Any

class _Chain_prefetch:
    """`prefetch` — space∈{global}; level∈{L2}
    """
    L2: _Chain_prefetch
    global_: _Chain_prefetch
    def __call__(self, addr: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_ld:
    """`ld` — sem∈{acquire,relaxed,volatile} (opt); scope∈{cta,gpu,sys} (opt); space∈{global};
    type∈{b32,b64,u32,f32} — acquire/relaxed require a scope; weak (omitted) and volatile forms
    take none.
    """
    acquire: _Chain_ld
    b32: _Chain_ld
    b64: _Chain_ld
    cta: _Chain_ld
    f32: _Chain_ld
    global_: _Chain_ld
    gpu: _Chain_ld
    relaxed: _Chain_ld
    sys: _Chain_ld
    u32: _Chain_ld
    volatile: _Chain_ld
    def __call__(self, addr: Any, *args: Any) -> Any: ...

class _Chain_st:
    """`st` — sem∈{weak,release,relaxed,volatile} (opt); scope∈{cta,gpu,sys} (opt);
    space∈{global,shared::cta}; type∈{b32,b64,u32,f32} — release/relaxed require a scope;
    weak/volatile take none; shared::cta caps scope at cta.
    """
    b32: _Chain_st
    b64: _Chain_st
    cta: _Chain_st
    f32: _Chain_st
    global_: _Chain_st
    gpu: _Chain_st
    relaxed: _Chain_st
    release: _Chain_st
    shared__cta: _Chain_st
    sys: _Chain_st
    u32: _Chain_st
    volatile: _Chain_st
    weak: _Chain_st
    def __call__(self, addr: Any, value: Any, *args: Any, pred: Any = None) -> None: ...

class _Chain_red:
    """`red` — sem∈{relaxed,release}; scope∈{gpu,sys}; space∈{global}; op∈{add}; type∈{u32,s32,f32}
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
    """`cvta` — dir∈{to}; space∈{shared}; type∈{u64}
    """
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
