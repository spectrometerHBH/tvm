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
# pylint: disable=invalid-name
"""Synchronization primitives.

PTX side:
* ``bar.arrive`` / ``bar.sync`` — aligned named-barrier aliases
* ``barrier.sync`` — unaligned named barrier for divergent control flow
* ``fence{.sem}.scope`` / ``fence.proxy.async`` / ``fence.mbarrier_init``
* ``barrier.cluster.arrive`` / ``barrier.cluster.wait``
* ``mbarrier.try_wait``
* ``elect.sync``  — warp leader election
* warp-vote ``__any_sync``

CUDA-side helpers:
* ``__threadfence`` / ``__syncwarp`` / ``__syncthreads`` / ``__syncthreads_and|or``
* cooperative-groups grid sync
* cluster sync (open-coded ``barrier.cluster.arrive/wait`` pair)
* warpgroup sync (``bar.sync``)
"""

from tvm.tirx.operator.intrinsics._common import (
    CLUSTER_BARRIER_SEM,
)

from ._schema import device_intrinsic
from .utils import parse_str

# =============================================================================
# bar.arrive / bar.sync — aligned named-barrier aliases. 1 form each.
#   bar.sync   a, b ;
#   bar.arrive a, b ;
# barrier.sync — unaligned named barrier. 1 form.
#   barrier.sync a, b ;
# =============================================================================


# =============================================================================
# fence{.sem}.scope — 1 form (sem/scope are modifier values).


# =============================================================================
# fence.proxy.async{.<space>} — 1 form, optional .space modifier.


# =============================================================================
# fence.mbarrier_init.release.cluster — 1 form, no operands.
# =============================================================================


# =============================================================================
# barrier.cluster.arrive{.sem}{.aligned} — 1 form.
# =============================================================================
def _ptx_barrier_cluster_arrive(sem, aligned):
    sem = parse_str(sem)
    aligned = bool(int(aligned)) if hasattr(aligned, "value") else bool(aligned)
    assert sem in CLUSTER_BARRIER_SEM, (
        f"invalid cluster.arrive sem {sem!r}, expected one of {CLUSTER_BARRIER_SEM}"
    )
    sem_suffix = f".{sem}" if sem else ""
    aligned_suffix = ".aligned" if aligned else ""
    name_sem = "_" + sem.replace("::", "_").replace(".", "_") if sem else ""
    name_aligned = "_aligned" if aligned else ""
    return (
        f"tvm_builtin_ptx_barrier_cluster_arrive{name_sem}{name_aligned}",
        f'    asm volatile("barrier.cluster.arrive{sem_suffix}{aligned_suffix};" ::: "memory");',
    )


# =============================================================================
# barrier.cluster.wait{.acquire}{.aligned} — 1 form.
# =============================================================================
def _ptx_barrier_cluster_wait(acquire, aligned):
    acquire = bool(int(acquire)) if hasattr(acquire, "value") else bool(acquire)
    aligned = bool(int(aligned)) if hasattr(aligned, "value") else bool(aligned)
    acq_suffix = ".acquire" if acquire else ""
    aligned_suffix = ".aligned" if aligned else ""
    return (
        f"tvm_builtin_ptx_barrier_cluster_wait"
        f"{'_acquire' if acquire else ''}{'_aligned' if aligned else ''}",
        f'    asm volatile("barrier.cluster.wait{acq_suffix}{aligned_suffix};" ::: "memory");',
    )


# =============================================================================
# clusterlaunchcontrol.try_cancel / query_cancel — Blackwell Cluster Launch
# Control (CLC) work-stealing, written from the PTX ISA spec (section
# "clusterlaunchcontrol", PTX ISA 8.6). try_cancel async-requests cancelling the
# next cluster's launch, writing a 16B response to smem + signalling mbar. query
# decodes the response: on success it extracts the cancelled cluster's first
# ctaid.x (via the get_first_ctaid::x form); a single uint32 is returned, with
# 0xFFFFFFFF as the "no work stolen" sentinel (a device helper returns one scalar).
# =============================================================================


def _ptx_clc_query_cancel_parts(use_ld_acquire):
    use_ld_acquire = (
        bool(int(use_ld_acquire)) if hasattr(use_ld_acquire, "value") else bool(use_ld_acquire)
    )
    name = f"tvm_builtin_ptx_clc_query_cancel{'_ld_acquire' if use_ld_acquire else ''}"
    load_instr = "ld.acquire.cta.shared.b128" if use_ld_acquire else "ld.shared.b128"
    body = (
        "    unsigned int addr = (unsigned int)__cvta_generic_to_shared(handle);\n"
        "    unsigned int first_ctaid_x;\n"
        "    asm volatile(\n"
        '        "{\\n"\n'
        '        ".reg .pred canceled;\\n"\n'
        '        ".reg .b128 response;\\n"\n'
        f'        "{load_instr} response, [%1];\\n"\n'
        '        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 canceled, response;\\n"\n'
        '        "mov.u32 %0, 0xffffffff;\\n"\n'
        '        "@canceled clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128"\n'
        '        " %0, response;\\n"\n'
        '        "}\\n"\n'
        '        : "=r"(first_ctaid_x) : "r"(addr) : "memory");\n'
        '    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");\n'
        "    return first_ctaid_x;"
    )
    return name, body


device_intrinsic(
    "ptx_clc_query_cancel",
    n_attrs=1,
    helper_name=lambda *args: _ptx_clc_query_cancel_parts(args[-1])[0],
    c_signature="(void* handle)",
    return_type="uint32_t",
    tvm_return_type="uint32",
    body=lambda *args: _ptx_clc_query_cancel_parts(args[-1])[1],
)


# =============================================================================
# mbarrier.try_wait.parity.shared::cta.b64 — 1 form. Body wraps the asm in a
# label loop (TIRx convention; the magic ``ticks = 0x989680`` is the timeout
# hint in ns).
# =============================================================================
device_intrinsic(
    "cuda_mbarrier_wait",
    c_signature="(void* barrier, int phase)",
    body=(
        "    unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);\n"
        "    unsigned int ticks = 0x989680;\n"
        "    asm volatile(\n"
        '        "{\\n"\n'
        '        ".reg .pred                P1;\\n"\n'
        '        "LAB_WAIT:\\n"\n'
        '        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\\n"\n'
        '        "@P1                       bra.uni DONE;\\n"\n'
        '        "bra.uni                   LAB_WAIT;\\n"\n'
        '        "DONE:\\n"\n'
        '        "}\\n"\n'
        '        :: "r"(barrier_addr_int), "r"(phase), "r"(ticks) : "memory");'
    ),
)


# mbarrier.try_wait.parity.acquire.cluster — cluster-scope acquire wait used for
# cross-CTA barrier handshakes (e.g. the tmem-finished handoff).
device_intrinsic(
    "cuda_mbarrier_wait_acquire_cluster",
    c_signature="(void* barrier, int phase)",
    body=(
        "    unsigned int barrier_addr_int = __cvta_generic_to_shared(barrier);\n"
        "    asm volatile(\n"
        '        "{\\n"\n'
        '        ".reg .pred                P1;\\n"\n'
        '        "LAB_WAIT_AC:\\n"\n'
        '        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64 P1, [%0], %1;\\n"\n'
        '        "@P1                       bra.uni DONE_AC;\\n"\n'
        '        "bra.uni                   LAB_WAIT_AC;\\n"\n'
        '        "DONE_AC:\\n"\n'
        '        "}\\n"\n'
        '        :: "r"(barrier_addr_int), "r"(phase) : "memory");'
    ),
)


# =============================================================================
# elect.sync — TIRx uses the CUDA builtin ``tvm_builtin_elect_one_sync()``
# helper (declared in the CUDA header tags), not direct PTX.
# =============================================================================
device_intrinsic(
    "ptx_elect_sync",
    helper_name="tvm_builtin_elect_one_sync_op",
    return_type="uint32_t",
    body="    return tvm_builtin_elect_one_sync();",
    extra_deps=("elect_one_sync",),
)


# =============================================================================
# __any_sync — warp-vote (pure CUDA helper).
# =============================================================================
device_intrinsic(
    "cuda_any_sync",
    c_signature="(unsigned mask, int pred)",
    body="    return __any_sync(mask, pred);",
    return_type="int",
    tvm_return_type="int32",
)


# =============================================================================
# CUDA-side sync helpers (zero-arg void unless noted).
# =============================================================================
device_intrinsic("cuda_thread_fence", body="    __threadfence();")
device_intrinsic("cuda_warp_sync", body="    __syncwarp();")
device_intrinsic("cuda_cta_sync", body="    __syncthreads();")
device_intrinsic(
    "cuda_grid_sync",
    body="    namespace cg = cooperative_groups;\n    cg::this_grid().sync();",
    extra_deps=("cooperative_groups",),
)
device_intrinsic(
    "cuda_cluster_sync",
    body=('    asm("barrier.cluster.arrive.aligned;");\n    asm("barrier.cluster.wait.aligned;");'),
)
device_intrinsic(
    "cuda_warpgroup_sync",
    c_signature="(int name_bar_id)",
    body='    asm volatile("bar.sync %0, 128;" : : "r"(name_bar_id));',
)
device_intrinsic(
    "cuda_syncthreads_and",
    c_signature="(int predicate)",
    body="    return __syncthreads_and(predicate);",
    return_type="int",
    tvm_return_type="int32",
)
device_intrinsic(
    "cuda_syncthreads_or",
    c_signature="(int predicate)",
    body="    return __syncthreads_or(predicate);",
    return_type="int",
    tvm_return_type="int32",
)


# =============================================================================
# Additional mbarrier, grid-sync, and warp collective helpers.
# =============================================================================


device_intrinsic(
    "cuda_ballot_sync",
    helper_name="tvm_builtin_ballot_sync",
    c_signature="(unsigned int mask, int pred)",
    return_type="unsigned int",
    body="    return __ballot_sync(mask, pred);",
)
device_intrinsic(
    "cuda_reduce_add_sync_u32",
    helper_name="tvm_builtin_reduce_add_sync_u32",
    c_signature="(unsigned int mask, unsigned int value)",
    return_type="unsigned int",
    body="    return __reduce_add_sync(mask, value);",
)
device_intrinsic(
    "cuda_reduce_min_sync_u32",
    helper_name="tvm_builtin_reduce_min_sync_u32",
    c_signature="(unsigned int mask, unsigned int value)",
    return_type="unsigned int",
    body="    return __reduce_min_sync(mask, value);",
)


# =============================================================================
# griddepcontrol.wait / griddepcontrol.launch_dependents (sm_90+)
# Programmatic Dependent Launch (PDL) synchronization. Both carry memory
# clobber to prevent CSE / cross-barrier reordering.
# =============================================================================
