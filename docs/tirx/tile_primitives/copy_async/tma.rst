..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

copy_async → tma
================

The ``tma`` variant lowers ``copy_async`` between **global and shared** to the
hardware **Tensor Memory Accelerator**: a single elected thread issues a
descriptor-driven bulk copy (``cp.async.bulk.tensor``), and the hardware walks the
multi-dimensional tile described by a ``cuTensorMap``. The descriptor is built once
on the host (``cuTensorMapEncodeTiled``); the device only *issues* the copy — the
hardware signals the caller's mbarrier when the transfer completes (the dispatch
itself emits no completion op). Source:
``python/tvm/backend/cuda/operator/tile_primitive/copy_async/tma.py``.

What it accepts
---------------

The dispatch registers two predicates — a valid copy and a **single-thread** scope:

.. code-block:: python

    # register_dispatch(..., priority=10, when=[
    predicate("validate_tma_copy_op", lambda op, sctx: (validate_tma_copy_op(op, sctx), "not a valid copy op")),
    predicate("single_thread",    lambda op, sctx: (single_thread(op, sctx),    "expected single thread")),
    # ])

    def single_thread(op_call, sctx):
        return sctx.is_thread            # exactly one elected thread issues the TMA

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Property
     - Requirement
   * - target / priority
     - ``cuda``; priority ``10`` (the bulk path for ``copy_async`` global ↔ shared)
   * - scope
     - **single thread** (``sctx.is_thread``) — TMA is issued by one thread, not a
       partitioned warp
   * - direction
     - ``global → shared`` (g2s) or ``shared → global`` (s2g), inferred from the
       buffer scopes at lowering
   * - dtype / shape
     - ``validate_tma_copy_op``: both sides have layouts, equal dtype, and equal
       selected element count.  For gather4, the source gather axis contributes
       ``len(indexer)`` elements rather than its sliced extent.
   * - layout
     - must form a legal descriptor: rank ≤ 5, innermost stride 1, innermost box
       fits the shared swizzle atom (else the plan search shrinks / declines)
   * - gather4
     - optional ``gather_axis=0`` and ``indexer=[...]`` on ``global → shared``.
       ``len(indexer)`` must be a multiple of four; lowering splits it into one
       ``tile::gather4`` TMA issue per group of four coordinates.
   * - cache policy
     - ``cache_hint`` may be a named string such as ``"evict_first"`` or a
       ``PrimExpr`` policy value.  Non-string values are forwarded as the PTX
       ``cache_policy`` operand and enable ``.L2::cache_hint``.

Demonstration program
----------------------

One thread bulk-copies an ``8×256`` ``float16`` tile global → shared (with a
128-byte swizzled shared layout), signals an mbarrier, waits, then reads it back
(mirrors ``test_tma.py``'s G2S smoke test):

.. code-block:: python

    from tvm.tirx.cuda.operator.tile_primitive.tma_utils import mma_shared_layout

    g_shape = s_shape = (8, 256); dtype = "float16"
    shared_layout = mma_shared_layout(dtype, 3, (8, 256))    # 128-B swizzle
    smem_bytes = 8 * 256 * 2

    @T.prim_func
    def copy_async(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, g_shape, dtype, layout=TileLayout(S[8, 256]))
        B = T.match_buffer(B_ptr, g_shape, dtype, layout=TileLayout(S[8, 256]))
        T.device_entry(); T.cta_id([1]); tid = T.thread_id([8])
        dyn = T.alloc_buffer([smem_bytes + 8], "uint8", scope="shared.dyn")   # arena
        A_smem   = T.decl_buffer(s_shape, dtype, dyn.data, elem_offset=0, layout=shared_layout)
        mbarrier = T.decl_buffer([1], "uint64", dyn.data, elem_offset=smem_bytes // 8)
        phase: T.int32 = 0
        if tid == 0:
            T.ptx.mbarrier.init(mbarrier.ptr_to([0]), 1)
        T.ptx.fence.proxy_async("shared::cta"); T.cuda.cta_sync()
        if tid == 0:
            Tx.copy_async(A_smem[0:8, 0:256], A[0:8, 0:256], dispatch="tma", mbar=mbarrier.ptr_to([0]))
            T.ptx.mbarrier.arrive.expect_tx(mbarrier.ptr_to([0]), smem_bytes)
        T.ptx.mbarrier.try_wait(mbarrier.ptr_to([0]), phase)
        T.ptx.fence.proxy_async("shared::cta"); T.cuda.cta_sync()
        Tx.cta.copy(B[0:8, 0:256], A_smem[0:8, 0:256])

Algorithm
---------

**1. Infer direction from scopes.** ``global → shared`` is g2s, ``shared → global``
is s2g (anything else is an error):

.. code-block:: python

    if src.scope() == "global" and dst.scope().startswith("shared"):
        direction, s_buf, g_buf = "g2s", dst, src
    elif src.scope().startswith("shared") and dst.scope() == "global":
        direction, s_buf, g_buf = "s2g", src, dst

**2. Plan the descriptor (L1 → L2 → L3).** The dispatch canonicalizes both
layouts (L1), then for each global iter finds the maximal contiguous stride-1 shard
chain and cuts the axis into descriptor **box** segments (L2), then stacks those
into a ``cuTensorMap`` and validates the hardware constraints — rank ≤ 5, innermost
stride 1, innermost box fits the shared swizzle atom — shrinking the chain prefix
and retrying if a constraint fails (L3). Adjacent fully-boxed contiguous dims are
merged, and an over-256 box may trigger element-type promotion.

When ``tensor_map_dim_order="natural"`` is set, the planner additionally moves the
physical unit-stride global dimension to the descriptor's innermost slot before
validation. This is needed for layouts such as ``(D, H, S):(1, D, D*H)`` where the
logical slice order is not the same as the physical innermost memory order. The
default path keeps the historical descriptor ordering.

For sparse gathers, ``indexer`` describes absolute source coordinates along the
source gather axis.  Hardware supports only four gathered coordinates per
instruction (``tile::gather4``), so a longer indexer is lowered by slicing the
destination along the matching destination axis:

.. code-block:: python

    Tx.copy_async(
        k_smem[0:16, col:col + 64],
        kv[:, col:col + 64],
        dispatch="tma",
        mbar=bar.ptr_to([buf]),
        cta_group=2,
        gather_axis=0,
        indexer=[idx0, idx1, idx2, idx3, idx4, idx5, idx6, idx7,
                 idx8, idx9, idx10, idx11, idx12, idx13, idx14, idx15],
    )

lowers to four issue calls:

.. code-block:: text

    gather4 dst=k_smem[0:4,   col:col+64], coords=(idx0,  idx1,  idx2,  idx3)
    gather4 dst=k_smem[4:8,   col:col+64], coords=(idx4,  idx5,  idx6,  idx7)
    gather4 dst=k_smem[8:12,  col:col+64], coords=(idx8,  idx9,  idx10, idx11)
    gather4 dst=k_smem[12:16, col:col+64], coords=(idx12, idx13, idx14, idx15)

**3. Emit the host descriptor once,** keyed by a cache so a repeated copy reuses it:

.. code-block:: python

    T.call_packed("runtime.cuTensorMapEncodeTiled", tensormap, dtype_str, rank,
                  tensor_ptr, *reversed(shape), *reversed(strides[:-1]),
                  *reversed(box_dim), *element_strides, 0, swizzle_mode, 2, oob_fill)

**4. Emit the device issue loop** — an unrolled loop over the issue axes, one
``cp.async.bulk.tensor`` per step, direction-specific:

.. code-block:: python

    if direction == "g2s":
        T.ptx.cp_async.bulk.tensor.g2s_cluster(plan.rank, s_buf.ptr_to(s_st), mbar,
                                               T.address_of(tensor_map), cta_mask,
                                               cta_group, cache_hint, *tma_coords)
    else:
        T.ptx.cp_async.bulk.tensor.s2g(plan.rank, s_buf.ptr_to(s_st),
                                       T.address_of(tensor_map), cache_hint, *tma_coords)

Like all ``copy_async`` variants the dispatch emits no completion — the caller's
mbarrier ``arrive.expect_tx`` / ``try_wait`` (g2s) close the loop.

For ``global → shared`` with ``cta_group=2``, the PTX mbarrier operand must be the
leader CTA's 32-bit shared-address operand.  The dispatch converts a generic
shared pointer with ``T.cuda.sm100_2sm_leader_smem_addr`` and passes
``mbar_is_shared_addr=True`` to the PTX wrapper.  If a caller already passes a
shared-address operand through the lower-level PTX wrapper, that conversion is
the caller's responsibility.

Generated TIRx IR
-----------------

The ``8×256`` swizzled tile produces a **rank-3** descriptor and a single issue:

.. code-block:: python

    # host (once): encode the tensor map (rank 3, reversed shape/box/strides, swizzle 3)
    T.call_packed("runtime.cuTensorMapEncodeTiled", A_ptr_tensormap, "float16", 3,
                  A.data, 64, 8, 4, 512, 128, 64, 8, 4, 1, 1, 1, 0, 3, 2, 0)
    # device:
    for loop_vars in T.unroll(1):
        T.ptx.cp_async.bulk.tensor.g2s_cluster(3, T.address_of(s_buf_w_offset[0]),
                                               T.address_of(mbarrier[0]),
                                               T.address_of(A_ptr_tensormap), 0, 1, ..., 0, 0, 0)

Generated CUDA
--------------

.. code-block:: c++

    // one TMA instruction copies the whole rank-3 tile, async, into shared
    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes"
    ".cta_group::1 [%0], [%1, {%3, %4, %5}], [%2];"
    // call: ptx_cp_async_bulk_tensor_g2s_cluster_tile_3d(smem, mbar, tensormap, coords...)

The three ``{%3, %4, %5}`` are the descriptor coordinates; ``[%1]`` is the
tensor-map address, ``[%2]`` the mbarrier. One thread launches the entire 8×256
copy. (This was compiled for ``sm_100a`` — Blackwell — so the instruction carries
the ``.cta_group::1`` qualifier; on Hopper the qualifier is omitted.)

How inputs change the algorithm
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - input
     - effect
   * - direction
     - ``g2s`` → ``cp.async.bulk.tensor.*.g2c``; ``s2g`` → ``…s2g``; with a reduce
       op → ``…s2g_reduce`` (e.g. ``add``)
   * - shared swizzle mode
     - sets the ``swizzle_mode`` in the descriptor and the innermost-box constraint;
       a 128-B swizzle on a 2-D tile yields a **rank-3** descriptor (the inner axis
       splits into swizzle atoms), as in the demo
   * - box shape / chain prefix
     - more selected stride-1 shards → more box>1 descriptor dims; merge collapses
       contiguous full-box dims; box > 256 triggers dtype promotion (1→2→4→8 B)
   * - dtype
     - sets element size and the descriptor's element strides / box byte width
   * - ``tensor_map_dim_order="natural"``
     - keeps the descriptor legal when the physical unit-stride global dimension is
       not already the TensorMap innermost dimension; the planner remaps descriptor
       dims and issue-axis coordinates together before hardware validation
   * - ``gather_axis`` / ``indexer``
     - selects ``tile::gather4`` load mode and changes issue coordinates from one
       dense coordinate on the gather axis to four explicit absolute coordinates.
       Longer indexers generate multiple issue calls, each advancing the
       destination gather-axis slice by four rows.
   * - ``cta_group=2`` mbarrier
     - converts a generic shared pointer to the SM100 2SM leader CTA shared
       address before calling the PTX wrapper, so the generated instruction uses
       the correct mbarrier operand for 2SM TMA.

A copy whose layout cannot form a legal descriptor (rank > 5 after shrinking, or no
swizzle-atom-aligned innermost box) makes the plan search fail and the variant
declines.
