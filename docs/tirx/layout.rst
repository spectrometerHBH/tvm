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

Tensor Layout
=============

A tensor layout describes how a logical tensor is stored in physical resources.
TIRx generalizes the classical *shape–stride* model: strides are semantically
**named** and bound to **axes** that represent hardware resources — memory,
threads, and devices. A layout maps each logical index to a *set* of coordinates
on these named axes, decomposed into shard (``D``), replica (``R``), and offset
(``O``).

Interactive demo
----------------

Pick a preset, edit the logical shape and the ``S/R/O`` layout, choose a dtype +
swizzle mode, then click an element to see exactly which physical thread(s) own
it.

.. raw:: html

   <p>
     <a class="reference external" href="../_static/tirx-layout-demo/index.html"
        target="_blank" rel="noopener"
        style="display:inline-block; padding:10px 18px; background:#3b82f6;
        color:#fff !important; font-weight:700; border-radius:8px;
        text-decoration:none;">▶ Open the demo full screen ↗</a>
   </p>
   <iframe src="../_static/tirx-layout-demo/index.html?notitle"
           style="width:100%; height:1040px; border:1px solid #dfe1e6;
           border-radius:10px; margin:10px 0 6px;"
           title="TIRx interactive layout demo" loading="lazy"></iframe>

The model
---------

An **iter** is a triple ``(extent, stride, axis)`` that defines a linear,
strided access on one axis.

- **D (Shard).** A list of one or more iters, each with an extent and a stride on
  some axis. ``D`` partitions the logical index across these iters and produces a
  base coordinate; this generalizes shape–stride to multiple axes. Written in
  parentheses, e.g. ``S[(8,2,4,2):(4@laneid,1@warpid,1@laneid,1)]``.
- **R (Replica).** A set of replication iters that enumerate offsets in hardware
  space, independent of the logical index. Adding each element of the set to the
  ``D`` result yields replication or broadcasting. Written in square brackets,
  e.g. ``R[2:4@warpid]``.
- **O (Offset).** A fixed coordinate offset (one integer per axis) added to every
  result. This places data at a base position or reserves exclusive resources.

Formally, for a logical index ``x`` the layout produces

.. math::

   L(x) = \{\, D(x) + r + O \mid r \in R \,\},

where ``D(x)`` is the base coordinate from the sharded iters, ``r`` ranges over
all combinations of the replica iters (a single zero offset when ``R`` is empty),
and ``O`` is the constant offset. ``L(x)`` can be a singleton or contain multiple
coordinates. A term is written ``n @ axis``; if a stride is not paired with an
axis, the memory axis ``m`` is used by default.

Operationally, the mapping flattens the logical coordinate row-major, splits it
by the shard extents into one component per shard iter, accumulates each
component ``c_k * stride_k`` onto its axis, adds the per-axis offsets, and
broadcasts the replica iters so one element can be owned by multiple threads.

Case study: NVIDIA tensor-core tile
-----------------------------------

Consider a logical ``(8, 16)`` tile distributed across 2 warps of 32 lanes each,
with each lane holding part of the tile in its registers (the ``reg`` slot is the
default memory axis ``m``)::

    S[(8,2,4,2):(4@laneid,1@warpid,1@laneid,1)] + R[2:4@warpid] + 5@warpid

The shard factors the logical indices into iters of extent ``8, 2, 4, 2`` over
``laneid``, ``warpid``, ``laneid``, ``m``. The row dimension (extent 8) maps to
``laneid`` with stride 4; the column dimension (extent 16) splits into
``2 × 4 × 2`` over ``warpid``, ``laneid``, and ``m``. The replica copies the tile
twice along ``warpid`` with stride 4 (warps ``{0,1}`` and ``{4,5}``), and the
offset shifts ``warpid`` by 5, so the final copies live on warps ``{5,6}`` and
``{9,10}``. Concretely, element ``(i, j)`` maps to ``laneid = 4i + ⌊j/2⌋ % 4``,
``warpid = ⌊j/8⌋ + 5 + 4r`` for ``r ∈ [0, 2)``, and ``m = j % 2``.

Beyond GPU registers
--------------------

The same layout describes more than register tiles. Binding strides to a device
axis (``pid``) expresses **distributed sharding** across a GPU mesh; binding them
to on-chip memory axes expresses native accelerator memories — a 2D-partitioned
scratchpad (partition ``P`` and free ``F`` axes), or NVIDIA Blackwell tensor
memory with native 2D addressing (``TLane`` × ``TCol``). The demo includes
presets for each.

Swizzle
-------

Some layouts also need a *swizzle*: a non-linear, XOR-based permutation of the
linear memory address that scatters shared-memory bank conflicts. Because it is
not expressible as a strided ``TileLayout``, TIRx represents it as a separate
``SwizzleLayout`` composed with the tile layout (``ComposeLayout(swizzle, tile)``).
A swizzle keeps the low ``per_element`` address bits and XORs a higher bit group
into a lower one. In the demo, choose a dtype and a swizzle mode
(``none`` / ``32B`` / ``64B`` / ``128B``); the physical panel switches to a
bank-by-line view (``bank = address mod 32``): without a swizzle a column access
lands in a single bank (a conflict); with a swizzle the same access is scattered
across banks.

Design rationale
----------------

- **General shape support.** Non-power-of-two shapes are common — in global
  tensors, multi-stage shared-memory buffers, and capacity-limited on-chip
  scratchpads — so the layout supports general shapes directly rather than as a
  special case.
- **Logical-to-physical mapping.** The map goes from logical coordinates to a set
  of physical coordinates. This lets replication (one logical element in multiple
  physical locations) be expressed cleanly, which a physical-to-logical
  formulation cannot always represent for strided patterns.
- **Explicit hardware axes.** Axes carry their hardware meaning in the layout
  itself, so an expression is unambiguous without external context. For instance
  ``1@tid`` (block-wide thread id) and ``1@tid_in_wg`` (thread id within a
  warpgroup) are distinct rather than a generic ``t`` whose meaning depends on the
  definition site. Legality and feasibility checks are left to tile primitive
  dispatch.
