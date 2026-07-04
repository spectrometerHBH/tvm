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

"""TMA (Tensor Memory Accelerator) utilities for CUDA op dispatches."""

import copy
from enum import Enum

import tvm
from tvm.arith.analyzer import Analyzer
from tvm.tirx.layout import ComposeLayout, Layout, S, SwizzleLayout, TileLayout


class SwizzleMode(Enum):
    """The swizzle mode of the TMA."""

    SWIZZLE_NONE = 0
    SWIZZLE_32B_ATOM = 1
    SWIZZLE_64B_ATOM = 2
    SWIZZLE_128B_ATOM = 3


def mma_atom_layout(dtype: str, swizzle_mode: SwizzleMode | int) -> SwizzleLayout:
    """Generate the MMA-compatible shared-memory atom layout."""
    bits = tvm.DataType(dtype).bits
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    return SwizzleLayout(
        per_element=(128 // bits).bit_length() - 1, swizzle_len=swizzle_mode.value, atom_len=3
    )


def mma_atom_shape(dtype: str, swizzle_mode: SwizzleMode | int, shape: list[int] | None = None):
    """Generate the MMA-compatible shared-memory atom shape."""
    bits = tvm.DataType(dtype).bits
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    atom_shape = {
        SwizzleMode.SWIZZLE_32B_ATOM: [8, 256],
        SwizzleMode.SWIZZLE_64B_ATOM: [8, 512],
        SwizzleMode.SWIZZLE_128B_ATOM: [8, 1024],
    }[swizzle_mode]
    atom_shape[-1] //= bits
    if shape is None:
        return atom_shape
    atom_shape = [1] * (len(shape) - len(atom_shape)) + atom_shape
    return atom_shape


def mma_shared_layout(dtype: str, swizzle_mode: SwizzleMode | int, shape) -> Layout:
    """Generate the MMA-compatible shared-memory layout for shape and dtype.

    It uses a default tiling strategy to tile the TMA atom layout into the shared memory.
    """
    if isinstance(swizzle_mode, int):
        swizzle_mode = SwizzleMode(swizzle_mode)
    if swizzle_mode == SwizzleMode.SWIZZLE_NONE:
        return TileLayout(S[tuple(shape)]).canonicalize()
    atom_shape = mma_atom_shape(dtype, swizzle_mode, shape)
    layout = mma_atom_layout(dtype, swizzle_mode)
    tile_to_shape = copy.copy(atom_shape)
    tile_to_shape[-2] = shape[-2]
    return layout.tile_to(tile_to_shape, atom_shape).tile_to(shape, tile_to_shape).canonicalize()


# Backward-compatible aliases kept during the alloc_mma migration.
tma_atom_layout = mma_atom_layout
tma_atom_shape = mma_atom_shape
tma_shared_layout = mma_shared_layout


def tma_atom_compatible(dst_shape, dst_st, dst_extent, atom_shape):
    """Check if the copy region in dst is compatible with the TMA atom shape."""
    analyzer = Analyzer()
    for i, _ in enumerate(dst_st):
        if any(
            not analyzer.can_prove_equal(x % atom_shape[i], 0)
            for x in [dst_shape[i], dst_st[i], dst_extent[i]]
        ):
            return False
    return True


def get_swizzle_mode_from_layout(layout: Layout) -> SwizzleMode | None:
    """Extract swizzle mode from a shared memory layout.

    The hardware swizzle modes this maps to (TMA descriptor swizzles and the
    tcgen05 smem matrix descriptor walk) implement the ``swizzle_inner=True``
    address permutation ``x ^ ((x & outer_mask) >> atom_len)`` — the atom-row
    bits XORed into the 16B-unit bits — pinned bit-exactly on hardware by the
    TMA / tcgen05.cp GPU round-trip tests. ``swizzle_inner=False`` is the
    mirrored permutation ``x ^ ((x & inner_mask) << atom_len)``; for
    ``swizzle_len=sw >= 1`` (``atom_len=3``) the two coincide only on the
    ``2^(3-sw)`` of the ``2^(3+sw)`` chunks per atom period whose inner
    ``[0, sw)`` and outer ``[3, 3+sw)`` chunk bits are all zero (exhaustive
    enumeration over the full domain), so planning a flipped layout by its
    ``swizzle_len`` alone would silently misplace data — reject it instead.
    For ``swizzle_len == 0`` both masks are 0 and either value is the
    identity, so ``swizzle_inner`` is a don't-care there.
    """
    if isinstance(layout, ComposeLayout):
        swizzle = layout.swizzle  # SwizzleLayout is named 'swizzle' in ComposeLayout
    elif isinstance(layout, SwizzleLayout):
        swizzle = layout
    elif isinstance(layout, TileLayout):
        # TileLayout without SwizzleLayout means no swizzle (mode 0)
        return SwizzleMode.SWIZZLE_NONE
    else:
        return None

    swizzle_len = swizzle.swizzle_len
    if int(swizzle_len) > 0 and not bool(swizzle.swizzle_inner):
        raise ValueError(
            "swizzle direction mismatch: the hardware swizzle modes implement "
            "the swizzle_inner=True address permutation "
            "(x ^ ((x & outer_mask) >> atom_len)); got a SwizzleLayout with "
            f"swizzle_inner=False (per_element={int(swizzle.per_element)}, "
            f"swizzle_len={int(swizzle_len)}, atom_len={int(swizzle.atom_len)}), "
            "whose mirrored permutation would be silently misplaced"
        )

    # Map swizzle_len to SwizzleMode
    return {
        0: SwizzleMode.SWIZZLE_NONE,
        1: SwizzleMode.SWIZZLE_32B_ATOM,
        2: SwizzleMode.SWIZZLE_64B_ATOM,
        3: SwizzleMode.SWIZZLE_128B_ATOM,
    }.get(swizzle_len)
