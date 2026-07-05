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
"""Abstraction for array data structures."""

import functools
from numbers import Integral

import tvm_ffi

import tvm
from tvm.ir import PointerType, PrimExpr, PrimType, Range
from tvm.runtime import Object, Scriptable, convert

from . import _ffi_api


@tvm_ffi.register_object("tirx.Buffer")
class Buffer(Object, Scriptable):
    """Symbolic data buffer in TVM.

    Buffer provide a way to represent data layout
    specialization of data structure in TVM.

    Do not construct directly, use :py:func:`~decl_buffer` instead.
    See the documentation of :py:func:`decl_buffer` for more details.

    See Also
    --------
    decl_buffer : Declare a buffer
    """

    READ = 1
    WRITE = 2

    def access_ptr(self, access_mask, ptr_type="handle", content_lanes=1, offset=0, extent=None):
        """Get an access pointer to the head of buffer.

        This is the recommended method to get buffer data
        ptress when interacting with external functions.

        Parameters
        ----------
        access_mask : int
            The access pattern MASK. Indicate whether the
            access will read or write to the data content.

        ptr_type : str, optional
            The data type of the result pointer. Do not specify
            unless we want to cast pointer to specific type.

        content_lanes: int, optional
            The number of lanes for the data type. This value
            is greater than one for vector types.

        offset: Expr, optional
            The offset of pointer. We can use it to offset by
            the number of elements from the address of ptr.

        extent: Expr, optional
            The extent of pointer.

        Examples
        --------
        .. code-block:: python

          # Get access ptr for read
          buffer.access_ptr("r")
          # Get access ptr for read/write with bitmask
          buffer.access_ptr(Buffer.READ | Buffer.WRITE)
          # Get access ptr for read/write with str flag
          buffer.access_ptr("rw")
          # Get access ptr for read with offset
          buffer.access_ptr("r", offset = 100)
          # Get access ptr for read with extent
          buffer.access_ptr("r", extent = 100)
        """
        if isinstance(access_mask, str):
            mask = 0
            for value in access_mask:
                if value == "r":
                    mask = mask | Buffer.READ
                elif value == "w":
                    mask = mask | Buffer.WRITE
                else:
                    raise ValueError(f"Unknown access_mask {access_mask}")
            access_mask = mask
        offset = convert(offset)
        extent = convert(extent)
        return _ffi_api.BufferAccessPtr(
            self,
            access_mask,
            ptr_type,
            content_lanes,
            offset,
            extent,  # type: ignore
        )

    def vload(self, begin, dtype=None, predicate=None):
        """Generate an Expr that loads dtype from begin index.

        Parameters
        ----------
        begin : Array of Expr
            The beginning index in unit of Buffer.dtype

        dtype : str
            The data type to be loaded,
            can be vector type which have lanes that is multiple of Buffer.dtype

        predicate : Optional[PrimExpr]
            A vector mask of boolean values indicating which lanes of a vector are to be
            loaded. The number lanes of the mask must be equal to the number of lanes being loaded.

        Returns
        -------
        load : Expr
            The corresponding load expression.
        """
        begin = (begin,) if isinstance(begin, int | PrimExpr) else begin
        dtype = dtype if dtype else self.dtype
        return _ffi_api.BufferVLoad(self, begin, dtype, predicate)  # type: ignore

    def vstore(self, begin, value, predicate=None):
        """Generate a Stmt that store value into begin index.

        Parameters
        ----------
        begin : Array of Expr
            The beginning index in unit of Buffer.dtype

        value : Expr
            The value to be stored.

        predicate : Optional[PrimExpr]
            A vector mask of boolean values indicating which lanes of a vector are to be
            stored. The number lanes of the mask must be equal to the number of lanes in
            value.

        Returns
        -------
        store : Stmt
            The corresponding store stmt.
        """
        begin = (begin,) if isinstance(begin, int | PrimExpr) else begin
        return _ffi_api.BufferVStore(self, begin, value, predicate)  # type: ignore

    def scope(self):
        """Return the storage scope associated with this buffer.
        Returns
        -------
        scope : str
            The storage scope associated with this buffer.
        """
        return _ffi_api.BufferStorageScope(self)  # type: ignore

    def get_flattened_buffer(self):
        """Generate a Buffer that is a flattened version of this buffer.

        Returns
        -------
        flattened : Buffer
            The corresponding flat buffer.
        """
        return _ffi_api.BufferGetFlattenedBuffer(self)  # type: ignore

    def with_allocated_addr(self, allocated_addr):
        """Return a new buffer with the allocated address."""
        return _ffi_api.BufferWithAllocatedAddr(self, allocated_addr)  # type: ignore

    def with_dtype(self, dtype):
        """Return a new buffer with the dtype."""
        return _ffi_api.BufferWithDtype(self, dtype)  # type: ignore

    def with_data(self, data):
        """Return a new buffer with the data."""
        return _ffi_api.BufferWithData(self, data)  # type: ignore

    def offset_of(self, indices):
        """Determine the offset of the provided indices in the flattened buffer.

        Parameters
        ----------
        indices : Union[PrimExpr, List[PrimExpr]]

            The indices of the element in the original buffer.

        Returns
        -------
        flattened_indices: List[PrimExpr]

            The offset indices of the element in the flattened buffer.
        """
        return _ffi_api.BufferOffsetOf(self, indices)  # type: ignore

    @property
    def byte_offset(self):
        """Get the byte offset of the buffer."""
        return self.elem_offset * tvm.DataType(self.dtype).bits // 8

    def elem_offset_of(self, indices, inner=True):
        """Get the element offset of the buffer at the given indices.
        Note that indices subject to buffer's layout mapping.

        Parameters
        ----------
        indices : Union[PrimExpr, List[PrimExpr]]
            The indices of the element in the original buffer.

        inner : bool, optional
            If False, the offset is relative to the original buffer.
            Default is True.

        Returns
        -------
        offset: PrimExpr
            The element offset of the buffer at the given indices.
        """
        if inner:
            return _ffi_api.BufferOffsetOfp(self, indices)
        return self.elem_offset + _ffi_api.BufferOffsetOfp(self, indices)

    def byte_offset_of(self, indices, inner=True):
        """Get the byte offset of the buffer at the given indices.
        Note that indices subject to buffer's layout mapping.

        Parameters
        ----------
        indices : Union[PrimExpr, List[PrimExpr]]
            The indices of the element in the original buffer.

        inner : bool, optional
            If False, the offset is relative to the original buffer.
            Default is True.

        Returns
        -------
        offset: PrimExpr
            The byte offset of the buffer at the given indices.
        """
        return self.elem_offset_of(indices, inner) * tvm.DataType(self.dtype).bits // 8

    def is_scalar(self, alloc_or_decl=True):
        """Check if the buffer is a scalar.

        Parameters
        ----------
        alloc_or_decl : bool, optional
            Whether to consider alloc_scalar and decl_scalar as scalar. True for alloc_scalar,
            False for decl_scalar.

        Returns
        -------
            bool: True if the buffer is a scalar, False otherwise.
        """
        return _ffi_api.BufferIsScalar(self, alloc_or_decl)

    def ptr_to(self, indices):
        """Get the pointer to the buffer at the given indices (logical indices).

        Note that the bufferload inside requires LowerTIPp pass to apply the layout to get the physical indices.
        """  # noqa: E501
        assert len(indices) == len(self.shape), (
            f"The number of indices {indices} does not match the shape of the buffer {self.shape}"
        )
        return tvm.tirx.address_of(self[tuple(indices)])

    def view(self, *args, **kwargs) -> "Buffer":
        """Creates a new view of the buffer. (used by parser)

        Supported signatures are ``view(*shape, layout=None)``, where shape can contain
        ``-1`` to indicate that the dimension size is auto-inferred, and
        ``view(dtype: Union[str, tvm.DataType])``.

        Returns
        -------
        view : DeclBufferFrame
            The corresponding view buffer.
        """

        def _infer_shape(shape):
            shape = list(shape)
            if -1 in shape and shape.count(-1) == 1:
                size = functools.reduce(lambda x, y: x * y, self.shape)
                n_size = functools.reduce(lambda x, y: x * y, [s for s in shape if s != -1], 1)
                shape[shape.index(-1)] = size // n_size
            else:
                # Only validate the shape product when both old and new shapes
                # are fully concrete: a PrimExpr `==` returns an `EQ` node, not
                # a Python bool, and `assert <PrimExpr>` raises (no __bool__).
                if all(isinstance(s, int) for s in shape) and all(
                    isinstance(s, int) for s in self.shape
                ):
                    assert functools.reduce(lambda x, y: x * y, shape) == functools.reduce(
                        lambda x, y: x * y, self.shape
                    ), (
                        "The shape of the buffer "
                        + str(self.shape)
                        + " and the new shape "
                        + str(shape)
                        + " are not compatible"
                    )
            return shape

        if len(args) == 1 and isinstance(args[0], str | tvm.DataType) and not kwargs:
            cast_dtype = tvm.DataType(args[0])
            cur_dtype = tvm.DataType(self.dtype)
            if cast_dtype.bits > cur_dtype.bits:
                # cast up
                assert cast_dtype.bits % cur_dtype.bits == 0
                ratio = cast_dtype.bits // cur_dtype.bits
                layout = self.layout.pack(ratio)
                shape = [s for s in self.shape[:-1]] + [self.shape[-1] // ratio]
                new_elem_offset = self.elem_offset // ratio
            else:
                # cast down
                assert cur_dtype.bits % cast_dtype.bits == 0
                ratio = cur_dtype.bits // cast_dtype.bits
                layout = self.layout.unpack(ratio)
                shape = [s for s in self.shape[:-1]] + [self.shape[-1] * ratio]
                new_elem_offset = self.elem_offset * ratio
            return tvm.tirx.script.builder.decl_buffer(
                shape,
                cast_dtype,
                self.data,
                self.strides,
                new_elem_offset,
                None,
                self.scope(),
                self.data_alignment,
                self.offset_factor,
                "",
                self.axis_separators,
                layout,
            )
        else:
            # --- Signature 1: view(*shape, **opts) ---
            # Check if all positional args are integers/PrimExprs with dtype int32 or int64 (the shape)  # noqa: E501
            shape = args
            assert all(
                isinstance(arg, int)
                or (isinstance(arg, PrimExpr) and arg.ty.dtype in ["int32", "int64"])
                for arg in shape
            ), "shape must be a list of integers or PrimExprs with dtype int32 or int64"
            # Safely get optional keyword arguments
            layout = kwargs.get("layout", None)
            # Assert there are no other kwargs
            assert set(kwargs.keys()).issubset({"layout"}), (
                f"Unsupported kwargs for view: {set(kwargs.keys()) - {'layout'}}"
            )

            if layout is None:
                shape = _infer_shape(shape)

            return tvm.tirx.script.builder.decl_buffer(
                shape,
                self.dtype,
                self.data,
                self.strides,
                self.elem_offset,
                None,
                self.scope(),
                self.data_alignment,
                self.offset_factor,
                "",
                self.axis_separators,
                self.layout if layout is None else layout,
            )

    def local(self, *shape, layout=None) -> "Buffer":
        """Create a thread-local view of this buffer.

        When called with no shape arguments, auto-infers a 1D shape from
        the layout's non-thread component (i.e. ``layout.storage().shard``).

        Parameters
        ----------
        shape : tuple of Expr
            The shape of the local view for indexing. If omitted, a 1D
            shape is computed automatically.

        layout : optional
            Override layout. If None, uses the storage layout
            (parent layout with thread axes removed).

        Returns
        -------
        local : DeclBufferFrame
            The corresponding local buffer.
        """
        if not shape:
            local_layout = self.layout.storage()
            total = functools.reduce(
                lambda x, y: x * y, [it.extent for it in local_layout.shard], 1
            )
            shape = (total,)
        return tvm.tirx.script.builder.decl_buffer(
            shape,
            self.dtype,
            self.data,
            self.strides,
            self.elem_offset,
            None,
            self.scope(),
            self.data_alignment,
            self.offset_factor,
            "",
            self.axis_separators,
            self.layout.storage() if layout is None else layout,
        )

    def permute(self, *dims) -> "Buffer":
        """Permute the dimensions of the buffer.

        Parameters
        ----------
        dims : tuple of int
            The permutation of dimensions.

        Returns
        -------
        permuted : DeclBufferFrame
            The buffer with permuted dimensions.
        """
        new_shape = [self.shape[d] for d in dims]
        # Permute *logical* dims, not the layout's fine-grained shard iters: a
        # tcgen05/atom layout maps several shard iters to each logical axis, so
        # group by the current shape first and permute whole groups. ``group``
        # returns a regrouped layout (degenerate extent-1 iters folded away)
        # plus seps over *that* layout — permute the regrouped one, not
        # ``self.layout``. For a simple layout (one shard iter per axis) this
        # reduces to ``permute_dims(dims)``.
        layout = self.layout
        swizzle = None
        if isinstance(layout, tvm.tirx.layout.ComposeLayout):
            # The swizzle permutes the flat offset, so it commutes with a
            # permutation of the logical dims: permute the inner tile layout
            # and re-compose.
            swizzle = layout.swizzle
            layout = layout.tile_layout
        grouped, seps = layout.group(list(self.shape))
        new_layout = grouped.permute_by_groups(seps, list(dims))
        if swizzle is not None:
            new_layout = tvm.tirx.layout.ComposeLayout(swizzle, new_layout)
        return tvm.tirx.script.builder.decl_buffer(
            new_shape,
            self.dtype,
            self.data,
            self.strides,
            self.elem_offset,
            None,
            self.scope(),
            self.data_alignment,
            self.offset_factor,
            "",
            self.axis_separators,
            new_layout,
        )

    # ------------------------------------------------------------------
    # Dimension-surgery views: ``view`` / ``permute`` (reshape / transpose),
    # the numpy-style ``sub`` indexer, ``tile`` (split+pick+merge), and
    # ``chunk`` (equal contiguous split). All return derived views sharing
    # this buffer's data; the physical placement (layout iters + swizzle) is
    # carried, never restated, and data is never moved: an operation either
    # yields a valid view or raises.
    # ------------------------------------------------------------------

    def _normalized_dim(self, dim, name):
        ndim = len(self.shape)
        if dim < 0:
            dim += ndim
        if not 0 <= dim < ndim:
            raise ValueError(f"{name}: dim {dim} out of range for buffer of rank {ndim}")
        return dim

    @staticmethod
    def _concrete_int(value):
        """Return ``value`` as a python int when it is statically known."""
        if isinstance(value, bool):
            return None
        if isinstance(value, Integral):
            return int(value)
        if isinstance(value, tvm.tirx.IntImm):
            return int(value)
        return None

    def _surgery_parts(self):
        """Split ``self.layout`` into (inner tile layout, optional swizzle)."""
        layout = self.layout
        swizzle = None
        if isinstance(layout, tvm.tirx.layout.ComposeLayout):
            swizzle = layout.swizzle
            layout = layout.tile_layout
        return layout, swizzle

    @staticmethod
    def _rewrap_swizzle(layout, swizzle):
        if swizzle is not None:
            return tvm.tirx.layout.ComposeLayout(swizzle, layout)
        return layout

    @staticmethod
    def _dim_group_offset(group, index):
        """Offset of ``index`` along a dim made of the given layout iters.

        The dim's iters are outer→inner in row-major coordinate order:
        decompose ``index`` mixed-radix over their extents and weight each
        coordinate by its iter's stride. A single-iter dim reduces to
        ``index * stride``. ``index`` may be a PrimExpr; multi-iter dims
        need concrete inner extents.
        """
        if len(group) == 1:
            return index * group[0].stride
        extents = []
        for it in group[1:]:
            extent = it.extent
            if not isinstance(extent, tvm.tirx.IntImm):
                raise ValueError(
                    "select/narrow: dim with multiple layout iters requires concrete iter extents"
                )
            extents.append(int(extent))
        offset = None
        remaining = index
        for pos, it in enumerate(group):
            inner = functools.reduce(lambda a, b: a * b, extents[pos:], 1)
            if inner == 1:
                coord = remaining
            elif isinstance(remaining, Integral):
                coord = remaining // inner
                remaining = remaining % inner
            else:
                coord = tvm.tirx.floordiv(remaining, inner)
                remaining = tvm.tirx.floormod(remaining, inner)
            term = coord * it.stride
            offset = term if offset is None else offset + term
        return offset

    def _swizzle_offset_commutes(self, swizzle, extra_offset):
        """Whether ``extra_offset`` commutes with the swizzle permutation.

        Folding an offset into ``elem_offset`` moves it *outside* the layout,
        so the address becomes ``offset + swizzle(rest)``. The swizzle XORs
        bits within a period of ``2^(per_element + atom_len + swizzle_len)``
        elements, so this equals the true ``swizzle(offset + rest)`` only
        when the offset is a multiple of that period.
        """
        if swizzle is None or extra_offset is None:
            return True
        sw_len = int(swizzle.swizzle_len)
        if sw_len == 0:
            return True  # identity permutation, everything commutes
        period = 1 << (int(swizzle.per_element) + int(swizzle.atom_len) + sw_len)
        offset_c = self._concrete_int(extra_offset)
        if offset_c is not None:
            return offset_c % period == 0
        from ..arith import Analyzer  # pylint: disable=import-outside-toplevel

        return Analyzer().can_prove_equal(tvm.tirx.floormod(extra_offset, period), 0)

    def _rebuild_view(self, new_shape, new_shard, grouped, swizzle, extra_offset):
        """Rebuild a derived view; ``extra_offset`` goes into ``elem_offset``
        when it commutes with the swizzle (provable period multiple),
        otherwise it stays inside the tile layout's offset so the swizzle
        keeps applying to it and the view addresses the same bytes."""
        offset_map = dict(grouped.offset.items())
        if extra_offset is not None and not self._swizzle_offset_commutes(swizzle, extra_offset):
            m_axis = tvm.tirx.layout.Axis.get("m")
            prev = offset_map.get(m_axis)
            offset_map[m_axis] = extra_offset if prev is None else prev + extra_offset
            extra_offset = None
        new_layout = tvm.tirx.layout.TileLayout.from_iters(
            new_shard, list(grouped.replica), offset_map
        )
        new_layout = self._rewrap_swizzle(new_layout, swizzle)
        elem_offset = self.elem_offset
        if extra_offset is not None:
            elem_offset = elem_offset + extra_offset
        return tvm.tirx.script.builder.decl_buffer(
            new_shape,
            self.dtype,
            self.data,
            self.strides,
            elem_offset,
            None,
            self.scope(),
            self.data_alignment,
            self.offset_factor,
            "",
            self.axis_separators,
            new_layout,
        )

    @property
    def sub(self) -> "_SubIndexer":
        """Numpy-style view indexer: ``buf.sub[2, 4:8, ::4]``.

        Unlike plain ``buf[...]`` (BufferLoad for scalar indices, extent-1
        BufferRegion dims for tile-primitive operands), ``sub`` follows numpy
        basic-indexing semantics as a *view constructor*: an integer index
        removes the dim (``select``), ``a:b`` narrows it, and ``a::s`` takes
        every s-th element (requires the extent divisible by ``s`` and
        ``a < s``). Trailing dims are kept whole.
        """
        return _SubIndexer(self)

    def tile(self, *specs) -> "_TileIndexer":
        """Chunk a dim: split it into factors, pick a chunk, keep the rest.

        Rank-preserving — the picked dim's remaining factors merge back into
        that one dim, and every other dim is untouched, so N dims in gives N
        dims out. Chunk multiple dims by chaining (dims never shift):
        ``buf.tile(0, (nx, -1))[cx, :].tile(1, (-1, ny))[:, cy]``.

        Call as ``tile(dim, factors)`` for one dim, or pass several
        ``(dim, factors)`` specs as sugar for a chain. ``factors`` is the
        tuple the dim splits into (row-major, like :meth:`unflatten`; one
        ``-1`` inferred). The indexer takes one entry per factor: an ``int`` /
        ``PrimExpr`` **picks** it (fixing the chunk, dropping the axis, folding
        its offset) and ``:`` **keeps** it. At least one factor per dim must be
        picked — a pure keep-everything split is :meth:`unflatten`, not a
        chunk::

            # 64 rows split into (stripe, warp, row) = (-1, WARPS, 4); this
            # warp's 16 interleaved rows (stripe x row merged):
            buf.tile(1, (-1, WARPS, 4))[:, warp, :]

            tile(d, (n, -1))[c, :]   # contiguous block c
            tile(d, (-1, n))[:, c]   # round-robin chunk c

        A picked index may be a dynamic PrimExpr (e.g. a warp id); picking
        several factors of one dim is allowed.
        """
        if specs and isinstance(specs[0], int | Integral):
            if len(specs) != 2:
                raise ValueError("tile(dim, factors) takes a dim and a factors tuple")
            specs = (specs,)  # single-dim positional form
        norm = []
        seen = set()
        for spec in specs:
            if not isinstance(spec, tuple | list) or len(spec) != 2:
                raise ValueError(f"tile: each spec must be (dim, factors), got {spec!r}")
            dim = self._normalized_dim(spec[0], "tile")
            if dim in seen:
                raise ValueError(f"tile: dim {dim} tiled more than once")
            seen.add(dim)
            factors = spec[1]
            if not isinstance(factors, tuple | list) or len(factors) < 1:
                raise ValueError(f"tile: factors for dim {dim} must be a non-empty tuple")
            norm.append((dim, tuple(factors)))
        return _TileIndexer(self, norm)

    def chunk(self, spec) -> "_ChunkIndexer":
        """Split dims into equal contiguous chunks and pick a chunk per dim —
        **rank-preserving**. Index the result with ``[picks]``.

        ``spec`` is a per-dim tuple (length = rank). Each entry is ``None``
        (leave the dim) or a positive int ``n`` (split that dim, extent ``E``
        with ``E % n == 0``, into ``n`` equal chunks of ``E // n``). Then
        ``chunk(spec)[picks]`` takes one entry per dim: a chunked dim's pick is
        the chunk index (int / PrimExpr) and **narrows that dim** to the chunk's
        ``[c*E//n : (c+1)*E//n)`` range — the dim is kept at ``E // n``, no
        dimension is added; an unchunked dim's pick is a normal index (``:`` /
        int / slice). The result is the *same BufferRegion* as the hand-written
        slice — one line instead of the ``c*k : (c+1)*k`` arithmetic::

            X[.., c * k : (c + 1) * k, ..]        # before (k = E // n)
            X.chunk((None, .., n, ..))[.., c, ..]  # after (k inferred)
        """
        if not isinstance(spec, tuple | list):
            raise ValueError(f"chunk: spec must be a per-dim tuple, got {spec!r}")
        if len(spec) != len(self.shape):
            raise ValueError(f"chunk: spec length {len(spec)} != buffer rank {len(self.shape)}")
        for d, n in enumerate(spec):
            if n is not None and (not isinstance(n, Integral) or int(n) < 1):
                raise ValueError(f"chunk: spec[{d}] must be None or a positive int, got {n!r}")
        return _ChunkIndexer(self, tuple(spec))

    def __getitem__(self, indices):
        from ..arith import Analyzer  # pylint: disable=import-outside-toplevel
        from .expr import BufferLoad, Ramp  # pylint: disable=import-outside-toplevel
        from .stmt import BufferRegion  # pylint: disable=import-outside-toplevel

        if not isinstance(indices, tuple | list):
            indices = [indices]
        has_slice = any(isinstance(i, slice) for i in indices)
        has_step = any(
            isinstance(i, slice) and (i.step is not None and i.step != 1) for i in indices
        )
        has_implicit_slice = len(indices) < len(self.shape)
        analyzer = Analyzer()
        if (has_slice and not has_step) or has_implicit_slice:
            region = []
            for i, index in enumerate(indices):
                if isinstance(index, slice):
                    start = 0 if index.start is None else index.start
                    stop = self.shape[i] if index.stop is None else index.stop
                    region.append(Range.from_min_extent(start, analyzer.simplify(stop - start)))
                else:
                    region.append(
                        Range.from_min_extent(
                            index,
                            tvm.tirx.expr.IntImm(index.ty, 1) if isinstance(index, PrimExpr) else 1,
                        )
                    )
            if has_implicit_slice:
                for i in range(len(indices), len(self.shape)):
                    region.append(Range.from_min_extent(0, self.shape[i]))
            return BufferRegion(self, region)
        else:
            expr_indices = []
            for i, index in enumerate(indices):
                if isinstance(index, slice):
                    start = 0 if index.start is None else index.start
                    stop = self.shape[i] if index.stop is None else index.stop
                    step = 1 if index.step is None else index.step
                    # We should ensure the dtype of start is the same with that of step.
                    if isinstance(start, tvm.tirx.expr.PrimExpr) and isinstance(step, int):
                        step = tvm.tirx.expr.IntImm(start.ty, step)
                    lanes = analyzer.simplify((stop - start + step - 1) // step)
                    if lanes == 1:
                        expr_indices.append(start)
                    else:
                        expr_indices.append(Ramp(start, step, int(lanes)))
                else:
                    expr_indices.append(index)
            return BufferLoad(self, expr_indices)


def _view_drop(buf, dim, index):
    """Internal machinery for ``sub`` (int index) / ``tile`` (factor pick): a
    view with ``dim`` removed, fixed at ``index`` — the offset folds into
    elem_offset through the dim's layout iters. Not a public method."""
    dim = buf._normalized_dim(dim, "index")
    index_c = buf._concrete_int(index)
    extent_c = buf._concrete_int(buf.shape[dim])
    if index_c is not None and (index_c < 0 or (extent_c is not None and index_c >= extent_c)):
        raise ValueError(
            f"index {index_c} out of range for dim {dim} "
            f"of extent {extent_c if extent_c is not None else buf.shape[dim]}"
        )
    layout, swizzle = buf._surgery_parts()
    grouped, seps = layout.group(list(buf.shape))
    iters = list(grouped.shard)
    lo, hi = seps[dim], seps[dim + 1]
    offset = buf._dim_group_offset(iters[lo:hi], index)
    new_shape = list(buf.shape[:dim]) + list(buf.shape[dim + 1 :])
    new_shard = iters[:lo] + iters[hi:]
    return buf._rebuild_view(new_shape, new_shard, grouped, swizzle, offset)


def _view_narrow(buf, dim, start, length):
    """Internal machinery for ``sub`` (slice index): a view of ``dim`` narrowed
    to ``[start, start + length)`` — the offset folds into elem_offset. Multi-iter
    dims need start/length on the inner iter block. Not a public method."""
    dim = buf._normalized_dim(dim, "index")
    start_c = buf._concrete_int(start)
    length_c = buf._concrete_int(length)
    extent_c = buf._concrete_int(buf.shape[dim])
    if start_c is not None and start_c < 0:
        raise ValueError(f"sub: start {start_c} must be non-negative")
    if length_c is not None and length_c < 1:
        raise ValueError(f"sub: length {length_c} must be positive")
    if (
        start_c is not None
        and length_c is not None
        and extent_c is not None
        and start_c + length_c > extent_c
    ):
        raise ValueError(
            f"sub: range [{start_c}, {start_c + length_c}) exceeds dim {dim} extent {extent_c}"
        )
    layout, swizzle = buf._surgery_parts()
    grouped, seps = layout.group(list(buf.shape))
    iters = list(grouped.shard)
    lo, hi = seps[dim], seps[dim + 1]
    group = iters[lo:hi]
    Iter = tvm.tirx.layout.Iter
    if len(group) == 1:
        it = group[0]
        offset = start * it.stride
        new_group = [Iter(length, it.stride, it.axis)]
    else:
        inner = 1
        for it in group[1:]:
            extent = it.extent
            if not isinstance(extent, tvm.tirx.IntImm):
                raise ValueError(
                    "sub: dim with multiple layout iters requires concrete iter extents"
                )
            inner *= int(extent)
        if not (isinstance(start, Integral) and isinstance(length, Integral)):
            raise ValueError(
                f"sub: dim {dim} is made of {len(group)} layout iters; "
                f"start/length must be concrete multiples of {inner}"
            )
        if start % inner != 0 or length % inner != 0:
            raise ValueError(
                f"sub: dim {dim} start/length must be multiples of the "
                f"inner iter block {inner}; got start={start} length={length}"
            )
        outer = group[0]
        offset = (start // inner) * outer.stride
        new_group = [Iter(length // inner, outer.stride, outer.axis), *group[1:]]
    new_shape = list(buf.shape)
    new_shape[dim] = length
    new_shard = iters[:lo] + new_group + iters[hi:]
    return buf._rebuild_view(new_shape, new_shard, grouped, swizzle, offset)


class _SubIndexer:
    """Indexer object returned by :attr:`Buffer.sub`.

    Translates numpy basic indexing into ``_view_drop`` / ``_view_narrow`` /
    ``view`` view operations (int drops the dim, ``a:b`` narrows, ``a::s`` strides).
    """

    def __init__(self, buffer: Buffer):
        self._buffer = buffer

    def __getitem__(self, indices) -> Buffer:
        if not isinstance(indices, tuple):
            indices = (indices,)
        buf = self._buffer
        if len(indices) > len(buf.shape):
            raise ValueError(f"sub: {len(indices)} indices for buffer of rank {len(buf.shape)}")
        dim = 0
        for item in indices:
            if isinstance(item, slice):
                step = item.step
                if step is None or (isinstance(step, Integral) and step == 1):
                    if item.start is None and item.stop is None:
                        dim += 1
                        continue
                    start = 0 if item.start is None else item.start
                    stop = buf.shape[dim] if item.stop is None else item.stop
                    buf = _view_narrow(buf, dim, start, stop - start)
                    dim += 1
                else:
                    if not isinstance(step, Integral) or step <= 0:
                        raise ValueError(f"sub: step must be a positive int, got {step!r}")
                    if item.stop is not None:
                        raise ValueError("sub: a stop bound with a step is not supported")
                    extent = buf.shape[dim]
                    if not isinstance(extent, tvm.tirx.IntImm):
                        raise ValueError("sub: a stepped slice requires a concrete dim extent")
                    extent = int(extent)
                    if extent % step != 0:
                        raise ValueError(
                            f"sub: dim extent {extent} is not divisible by step {step}"
                        )
                    start = 0 if item.start is None else item.start
                    # a::s over N = split into (N//s, s) and fix the remainder
                    # coordinate at a (requires 0 <= a < s).
                    start_c = Buffer._concrete_int(start)
                    if start_c is not None and not 0 <= start_c < step:
                        raise ValueError(
                            f"sub: stepped-slice start {start_c} must be in [0, {step})"
                        )
                    split = buf.view(*buf.shape[:dim], extent // step, step, *buf.shape[dim + 1 :])
                    buf = _view_drop(split, dim + 1, start)
                    dim += 1
            else:
                buf = _view_drop(buf, dim, item)
        return buf


class _ChunkIndexer:
    """Indexer returned by :meth:`Buffer.chunk`.

    ``[picks]`` picks a chunk per chunked dim — narrowing that dim to the
    chunk's ``[c*k : (c+1)*k)`` range (``k = E // n``, rank-preserving) — and
    passes unchunked-dim picks through as ordinary indices, yielding the same
    ``BufferRegion`` as the hand-written ``c*k:(c+1)*k`` slice.
    """

    def __init__(self, buffer: Buffer, spec):
        self._buffer = buffer
        self._spec = spec

    def __getitem__(self, picks):
        if not isinstance(picks, tuple):
            picks = (picks,)
        buf = self._buffer
        if len(picks) > len(self._spec):
            raise ValueError(
                f"chunk: {len(picks)} indices for a rank-{len(self._spec)} spec"
            )
        translated = []
        for d, pick in enumerate(picks):
            n = self._spec[d]
            if n is None:
                translated.append(pick)
            elif isinstance(pick, slice):
                raise ValueError(
                    f"chunk: chunked dim {d} takes a chunk index, not a slice; got {pick!r}"
                )
            else:
                k = buf.shape[d] // int(n)
                translated.append(slice(pick * k, (pick + 1) * k))
        return buf[tuple(translated)]


class _TileIndexer:
    """Indexer object returned by :meth:`Buffer.tile`.

    Holds ``(dim, factors)`` specs; ``[...]`` takes one entry per factor
    (``int`` / ``PrimExpr`` picks and drops it, ``:`` keeps it). Each dim is
    processed as ``view`` (split) -> per-factor ``_view_drop`` -> a single
    ``view`` (merge) of the kept factors. Dims are processed high-to-low so a
    fully-picked (rank-reducing) dim never shifts a lower dim's index.
    """

    def __init__(self, buffer: Buffer, specs):
        self._buffer = buffer
        self._specs = specs

    def __getitem__(self, picks) -> Buffer:
        if not isinstance(picks, tuple):
            picks = (picks,)
        n_factors = sum(len(factors) for _, factors in self._specs)
        if len(picks) != n_factors:
            raise ValueError(
                f"tile: split into {n_factors} factor(s) but {len(picks)} "
                f"index(es) given (int/PrimExpr picks, ':' keeps)"
            )
        # Distribute the flat index list across specs, left to right.
        groups, pos = [], 0
        for dim, factors in self._specs:
            groups.append((dim, factors, picks[pos : pos + len(factors)]))
            pos += len(factors)
        buf = self._buffer
        for dim, factors, dim_picks in sorted(groups, key=lambda g: g[0], reverse=True):
            buf = self._apply(buf, dim, factors, dim_picks)
        return buf

    @staticmethod
    def _is_keep(p):
        return isinstance(p, slice) and p.start is None and p.stop is None and p.step is None

    @classmethod
    def _apply(cls, buf, dim, factors, dim_picks):
        for p in dim_picks:
            if isinstance(p, slice) and not cls._is_keep(p):
                raise ValueError("tile: a factor slice must be ':' (keep whole); got a sub-slice")
        n_keep = sum(1 for p in dim_picks if cls._is_keep(p))
        if n_keep == len(dim_picks):
            raise ValueError(
                f"tile: dim {dim} picks no factor (all ':'); a chunk must pick "
                f"at least one factor. Use .view / .chunk for a pure reshape."
            )
        # Split `dim` into `factors` (view infers the -1).
        buf = buf.view(*buf.shape[:dim], *factors, *buf.shape[dim + 1 :])
        # Drop picked factors from the last to the first so earlier dim indices
        # stay valid; kept factors then sit contiguously starting at ``dim``.
        for i in reversed(range(len(dim_picks))):
            if not cls._is_keep(dim_picks[i]):
                buf = _view_drop(buf, dim + i, dim_picks[i])
        # Merge the kept factors back into the one dim (rank-preserving).
        if n_keep >= 2:
            buf = buf.view(*buf.shape[:dim], -1, *buf.shape[dim + n_keep :])
        return buf


def decl_buffer(
    shape,
    dtype=None,
    name="buffer",
    data=None,
    strides=None,
    elem_offset=None,
    scope="",
    data_alignment=-1,
    offset_factor=0,
    buffer_type="",
    axis_separators=None,
    span=None,
    layout="default",
):
    # pylint: disable=import-outside-toplevel
    from .expr import Var
    from .layout import S, TileLayout

    shape = (shape,) if isinstance(shape, PrimExpr | Integral) else shape
    dtype = "float32" if dtype is None else dtype
    strides = () if strides is None else strides

    if axis_separators is None:
        axis_separators = []

    if layout == "default":
        layout = TileLayout(S[tuple(shape)]) if shape else None

    if offset_factor != 0 and elem_offset is None:
        shape_ty = shape[0].ty if shape and isinstance(shape[0], PrimExpr) else "int32"
        elem_offset = Var(f"{name}_elem_offset", shape_ty)
    if data is None:
        # Bool is represented as uint1 in the IR, but stored as int8
        storage_type = dtype if isinstance(dtype, PrimType) else PrimType(dtype)
        storage_type = PrimType("int8") if storage_type.dtype == "bool" else storage_type
        data = Var(name, PointerType(storage_type, scope), span)
    return _ffi_api.Buffer(  # type: ignore
        data,
        dtype,
        shape,
        strides,
        elem_offset,
        name,
        data_alignment,
        offset_factor,
        buffer_type,
        axis_separators,
        span,
        layout,
    )


@tvm_ffi.register_object("tirx.DataProducer")
class DataProducer(Object):
    pass
