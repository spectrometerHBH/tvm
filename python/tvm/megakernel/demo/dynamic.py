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
"""Dynamic-scheduling demo: runtime-scalar-dispatched marker kernel.

A host-planted count drives the grid of the ``mark`` tile at kernel runtime:
``plant`` runs once, then ``mark`` is dynamically dispatched ``count`` times
via the MPMC push synthesized from the ``ready`` event edge, each instance
writing ``out[m] = m + 1``.  The terminal ``mark`` tile gets a synthesized
drain event (runtime-initialized by its pusher) whose last pre-notify pushes
the END tasks.
"""

from tvm.megakernel.dsl import KernelSpec, TileImpl
from tvm.megakernel.transform import LoweringOptions, lower_to_tirx_module
from tvm.script import tirx as T

UPPER = 64


class PlantTile(TileImpl):
    """Single seed task; the builder turns it into the mark tile's pusher."""

    def run(self, m_idx, n_idx, k_idx):
        pass


class MarkTile(TileImpl):
    """Mark ``out[m_idx]`` with its one-based task index."""

    def __init__(self, out):
        super().__init__()
        self.out = out

    def run(self, m_idx, n_idx, k_idx):
        T.buffer_store(self.out, m_idx + 1, [m_idx])


def build_dynamic_count_spec(upper=UPPER):
    """Build the runtime-scalar dispatch demo spec for one output capacity."""

    kernel = KernelSpec(
        "dynamic_count",
        attrs={
            "source": "out[m] = m + 1 for m < count[0] (dynamic dispatch)",
        },
    )
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (upper,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, upper))
    ready = kernel.event(
        "ready",
        shape=(1,),
        init_count=1,
        attrs={"meaning": "the plant task has run; mark tasks may start"},
    )
    kernel.tile(
        name="plant",
        impl=PlantTile(),
        tile_num=(1, 1, 1),
        reads=[count_buf],
    ).notify(ready, coord_map=lambda m, n, k: (0,))
    kernel.tile(
        name="mark",
        impl=MarkTile(out),
        tile_num=(n_tiles, 1, 1),
        writes=[out],
    ).wait(ready, coord_map=lambda m, n, k: (0,))
    return kernel


kernel = build_dynamic_count_spec()


class WriteCountTile(TileImpl):
    """Compute the dispatch count from input data at kernel runtime."""

    def __init__(self, in_vals, count):
        super().__init__()
        self.in_vals = in_vals
        self.count = count

    def run(self, m_idx, n_idx, k_idx):
        T.buffer_store(self.count, self.in_vals[0] + self.in_vals[1], [0])


def build_case_b_spec(upper=UPPER):
    """Case-B demo: the pusher tile itself produces the dispatch count.

    The writer tile computes ``count[0]`` from ``in_vals`` during its run and
    pushes the scalar-grid ``mark`` tile; the builder must move the push after
    the writer's run (the trigger then implies the write is complete).
    """

    kernel = KernelSpec(
        "dynamic_case_b",
        attrs={
            "source": "count[0] = in_vals[0] + in_vals[1]; out[m] = m + 1 for m < count[0]",
        },
    )
    in_vals = kernel.tensor("in_vals", (2,), "int32")
    count_buf = kernel.tensor("count", (1,), "int32")
    out = kernel.tensor("out", (upper,), "int32")
    n_tiles = kernel.scalar("n_tiles", source=(count_buf, (0,)), range=(1, upper))
    ready = kernel.event(
        "ready",
        shape=(1,),
        init_count=1,
        attrs={"meaning": "the writer task has published the count"},
    )
    kernel.tile(
        name="writer",
        impl=WriteCountTile(in_vals, count_buf),
        tile_num=(1, 1, 1),
        reads=[in_vals],
        writes=[count_buf],
    ).notify(ready, coord_map=lambda m, n, k: (0,))
    kernel.tile(
        name="mark",
        impl=MarkTile(out),
        tile_num=(n_tiles, 1, 1),
        writes=[out],
    ).wait(ready, coord_map=lambda m, n, k: (0,))
    return kernel


# Verify the spec and build the dynamic kernel from the runtime-library
# builder; the host-side MPMC seed arrays come from build_runtime_kernel.
if __name__ == "__main__":
    print(lower_to_tirx_module(kernel.validate(), LoweringOptions(scheduler="dynamic")).script())
