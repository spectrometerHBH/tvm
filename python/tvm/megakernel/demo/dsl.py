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

from tvm.megakernel.dsl import KernelSpec, SmemManager, TileImpl
from tvm.megakernel.transform import LoweringOptions, lower_to_tirx_module
from tvm.tirx.script import tile as Tx

# Configuration parameters
M = 1024
N = 1024

BLOCK_M = 64
BLOCK_N = 64

NUM_BLOCK_M = M // BLOCK_M
NUM_BLOCK_N = N // BLOCK_N


# ============================================================
# Lower DSL layer: tile implementations
# ============================================================


class Stage1ReduceTile(TileImpl):
    """Reduce one A[m-block, n-block] tile into B[m-block, n]."""

    def __init__(self, A, B, *, block_m, block_n):
        super().__init__()
        self.A = A
        self.B = B
        self.block_m = block_m
        self.block_n = block_n

        self.A_smem = None
        self.P_smem = None

    def device_init(self, smem_manager: SmemManager, m_idx, n_idx, k_idx):
        self.smem_manager = smem_manager
        self.A_smem = smem_manager.alloc((self.block_m, self.block_n), "float32")
        self.P_smem = smem_manager.alloc((self.block_m, 1), "float32")

    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.acquire_all("cta")
        Tx.copy(
            self.A_smem,
            self.A[
                m_idx * self.block_m : (m_idx + 1) * self.block_m,
                n_idx * self.block_n : (n_idx + 1) * self.block_n,
            ],
        )
        Tx.sum(
            self.P_smem,
            self.A_smem,
        )
        Tx.copy(
            self.B[
                m_idx * self.block_m : (m_idx + 1) * self.block_m,
                n_idx,
            ],
            self.P_smem,
        )
        self.smem_manager.release_all("cta")
        self.smem_manager.advance()


class Stage2ReduceTile(TileImpl):
    """Reduce all B n-block results for one m-block into C[m-block, 0]."""

    def __init__(self, B, C, *, block_m, num_block_n):
        super().__init__()
        self.B = B
        self.C = C
        self.block_m = block_m
        self.num_block_n = num_block_n

        self.B_smem = None
        self.C_smem = None

    def device_init(self, smem_manager: SmemManager, m_idx, n_idx, k_idx):
        self.smem_manager = smem_manager
        self.B_smem = smem_manager.alloc((self.block_m, self.num_block_n), "float32")
        self.C_smem = smem_manager.alloc((self.block_m, 1), "float32")

    def run(self, m_idx, n_idx, k_idx):
        self.smem_manager.acquire_all("cta")
        Tx.copy(
            self.B_smem,
            self.B[
                m_idx * self.block_m : (m_idx + 1) * self.block_m,
                :,
            ],
        )
        Tx.sum(
            self.C_smem,
            self.B_smem,
        )
        Tx.copy(
            self.C[
                m_idx * self.block_m : (m_idx + 1) * self.block_m,
                0,
            ],
            self.C_smem,
        )
        self.smem_manager.release_all("cta")
        self.smem_manager.advance()


# ============================================================
# Upper DSL layer: megakernel spec
# ============================================================


def build_kernel_spec(m=M, n=N, block_m=BLOCK_M, block_n=BLOCK_N):
    """Build the two-stage reduce spec for one problem size."""

    num_block_m = m // block_m
    num_block_n = n // block_n
    kernel = KernelSpec(
        "two_stage_reduce",
        attrs={
            "source": "B = reduce_each_n_block(A); C = reduce_all_n_blocks(B)",
        },
    )

    A = kernel.tensor(
        "A",
        shape=(m, n),
        dtype="float32",
    )

    B = kernel.tensor(
        "B",
        shape=(m, num_block_n),
        dtype="float32",
    )

    C = kernel.tensor(
        "C",
        shape=(m, 1),
        dtype="float32",
    )

    row_ready = kernel.event(
        "row_ready",
        shape=(num_block_m,),
        init_count=num_block_n,
        attrs={
            "meaning": "all n-block reductions for one m-block have written B",
        },
    )

    kernel.tile(
        name="stage1_partial_reduce",
        impl=Stage1ReduceTile(
            A,
            B,
            block_m=block_m,
            block_n=block_n,
        ),
        tile_num=(num_block_m, num_block_n, 1),
        reads=[A],
        writes=[B],
        attrs={
            "source_stage": "B = reduce_each_n_block(A)",
        },
    ).notify(row_ready, coord_map=lambda m, n, k: (m,))

    kernel.tile(
        name="stage2_final_reduce",
        impl=Stage2ReduceTile(
            B,
            C,
            block_m=block_m,
            num_block_n=num_block_n,
        ),
        tile_num=(num_block_m, 1, 1),
        reads=[B],
        writes=[C],
        attrs={
            "source_stage": "C = reduce_all_n_blocks(B)",
        },
    ).wait(row_ready, coord_map=lambda m, n, k: (m,))

    return kernel


kernel = build_kernel_spec()

# Verify the spec and build the static kernel from the runtime-library
# builder; the host-side exec queue comes from build_runtime_kernel.
if __name__ == "__main__":
    print(lower_to_tirx_module(kernel.validate(), LoweringOptions(scheduler="static")).script())
