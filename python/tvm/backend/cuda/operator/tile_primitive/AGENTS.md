<!--
Licensed to the Apache Software Foundation (ASF) under one
or more contributor license agreements.  See the NOTICE file
distributed with this work for additional information
regarding copyright ownership.  The ASF licenses this file
to you under the Apache License, Version 2.0 (the
"License"); you may not use this file except in compliance
with the License.  You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on an
"AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
KIND, either express or implied.  See the License for the
specific language governing permissions and limitations
under the License.
-->

# Tile-primitive dispatch: agent guidance

The dispatch logic in this directory is covered by written correctness
proofs. The proofs are normative: they state the invariants the dispatchers
rely on and why the emitted IR is semantically equal to the primitive's
declared semantics.

## Required reading before editing dispatch logic

Before modifying any of the following files, read the matching proof in
`.agents/docs/` (repo root) end to end:

| Dispatch file | Correctness proof |
| --- | --- |
| `copy_async/tcgen05_cp.py` | `.agents/docs/tcgen05_cp_dispatch_correctness_proof.md` |
| `copy_async/tma.py`, `tma_utils.py` | `.agents/docs/tma_dispatch_correctness_proof.md` |
| `gemm_async/tcgen05.py` (and the instruction-descriptor helpers in `../intrinsics/tcgen05.py`) | `.agents/docs/gemm_async_instr_desc_correctness_proof.md` |

## Rules for changes in this directory

1. **Check the change against the proof.** If your change touches a step,
   lemma, or invariant in the proof, verify the argument still holds. Do not
   land a change that contradicts the proof without updating the proof.
2. **Keep the proofs in sync.** Any behavioral change to dispatch (new
   shapes, new validation, changed descriptor encoding, changed planning
   order) must be reflected in the corresponding proof, including its
   `file:line` references.
3. **Every fix needs a regression unit test** under
   `tests/python/tirx/operator/tile_primitive/cuda/`.
4. **Validation-only changes must be byte-identity checked**: compile a
   representative kernel before and after in separate processes (the
   in-process compile cache poisons A/B comparisons) and diff the generated
   artifacts.
5. **Performance claims go through bench-suite** (in the tirx-kernels repo),
   not ad-hoc timing.
