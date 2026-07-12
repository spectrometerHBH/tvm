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

## Answering "does dispatch support X?"

Never conclude a primitive or case is unsupported from a name grep. The
support surface is defined by:

1. the `register_dispatch(...)` sites and their `predicate` functions in
   this directory (the executable ground truth);
2. the layout thread axes (`Axis.tid_in_wg` / `laneid` / `wid_in_wg`,
   `python/tvm/tirx/layout.py`) — local buffers with thread-axis TileLayouts
   are distributed register tiles (frags) and are first-class copy operands.

Check both before answering a support question or advising a kernel author
that something needs a new primitive.

## Rules for changes in this directory

1. **Every fix needs a regression unit test** under
   `tests/python/tirx/operator/tile_primitive/cuda/`.
2. **Validation-only changes must be byte-identity checked**: compile a
   representative kernel before and after in separate processes (the
   in-process compile cache poisons A/B comparisons) and diff the generated
   artifacts.
3. **Performance claims go through bench-suite** (in the tirx-kernels repo),
   not ad-hoc timing.
