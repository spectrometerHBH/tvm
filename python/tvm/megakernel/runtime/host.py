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
"""Host-side queue building helpers for the megakernel runtime.

These are the workload-independent queue builders: they know the task wire
format and the per-SM round-robin layout of the static central queue, but
nothing about any concrete kernel's task graph.
"""

from __future__ import annotations

import numpy as np

from .config import HardwareConfig
from .packing import pack_into_32bit
from .static_scheduler import StaticTileScheduler


def build_static_exec_queue(
    central_queue,
    *,
    sm_count: int | None = None,
    max_tasks: int = StaticTileScheduler.MAX_TASKS,
    end_task_type: int = 31,
    packing=None,
):
    """Distribute a central task list over the per-SM static exec queues.

    ``central_queue`` is a list of ``(m_idx, n_idx, k_idx, task_type)`` tuples
    in execution order.  Tasks are dealt round-robin into a
    ``(sm_count, max_tasks)`` int32 array; whenever the list runs out, the
    remaining slots of the row and one final row per SM are filled with the
    end marker.  Mirrors the production static queue construction.
    """

    if sm_count is None:
        sm_count = HardwareConfig().sm_count
    exec_queue = np.zeros((sm_count, max_tasks), dtype=np.int32)
    central_queue = list(central_queue)
    tile_idx = 0
    while central_queue:
        for bx in range(sm_count):
            if central_queue:
                exec_queue[bx, tile_idx] = pack_into_32bit(*central_queue.pop(0), packing=packing)
            else:
                exec_queue[bx, tile_idx] = pack_into_32bit(
                    -1, -1, -1, end_task_type, packing=packing
                )
        tile_idx += 1
    for bx in range(sm_count):
        exec_queue[bx, tile_idx] = pack_into_32bit(-1, -1, -1, end_task_type, packing=packing)
    return exec_queue


__all__ = ["build_static_exec_queue"]
