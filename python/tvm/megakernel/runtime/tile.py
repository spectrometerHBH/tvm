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
"""Shared tile/scheduler plumbing for the megakernel runtime.

There is deliberately no tile ABC here: the runtime framework manages
``tvm.megakernel.dsl.TileImpl`` objects directly.  This module carries only
the mbarrier helper (``Barriers``) and the scheduler interface
(``TileSchedulerBase``) migrated from the production megakernel utils.
"""

from __future__ import annotations

from tvm.script import tirx as T
from tvm.tirx import Expr


@T.meta_class
class Barriers:
    """Mbarrier wrapper class"""

    def __init__(self, smem_manager, pipe_depth, is_p2c, persistent=True):
        self.smem_manager = smem_manager
        self.hardware = smem_manager.hardware
        self.init_phase = 0 if is_p2c else 1
        self.pipe_depth = pipe_depth
        self.persistent = persistent

    def _alloc(self):
        self.mbar = self.smem_manager.alloc(
            (self.pipe_depth,), "uint64", policy="persistent" if self.persistent else "shared"
        )

    @T.inline
    def init(self, threads_num_wait):
        tid = T.thread_id([self.hardware.num_threads])
        self._alloc()
        if self.pipe_depth == 1:
            if tid == 0:
                T.ptx.mbarrier.init(self.mbar.ptr_to([0]), threads_num_wait)
        elif tid == 0:
            for i in T.serial(self.pipe_depth):
                T.ptx.mbarrier.init(self.mbar.ptr_to([i]), threads_num_wait)

    @T.inline
    def wait(self, idx, phase):
        T.ptx.mbarrier.try_wait(self.mbar.ptr_to([idx]), self.init_phase ^ phase)


@T.meta_class
class TileSchedulerBase:
    """Abstract base class for tile schedulers."""

    MAX_TASKS = 128

    def __init__(self):
        pass

    def get_idx_and_task_type(self) -> tuple[list[Expr], Expr]:
        raise NotImplementedError

    @T.inline
    def init(self):
        raise NotImplementedError

    @T.inline
    def next_tile(self):
        raise NotImplementedError

    @T.inline
    def wait(self, evt, *coord, wait_level, mask):
        raise NotImplementedError

    @T.inline
    def notify(self, evt, func_notify, scope, scope_id, release):
        raise NotImplementedError

    @T.inline
    def pre_notify_and_push(self, evt, func_notify, func_trigger_list, push_level, scope, scope_id):
        pass

    def valid(self):
        raise NotImplementedError


__all__ = ["Barriers", "TileSchedulerBase"]
