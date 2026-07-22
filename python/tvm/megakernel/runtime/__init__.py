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
"""Parser-style TIRX runtime building blocks for megakernel emission.

This package is the migration of the production hand-written megakernel's
runtime library (``tirx_kernels.megakernel.utils``).  It manages
``tvm.megakernel.dsl.TileImpl`` objects and emits the same TIRX as the
production code, with the hardware constants parameterized through
``HardwareConfig`` (B200 production values as defaults).
"""

from .config import HardwareConfig, RuntimeProfileEvent
from .device import (
    any_sync,
    atomic_add_int32,
    f_init_const,
    gt,
    stg,
    sts,
    while_ld_global_acquire,
)
from .dynamic_scheduler import (
    DynamicTileScheduler,
    MPMCQueue,
    MPMCQueueHost,
    SchedulerBarrier,
)
from .dynamic_scheduler import Semaphore as DynamicSemaphore
from .host import build_static_exec_queue
from .packing import (
    TaskPacking,
    pack_into_32bit,
    unpack_from_32bit,
    unpack_from_32bit_host,
)
from .semaphore import SemaphoreBase
from .smem import SmemManager
from .static_scheduler import Semaphore as StaticSemaphore
from .static_scheduler import StaticTileScheduler
from .tile import Barriers, TileSchedulerBase
from .wrapper import MegaKernelWrapper

__all__ = [
    "Barriers",
    "DynamicSemaphore",
    "DynamicTileScheduler",
    "HardwareConfig",
    "MPMCQueue",
    "MPMCQueueHost",
    "MegaKernelWrapper",
    "RuntimeProfileEvent",
    "SchedulerBarrier",
    "SemaphoreBase",
    "SmemManager",
    "StaticSemaphore",
    "StaticTileScheduler",
    "TaskPacking",
    "TileSchedulerBase",
    "any_sync",
    "atomic_add_int32",
    "build_static_exec_queue",
    "f_init_const",
    "gt",
    "pack_into_32bit",
    "stg",
    "sts",
    "unpack_from_32bit",
    "unpack_from_32bit_host",
    "while_ld_global_acquire",
]
