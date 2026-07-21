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
"""Hardware launch parameters for the megakernel runtime building blocks.

This is the parameterization replacement for the production
``KernelConfig`` constant table: every runtime class takes the values it
needs from one ``HardwareConfig`` instance instead of reading module-level
constants.  The defaults are the B200 production values.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware constants consumed by the runtime building blocks."""

    sm_count: int = 148
    num_threads: int = 256
    warps_per_warpgroup: int = 4
    warpgroup_count: int = 2
    warp_size: int = 32
    max_dynamic_smem: int = 232448

    def __post_init__(self):
        for field_value in (
            self.sm_count,
            self.num_threads,
            self.warps_per_warpgroup,
            self.warpgroup_count,
            self.warp_size,
            self.max_dynamic_smem,
        ):
            if not isinstance(field_value, int) or field_value <= 0:
                raise ValueError("hardware config values must be positive integers")
        if self.num_threads != self.warp_size * self.warps_per_warpgroup * self.warpgroup_count:
            raise ValueError(
                "num_threads must equal warp_size * warps_per_warpgroup * warpgroup_count"
            )

    @property
    def warp_count(self) -> int:
        """Total warps per CTA (warps per warp-group times warp-groups)."""

        return self.warps_per_warpgroup * self.warpgroup_count

    @property
    def warpgroup_size(self) -> int:
        """Threads per warp-group."""

        return self.num_threads // self.warpgroup_count

    @property
    def full_mask(self) -> int:
        """Full participation mask for warp-collective intrinsics."""

        return (1 << self.warp_size) - 1


class RuntimeProfileEvent(Enum):
    """Default profiler event ids for the runtime-owned generic roles.

    Callers that keep their own profiler event enum can pass its members to
    the wrapper/scheduler constructors instead; only ``.value`` is consumed.
    """

    FETCH = 5
    PUSH = 20
    PREFETCH = 26
    INIT_ETENSOR = 52
    WAIT_ETENSOR_INIT = 53


__all__ = ["HardwareConfig", "RuntimeProfileEvent"]
