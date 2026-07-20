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
"""Megakernel execution plans, physical programs, and lowering backends."""

from .lower import (
    LoweringOptions,
    LowerMegakernelDSL,
    MegakernelLowerer,
    TIRXStaticBackend,
    lower_execution_plan,
    lower_static_queue_init_to_tirx,
    lower_to_tirx,
    lower_to_tirx_module,
)
from .model import (
    BarrierStep,
    DeviceRegionPlan,
    EdgePlacement,
    ExecutionPlan,
    ExecutionPlanBackend,
    FetchGuardStep,
    HookStep,
    HostCallStep,
    HostRegionPlan,
    HostSyncStep,
    LogicalEdge,
    MidBodyPortStep,
    NotifyStep,
    ProgramStep,
    QueuePushStep,
    RegionDependencyPlan,
    RunStep,
    RuntimeEventInitStep,
    TileProgram,
    WaitStep,
    logical_edges,
    make_static_execution_plan,
)
from .semantic import SemanticEdge, SemanticPlan, build_semantic_plan, validate_semantic_plan

__all__ = [
    "BarrierStep",
    "DeviceRegionPlan",
    "EdgePlacement",
    "ExecutionPlan",
    "ExecutionPlanBackend",
    "FetchGuardStep",
    "HookStep",
    "HostCallStep",
    "HostRegionPlan",
    "HostSyncStep",
    "LogicalEdge",
    "LowerMegakernelDSL",
    "LoweringOptions",
    "MegakernelLowerer",
    "MidBodyPortStep",
    "NotifyStep",
    "ProgramStep",
    "QueuePushStep",
    "RegionDependencyPlan",
    "RunStep",
    "RuntimeEventInitStep",
    "SemanticEdge",
    "SemanticPlan",
    "TIRXStaticBackend",
    "TileProgram",
    "WaitStep",
    "build_semantic_plan",
    "logical_edges",
    "lower_execution_plan",
    "lower_static_queue_init_to_tirx",
    "lower_to_tirx",
    "lower_to_tirx_module",
    "make_static_execution_plan",
    "validate_semantic_plan",
]
