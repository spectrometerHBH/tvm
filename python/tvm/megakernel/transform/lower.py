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
"""TIRX lowering entry points for logical megakernel specifications.

All lowering routes to the runtime-library builder in
``transform.runtime_build``: ``scheduler="static"`` emits the central-queue
persistent kernel and ``scheduler="dynamic"`` the MPMC persistent kernel with
runtime-scalar dispatch.  Host-side queue contents are derived by
``build_runtime_kernel`` in the same module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import tvm
from tvm.ir import IRModule
from tvm.tirx import PrimFunc

from ..dsl import KernelSpec


@dataclass(frozen=True)
class LoweringOptions:
    """Options for the runtime-library megakernel builder.

    ``scheduler`` selects the build path: ``"static"`` (default) emits the
    central-queue persistent kernel, ``"dynamic"`` the MPMC persistent
    scheduler with runtime-scalar dispatch synthesis.  ``smem_chunk_size``
    chunks the managed dynamic shared memory, and ``attrs`` carries backend
    parameters (hardware overrides such as ``sm_count``/``num_threads``/
    ``max_dynamic_smem``, ``profiler``, ``tile_coalescing``, and the
    ``megakernel.*`` tile/event attr namespace).
    """

    smem_chunk_size: int = 16 * 1024
    scheduler: str = "static"
    attrs: dict[str, Any] = field(default_factory=dict)


def _resolve_options(options: LoweringOptions | None) -> LoweringOptions:
    if options is None:
        return LoweringOptions()
    if not isinstance(options, LoweringOptions):
        raise TypeError("options must be a LoweringOptions instance or None")
    return options


def _route_scheduler(options: LoweringOptions) -> str:
    """Validate the routing knob; return the selected scheduler."""

    scheduler = options.scheduler
    if scheduler not in ("static", "dynamic"):
        raise ValueError(f"unsupported scheduler {scheduler!r}; expected 'static' or 'dynamic'")
    return scheduler


def _emit_with_runtime_builder(kernel: KernelSpec, options: LoweringOptions) -> IRModule:
    from .runtime_build import emit_runtime_module  # local import avoids a cycle

    return emit_runtime_module(kernel, options)


def lower_to_tirx(kernel: KernelSpec, options: LoweringOptions | None = None) -> PrimFunc:
    """Validate a spec and lower it to a persistent kernel with the runtime builder."""

    resolved = _resolve_options(options)
    _route_scheduler(resolved)
    return _emit_with_runtime_builder(kernel, resolved)[kernel.name]


def lower_to_tirx_module(kernel: KernelSpec, options: LoweringOptions | None = None) -> IRModule:
    """Lower a spec to its persistent device kernel module."""

    resolved = _resolve_options(options)
    _route_scheduler(resolved)
    return _emit_with_runtime_builder(kernel, resolved)


@tvm.transform.module_pass(opt_level=0, name="LowerMegakernelDSL")
class LowerMegakernelDSL:
    """Module pass wrapper around runtime-builder megakernel lowering."""

    def __init__(self, kernel: KernelSpec, options: LoweringOptions | None = None):
        self.kernel = kernel
        self.options = options

    def transform_module(self, mod: IRModule, _ctx: tvm.transform.PassContext) -> IRModule:
        lowered = lower_to_tirx_module(self.kernel, self.options)
        return IRModule({**mod.functions, **lowered.functions}, attrs=mod.attrs)


__all__ = [
    "LowerMegakernelDSL",
    "LoweringOptions",
    "lower_to_tirx",
    "lower_to_tirx_module",
]
