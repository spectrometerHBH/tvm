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
"""Parser-style implementation hooks for the megakernel DSL."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal

from tvm.script import tirx as T


@dataclass(frozen=True)
class SmemAllocRecord:
    """Metadata recorded for one managed shared-memory allocation."""

    buffer: Any
    shape: Any
    dtype: str
    policy: str
    attrs: dict[str, Any] = field(default_factory=dict)


class SmemManager:
    """Shared-memory manager supplied to ``TileImpl`` lifecycle hooks."""

    VALID_POLICIES: ClassVar[set[str]] = {"shared", "exclusive", "persistent"}

    def alloc(
        self,
        shape,
        dtype="float32",
        strides=None,
        scope="shared.dyn",
        align=0,
        buffer_type="",
        axis_separators=None,
        layout="default",
        policy: Literal["shared", "exclusive", "persistent"] = "shared",
        **kwargs,
    ):
        """Allocate managed shared memory and return its buffer."""

        raise NotImplementedError

    def commit(self):
        """Finalize the managed shared-memory pool."""

        raise NotImplementedError

    def acquire_all(self, level="cta"):
        """Wait until the current tile's shared-memory phase is available."""

        raise NotImplementedError

    def wait_all(self, level="cta"):
        """Alias for ``acquire_all`` used by existing tile implementations."""

        return self.acquire_all(level)

    def release_all(self, level="cta"):
        """Release the current tile's shared-memory phase."""

        raise NotImplementedError

    def advance(self):
        """Advance the current tile's shared-memory phase."""

        raise NotImplementedError


class TileImpl(ABC):
    """Local parser-style implementation of one logical tile kind.

    Global scheduling, dependency placement, profiling, and region lifecycle
    are owned by the lowering backend.  Implementations own only their
    tensors and local resources.

    The optional endpoint-scope class attributes describe the physical sync
    granularity this implementation uses for its event endpoints:

    - ``wait_level``: scope at which the tile waits on events, one of
      ``"thread"``, ``"warp"``, ``"warpgroup"``, or ``"cta"``.
    - ``wait_mask``: 32-bit thread mask participating in each wait.
    - ``notify_scope``: ``(scope, scope_id)`` pair at which the tile signals
      events; ``scope`` uses the same legal values as ``wait_level`` and
      ``scope_id`` is a non-negative integer.

    The defaults (``"cta"``, full mask, ``("cta", 0)``) match the reference
    static backend, so existing implementations need no changes.
    """

    wait_level: ClassVar[str] = "cta"
    wait_mask: ClassVar[int] = 0xFFFFFFFF
    notify_scope: ClassVar[tuple[str, int]] = ("cta", 0)

    @classmethod
    def class_name(cls):
        """Return a stable implementation name."""

        return cls.__name__

    def __init__(self):
        self._instance_id = id(self)

    def __str__(self):
        return f"{self.__class__.__name__}-{self._instance_id:x}"

    @classmethod
    @T.inline
    def init_shared_resources(cls, smem_manager: SmemManager):
        """Optionally initialize resources shared by all class instances."""

    @classmethod
    @T.inline
    def finalize_shared_resources(cls, smem_manager: SmemManager):
        """Optionally finalize resources shared by all class instances."""

    @T.inline
    def device_init(self, smem_manager: SmemManager, m_idx, n_idx, k_idx):
        """Optionally initialize device resources owned by this instance."""

    def host_init(self):
        """Optionally initialize host resources owned by this instance."""

    @T.inline
    def prefetch(self, m_idx, n_idx, k_idx):
        """Optionally prefetch data before ``run``."""

    @abstractmethod
    @T.inline
    def run(self, m_idx, n_idx, k_idx):
        """Emit the parser-style tile body at ``(m_idx, n_idx, k_idx)``."""
