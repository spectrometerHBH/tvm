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
"""Backend-independent semantic plans for the megakernel DSL."""

from __future__ import annotations

from dataclasses import dataclass

from ...dsl import EventSpec, KernelSpec, TensorSpec, TileSpec, VarSpec


@dataclass(frozen=True, eq=False)
class SemanticEdge:
    """One logical event dependency between two tile specifications."""

    event: EventSpec
    producer: TileSpec
    consumer: TileSpec

    @property
    def key(self) -> tuple[int, int, int]:
        """Return an identity-based key that cannot alias foreign objects."""

        return id(self.event), id(self.producer), id(self.consumer)

    def __hash__(self) -> int:
        return hash(self.key)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SemanticEdge) and self.key == other.key


@dataclass(frozen=True)
class SemanticPlan:
    """The validated, backend-independent meaning of a ``KernelSpec``."""

    kernel: KernelSpec
    vars: tuple[VarSpec, ...]
    tensors: tuple[TensorSpec, ...]
    events: tuple[EventSpec, ...]
    tiles: tuple[TileSpec, ...]
    logical_edges: tuple[SemanticEdge, ...]


__all__ = ["SemanticEdge", "SemanticPlan"]
