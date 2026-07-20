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
"""Build semantic plans from logical megakernel specifications."""

from __future__ import annotations

from ...dsl import EventSpec, KernelSpec
from .model import SemanticEdge, SemanticPlan


def build_semantic_plan(kernel: KernelSpec) -> SemanticPlan:
    """Create a stable semantic view without choosing a physical policy."""

    if not isinstance(kernel, KernelSpec):
        raise TypeError("semantic plan input must be a KernelSpec")
    return SemanticPlan(
        kernel=kernel,
        vars=tuple(kernel.vars.values()),
        tensors=tuple(kernel.tensors.values()),
        events=tuple(kernel.events.values()),
        tiles=tuple(kernel.tiles),
        logical_edges=semantic_edges(kernel),
    )


def semantic_edges(kernel: KernelSpec) -> tuple[SemanticEdge, ...]:
    """Return logical event edges in stable event and tile order."""

    producers: dict[int, list] = {id(event): [] for event in kernel.events.values()}
    consumers: dict[int, list] = {id(event): [] for event in kernel.events.values()}
    for tile in kernel.tiles:
        for event, _ in tile.notifies:
            if all(candidate is not tile for candidate in producers.setdefault(id(event), [])):
                producers[id(event)].append(tile)
        for event, _ in tile.waits:
            if all(candidate is not tile for candidate in consumers.setdefault(id(event), [])):
                consumers[id(event)].append(tile)

    result = []
    for event in kernel.events.values():
        for producer in producers.get(id(event), ()):
            for consumer in consumers.get(id(event), ()):
                result.append(SemanticEdge(event, producer, consumer))
    return tuple(result)


def event_init_count(event: EventSpec, coord: tuple[int, ...]) -> int:
    """Evaluate one logical event count at a concrete coordinate."""

    return event.init_count(coord) if callable(event.init_count) else event.init_count


__all__ = ["build_semantic_plan", "event_init_count", "semantic_edges"]
