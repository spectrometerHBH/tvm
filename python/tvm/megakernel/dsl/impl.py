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
"""Tile implementation interface for the megakernel DSL."""

from abc import ABC, abstractmethod


class TileImpl(ABC):
    """Local implementation of one tile kind.

    A ``TileImpl`` describes how one tile instance runs.  It does not describe
    global scheduling, event layout, or cross-tile dependency policy.  Those
    relationships are represented by the logical specification classes.
    """

    @classmethod
    def class_name(cls):
        """Return a stable implementation name."""

        return cls.__name__

    def __init__(self):
        self._instance_id = id(self)

    def __str__(self):
        return f"{self.__class__.__name__}-{self._instance_id:x}"

    @classmethod
    def init_shared_resources(cls):
        """Initialize resources shared by all instances of this tile class."""

    @classmethod
    def finalize_shared_resources(cls):
        """Release resources created by ``init_shared_resources``."""

    def device_init(self):
        """Initialize device-side state owned by one tile instance."""

    def host_init(self):
        """Initialize host-side state for one tile instance."""

    def prefetch(self, m_idx, n_idx, k_idx):
        """Optionally prefetch data for one tile instance before ``run``."""

    @abstractmethod
    def run(self, m_idx, n_idx, k_idx):
        """Run one logical tile instance at index ``(m_idx, n_idx, k_idx)``."""
