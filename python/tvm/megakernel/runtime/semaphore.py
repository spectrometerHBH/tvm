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
"""Two-phase counter semaphore protocol base.

The semaphore counters implement the production two-phase protocol
(notes for ``decrement=False``; the logic for ``True`` is similar):

- For a semaphore with expected count ``expected_cnt``, the actual count is
  initialized to ``expected_cnt * (base + 1)``.  ``base >= expected_cnt`` is
  a power of 2 (for efficient mod below, and to give convenience for
  ``decrement=True``); by default ``base`` is ``1 << 16``.
- In dynamic scheduling, the semaphore is notified twice per task.  The
  first notify happens after the prefetch of the tile but before the
  corresponding semaphore wait, and atomic-adds the semaphore value by 1.
  The second happens after the tile runs, and atomic-adds by ``base``.
  Task pushing happens after the first semaphore notify, triggered by
  ``old_value % base == expected_cnt - 1``.  In this way tasks are
  pre-pushed when the last tile has already been dispatched to the SM,
  which avoids the deadlock.
- For semaphore wait, the condition ``value == expected_cnt * (base + 1)``
  still distinguishes a fully-signaled semaphore.

The static scheduler's semaphore notifies once per task with
``base + 1`` and waits for the counter to reach zero.
"""

from __future__ import annotations

from tvm.script import tirx as T


@T.meta_class
class SemaphoreBase:
    """Abstract base class for semaphore."""

    base = 1 << 16

    def __init__(self):
        pass

    def semaphore_wait(self, *coord, level, mask):
        raise NotImplementedError

    def semaphore_notify(self, *coord, rank=-1):
        raise NotImplementedError


__all__ = ["SemaphoreBase"]
