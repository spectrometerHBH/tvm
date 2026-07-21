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
"""CPU tests for the megakernel runtime building blocks.

Everything here runs without a GPU: host-side queue construction, task
packing, semaphore counter-protocol simulation, smem-manager bookkeeping,
and parse-level smoke tests of the emitted TIRX script.
"""

from typing import ClassVar

import numpy as np
import pytest

from tvm.error import DiagnosticError
from tvm.megakernel.dsl import TileImpl
from tvm.megakernel.runtime import (
    DynamicTileScheduler,
    HardwareConfig,
    MegaKernelWrapper,
    MPMCQueue,
    MPMCQueueHost,
    SemaphoreBase,
    SmemManager,
    StaticSemaphore,
    StaticTileScheduler,
    TaskPacking,
    build_static_exec_queue,
    dynamic_scheduler,
    pack_into_32bit,
    static_scheduler,
    unpack_from_32bit_host,
)
from tvm.script import tirx as T

# ---------------------------------------------------------------------------
# Hardware config / task packing
# ---------------------------------------------------------------------------


def test_hardware_config_defaults_and_derived_values():
    config = HardwareConfig()
    assert (config.sm_count, config.num_threads) == (148, 256)
    assert (config.warps_per_warpgroup, config.warpgroup_count, config.warp_size) == (4, 2, 32)
    assert config.max_dynamic_smem == 232448
    assert config.warp_count == 8
    assert config.warpgroup_size == 128
    assert config.full_mask == 0xFFFFFFFF


def test_hardware_config_rejects_inconsistent_values():
    with pytest.raises(ValueError, match="num_threads must equal"):
        HardwareConfig(num_threads=512)
    with pytest.raises(ValueError, match="positive integers"):
        HardwareConfig(sm_count=0)


def test_task_packing_defaults_and_validation():
    packing = TaskPacking()
    assert (packing.max_task_type, packing.max_m_idx) == (32, 8192)
    assert (packing.max_n_idx, packing.max_k_idx) == (1024, 16)
    assert (packing.m_shift, packing.n_shift, packing.k_shift) == (5, 18, 28)
    with pytest.raises(ValueError, match="add up to 32"):
        TaskPacking(k_bits=8)


def test_task_packing_bit_positions():
    assert pack_into_32bit(0, 0, 0, 7) == 7
    assert pack_into_32bit(1, 0, 0, 0) == 1 << 5
    assert pack_into_32bit(0, 1, 0, 0) == 1 << 18
    assert pack_into_32bit(0, 0, 1, 0) == 1 << 28


def test_task_packing_round_trip():
    cases = [
        (0, 0, 0, 0),
        (1, 2, 3, 4),
        (8191, 1023, 15, 31),
        (128, 512, 7, 18),
        (2047, 1000, 0, 29),
    ]
    for m_idx, n_idx, k_idx, task_type in cases:
        packed = pack_into_32bit(m_idx, n_idx, k_idx, task_type)
        assert isinstance(packed, int)
        assert unpack_from_32bit_host(packed) == (task_type, m_idx, n_idx, k_idx)


def test_task_packing_max_fields_saturate_int32():
    # All bit fields set is the all-ones pattern, i.e. int32 -1.  This is also
    # why the device dequeue spins on -1 for an empty queue slot.
    assert pack_into_32bit(8191, 1023, 15, 31) == -1


def test_task_packing_end_marker_bit_fields():
    # The host builders pack the end marker with -1 coordinates; on the wire
    # those become the all-ones bit fields, matching the device unpack.
    packed = pack_into_32bit(-1, -1, -1, 31)
    assert packed == -1
    assert unpack_from_32bit_host(packed) == (31, 8191, 1023, 15)


def test_task_packing_field_limits():
    with pytest.raises(ValueError, match="out of range"):
        pack_into_32bit(8192, 0, 0, 0, debug=True)
    with pytest.raises(ValueError, match="out of range"):
        pack_into_32bit(0, 1024, 0, 0, debug=True)
    with pytest.raises(ValueError, match="out of range"):
        pack_into_32bit(0, 0, 16, 0, debug=True)
    with pytest.raises(ValueError, match="out of range"):
        pack_into_32bit(0, 0, 0, 32, debug=True)
    # Boundary values are accepted.
    pack_into_32bit(8191, 1023, 15, 31, debug=True)


def test_task_packing_custom_layout_round_trip():
    packing = TaskPacking(task_type_bits=6, m_bits=12, n_bits=10, k_bits=4)
    packed = pack_into_32bit(5, 6, 7, 8, packing=packing)
    assert unpack_from_32bit_host(packed, packing=packing) == (8, 5, 6, 7)
    assert pack_into_32bit(1, 0, 0, 0, packing=packing) == 1 << 6


# ---------------------------------------------------------------------------
# Semaphore counter protocol (host-side simulation of the emitted atomics)
# ---------------------------------------------------------------------------


def _notify(value, number):
    """Mirror of Semaphore.semaphore_notify: atomic_add(-number) returns old."""

    old = value
    value -= number
    if old <= 0:
        # Retry path: spin until the counter is re-initialized, then subtract.
        return old, value, True
    return old, value, False


def test_semaphore_base_value():
    assert SemaphoreBase.base == 1 << 16


def test_static_semaphore_counter_protocol():
    base = SemaphoreBase.base
    expected_cnt = 2
    value = expected_cnt * (base + 1)

    old, value, retry = _notify(value, base + 1)
    assert (old, value, retry) == (2 * (base + 1), base + 1, False)
    assert value != 0  # a waiter would still spin

    old, value, retry = _notify(value, base + 1)
    assert (old, value, retry) == (base + 1, 0, False)
    assert value == 0  # wait condition reached exactly after all notifies


def test_static_semaphore_notify_retry_path():
    base = SemaphoreBase.base
    value = 0  # over-consumed counter: a late notify must wait for re-init
    old, value, retry = _notify(value, base + 1)
    assert retry and old == 0
    # Re-initialization by the next kernel phase, then the pending notify lands.
    value = base + 1
    old, value, retry = _notify(value, base + 1)
    assert (old, value, retry) == (base + 1, 0, False)


def test_dynamic_semaphore_two_phase_protocol():
    base = SemaphoreBase.base
    expected_cnt = 3
    value = expected_cnt * (base + 1)
    triggered = []

    for tile in range(expected_cnt):
        # pre-notify: subtract 1, trigger check on the OLD value
        old, value, _ = _notify(value, 1)
        triggered.append(old % base == 1)
        # post-notify: subtract base
        _, value, _ = _notify(value, base)

    # The pre-push fires exactly once, on the last tile's pre-notify, i.e.
    # when the final task has been dispatched (old_value % base == 1).
    assert triggered == [False, False, True]
    assert value == 0  # fully consumed: waiters observe zero


def test_event_init_complete_counter_math():
    # Mirror of MegaKernelWrapper.set_events_complete / task_impl_wait_*:
    # n user event tensors + the completion event itself notify once each;
    # every SM then consumes one (base + 1) from the completion counter.
    base = SemaphoreBase.base
    num_etensors = 4
    sm_count = HardwareConfig().sm_count
    value = (num_etensors + 1 + sm_count) * (base + 1)

    for _ in range(num_etensors + 1):  # INIT_ETENSOR task notifies
        _, value, _ = _notify(value, base + 1)
    assert value == sm_count * (base + 1)

    for remaining in range(sm_count, 0, -1):
        # The wait condition each SM evaluates before consuming.
        assert 0 < value <= sm_count * (base + 1)
        _, value, _ = _notify(value, base + 1)
    assert value == 0


# ---------------------------------------------------------------------------
# Shared-memory manager bookkeeping (parse-time, CPU only)
# ---------------------------------------------------------------------------

SMEM_MAX_BYTES = 31744
CHUNK_SIZE = 4096  # 7 chunks of transient region (28 KiB) + 3 KiB persistent region


def _make_manager(buf, hardware=None):
    return SmemManager(SMEM_MAX_BYTES, CHUNK_SIZE, buf.data, hardware=hardware)


_STASH = {}


def _stash(key, value):
    """Parser-safe way to keep a meta object reachable from the test body."""

    _STASH[key] = value


def test_smem_manager_chunk_limit():
    with pytest.raises(ValueError, match="chunk_num"):
        SmemManager(SMEM_MAX_BYTES * 8, CHUNK_SIZE, None)


def test_smem_manager_alloc_bookkeeping():
    _STASH.clear()

    @T.prim_func
    def main():
        buf = T.alloc_buffer([SMEM_MAX_BYTES], "uint8", scope="shared.dyn")
        mgr = _make_manager(buf)
        _stash("mgr", mgr)
        mgr.set_tile(None)
        a = mgr.alloc((128,), "float32")  # 512 bytes at offset 0
        b = mgr.alloc((64,), "float32")  # 256 bytes at offset 512
        c = mgr.alloc((16,), "int32", policy="persistent")
        T.evaluate(a[0] + b[0] + c[0])

    mgr = _STASH["mgr"]
    shared = mgr.tiles["default"][1]["shared"]
    assert shared == [(1, 0, 512, "shared"), (1, 512, 256, "shared")]
    # Highest transient chunk index touched by this tile.
    assert mgr.tiles["default"][0] == 0
    # Persistent allocation lives past the chunked region.
    [(beg_c, end_c)] = mgr.persistent_bufs.values()
    assert beg_c >= CHUNK_SIZE * mgr.chunk_num
    assert end_c <= SMEM_MAX_BYTES
    mgr.check_smem_well_formed()


def test_smem_manager_exclusive_split_chunk_tracking():
    _STASH.clear()

    @T.prim_func
    def main():
        buf = T.alloc_buffer([SMEM_MAX_BYTES], "uint8", scope="shared.dyn")
        mgr = _make_manager(buf)
        _stash("mgr", mgr)
        mgr.set_tile(None)
        # Exclusive buffer with split halves landing in disjoint chunks.
        b = mgr.alloc((8192,), "uint8", policy="exclusive", split=2)
        T.evaluate(b[0])

    mgr = _STASH["mgr"]
    assert mgr.tiles["default"][1]["exclusive"] == [(2, 0, 8192, "exclusive")]
    assert mgr.tiles["default"][0] == 1  # highest touched chunk index
    assert mgr.tiles["default"][2][0] == 1
    assert mgr.tiles["default"][2][1] == 1
    mgr.check_smem_well_formed()


def test_smem_manager_method_alias_matches_policy():
    _STASH.clear()

    @T.prim_func
    def main():
        buf = T.alloc_buffer([SMEM_MAX_BYTES], "uint8", scope="shared.dyn")
        mgr = _make_manager(buf)
        _stash("mgr", mgr)
        mgr.set_tile(None)
        a = mgr.alloc((32,), "float32", method="persistent")
        T.evaluate(a[0])

    assert len(_STASH["mgr"].persistent_bufs) == 1


def test_smem_manager_shared_exclusive_mix_rejected():
    with pytest.raises(DiagnosticError, match="Cannot use both shared and shared/exclusive"):

        @T.prim_func
        def main():
            buf = T.alloc_buffer([SMEM_MAX_BYTES], "uint8", scope="shared.dyn")
            mgr = _make_manager(buf)
            mgr.set_tile(None)
            a = mgr.alloc((128,), "float32", policy="shared")
            b = mgr.alloc((64,), "float32", policy="exclusive")
            T.evaluate(a[0] + b[0])


def test_smem_manager_transient_overflow_rejected():
    _STASH.clear()

    @T.prim_func
    def main():
        buf = T.alloc_buffer([SMEM_MAX_BYTES], "uint8", scope="shared.dyn")
        mgr = _make_manager(buf)
        _stash("mgr", mgr)
        mgr.set_tile(None)
        a = mgr.alloc((8192,), "float32")  # 32 KiB > 28 KiB chunk region
        T.evaluate(a[0])

    with pytest.raises(ValueError, match="exceeds the chunked region"):
        _STASH["mgr"].check_smem_well_formed()


def test_smem_manager_persistent_region_overflow_rejected():
    _STASH.clear()

    @T.prim_func
    def main():
        buf = T.alloc_buffer([SMEM_MAX_BYTES], "uint8", scope="shared.dyn")
        mgr = _make_manager(buf)
        _stash("mgr", mgr)
        mgr.set_tile(None)
        a = mgr.alloc((1024,), "float32", policy="persistent")  # 4 KiB > 3 KiB region
        T.evaluate(a[0])

    with pytest.raises(ValueError, match="persistent smem allocation is outside"):
        _STASH["mgr"].check_smem_well_formed()


def test_smem_manager_init_and_phase_ops_parse():
    @T.prim_func
    def main():
        T.device_entry()
        buf = T.alloc_buffer([SMEM_MAX_BYTES], "uint8", scope="shared.dyn")
        mgr = _make_manager(buf)
        mgr.set_tile(None)
        mgr.init()
        mgr.acquire_all("cta")
        mgr.release_all("cta")
        mgr.advance()
        mgr.wait_all("warpgroup")
        mgr.arrive_all("warpgroup")

    assert main is not None


# ---------------------------------------------------------------------------
# Host queue construction
# ---------------------------------------------------------------------------


def test_mpmc_queue_host_construction():
    queue = MPMCQueueHost(8)
    assert queue.tasks.shape == (8,)
    assert np.all(queue.tasks == -1)
    assert (queue.head[0], queue.tail[0]) == (0, 0)

    queue.enqueue(18, 1, 2, 3)
    queue.enqueue(19, 4, 5, 6)
    assert queue.tasks[0] == pack_into_32bit(1, 2, 3, 18)
    assert queue.tasks[1] == pack_into_32bit(4, 5, 6, 19)
    assert np.all(queue.tasks[2:] == -1)
    assert queue.tail[0] == 2
    assert unpack_from_32bit_host(queue.tasks[1]) == (19, 4, 5, 6)


def test_mpmc_queue_host_wraparound():
    queue = MPMCQueueHost(4)
    for i in range(5):
        queue.enqueue(i, i, 0, 0)
    assert queue.tail[0] == 5
    # Slot 0 was overwritten by the fifth task (position 4 & 3 == 0).
    assert queue.tasks[0] == pack_into_32bit(4, 0, 0, 4)
    assert queue.tasks[3] == pack_into_32bit(3, 0, 0, 3)


def test_mpmc_queue_capacity_must_be_power_of_two():
    with pytest.raises(ValueError, match="power-of-two"):
        MPMCQueue(1000, None, None, None, None)


def test_build_static_exec_queue_layout():
    central = [(m, 0, 0, 18) for m in range(5)]
    exec_queue = build_static_exec_queue(central, sm_count=2, max_tasks=6, end_task_type=31)
    end = pack_into_32bit(-1, -1, -1, 31)
    expected = np.zeros((2, 6), dtype=np.int32)
    expected[0, 0] = pack_into_32bit(0, 0, 0, 18)
    expected[1, 0] = pack_into_32bit(1, 0, 0, 18)
    expected[0, 1] = pack_into_32bit(2, 0, 0, 18)
    expected[1, 1] = pack_into_32bit(3, 0, 0, 18)
    expected[0, 2] = pack_into_32bit(4, 0, 0, 18)
    expected[1, 2] = end  # queue ran out mid-row: end marker for this SM
    expected[0, 3] = end  # final end row
    expected[1, 3] = end
    np.testing.assert_array_equal(exec_queue, expected)
    # The static scheduler decodes the first task of SM 1 as (m=1, type=18).
    assert unpack_from_32bit_host(exec_queue[1, 0]) == (18, 1, 0, 0)


def test_build_static_exec_queue_exact_row_boundary():
    central = [(m, 0, 0, 18) for m in range(4)]
    exec_queue = build_static_exec_queue(central, sm_count=2, max_tasks=4, end_task_type=31)
    end = pack_into_32bit(-1, -1, -1, 31)
    # Full rows have no end marker; exactly one final end row follows.
    assert exec_queue[0, 1] == pack_into_32bit(2, 0, 0, 18)
    assert exec_queue[1, 1] == pack_into_32bit(3, 0, 0, 18)
    assert list(exec_queue[:, 2]) == [end, end]
    assert np.all(exec_queue[:, 3] == 0)


# ---------------------------------------------------------------------------
# Scheduler construction
# ---------------------------------------------------------------------------


class _FakeSmemManager:
    def __init__(self, hardware):
        self.hardware = hardware


def test_dynamic_scheduler_hardware_defaults():
    sched = DynamicTileScheduler(None, None, None, _FakeSmemManager(HardwareConfig()))
    assert sched.scheduler_warp == 7  # last warp of the CTA
    assert sched.MAX_TASKS == 32768
    assert sched.queue.mask == 32768 - 1
    assert sched.end_task_type == 31


def test_dynamic_scheduler_custom_hardware():
    hardware = HardwareConfig(num_threads=128, warpgroup_count=1)
    sched = DynamicTileScheduler(None, None, None, _FakeSmemManager(hardware))
    assert sched.scheduler_warp == 3
    assert sched.end_task_type == 31


def test_static_scheduler_construction():
    sched = StaticTileScheduler("test", None, _FakeSmemManager(HardwareConfig()))
    assert sched.MAX_TASKS == 128
    assert sched.end_task_type == 31
    assert sched.hardware.num_threads == 256
    assert sched.prefix == "test"


def test_static_scheduler_init_parses():
    sm_count = HardwareConfig().sm_count
    max_tasks = StaticTileScheduler.MAX_TASKS

    @T.prim_func
    def main(exec_queue: T.Buffer((sm_count, max_tasks), "int32")):
        T.device_entry()
        buf = T.alloc_buffer([SMEM_MAX_BYTES], "uint8", scope="shared.dyn")
        mgr = _make_manager(buf)
        sched = StaticTileScheduler("test", exec_queue, mgr)
        sched.init()
        mgr.init()
        with T.While(sched.valid()):
            sched.next_tile()

    assert main is not None


def test_dynamic_scheduler_init_parses():
    max_tasks = DynamicTileScheduler.MAX_TASKS

    @T.prim_func
    def main(
        tasks: T.Buffer((max_tasks,), "int32"),
        head: T.Buffer((1,), "int32"),
        tail: T.Buffer((1,), "int32"),
    ):
        T.device_entry()
        buf = T.alloc_buffer([SMEM_MAX_BYTES], "uint8", scope="shared.dyn")
        mgr = _make_manager(buf)
        sched = DynamicTileScheduler(tasks, head, tail, mgr)
        sched.init()
        mgr.init()
        with T.While(sched.valid()):
            sched.next_tile()

    assert main is not None


# ---------------------------------------------------------------------------
# Wrapper lifecycle
# ---------------------------------------------------------------------------


class _RecordingTile(TileImpl):
    calls: ClassVar[list] = []

    @classmethod
    def init_shared_resources(cls, smem_manager):
        cls.calls.append(("class_init", smem_manager.cur_tile_name))

    @classmethod
    def finalize_shared_resources(cls, smem_manager):
        cls.calls.append(("class_finalize",))

    def device_init(self, smem_manager, m_idx, n_idx, k_idx):
        self.calls.append(("device_init", smem_manager.cur_tile_name, m_idx, n_idx, k_idx))

    def host_init(self):
        type(self).calls.append(("host_init",))

    def run(self, m_idx, n_idx, k_idx):
        pass


def test_wrapper_registration_host_init_and_reset():
    _RecordingTile.calls = []
    wrapper = MegaKernelWrapper()
    tile_a = wrapper._add_tile(_RecordingTile(), None)
    tile_b = wrapper._add_tile(_RecordingTile(), None, predicate=False)

    wrapper.host_init_all()
    assert _RecordingTile.calls == [("host_init",)]  # predicate=False skips tile_b

    assert wrapper.tile_attr[tile_a] == (None, True)
    assert wrapper.tile_attr[tile_b] == (None, False)
    assert wrapper.class_list == {_RecordingTile}

    wrapper.reset()
    assert wrapper.tile_attr == {} and wrapper.class_list == set()
    assert wrapper.etensor_and_f_init_pairs == [] and wrapper.etensor_workspace_offset == 0


def test_wrapper_drives_dsl_tile_hooks():
    _RecordingTile.calls = []
    wrapper = MegaKernelWrapper()
    tile = wrapper._add_tile(_RecordingTile(), None)

    @T.prim_func
    def main():
        buf = T.alloc_buffer([SMEM_MAX_BYTES], "uint8", scope="shared.dyn")
        wrapper.set_smem_manager(SMEM_MAX_BYTES, CHUNK_SIZE, buf.data)
        wrapper.device_init_all(wrapper.smem_manager)
        wrapper.class_init_all(wrapper.smem_manager)
        wrapper.class_finalize_all(wrapper.smem_manager)

    smem = wrapper.smem_manager
    assert smem.hardware.num_threads == 256
    assert ("device_init", str(tile), 0, 0, 0) in _RecordingTile.calls
    assert ("class_init", str(_RecordingTile)) in _RecordingTile.calls
    assert ("class_finalize",) in _RecordingTile.calls
    # Declaration-time allocation records exist for both the instance and class keys.
    assert str(tile) in smem.tiles
    assert str(_RecordingTile) in smem.tiles


def test_wrapper_profiler_strides_from_hardware():
    wrapper = MegaKernelWrapper()
    assert wrapper.NUM_GROUPS == 8
    assert wrapper.PROFILER_WRITE_STRIDE == 148 * 8
    custom = MegaKernelWrapper(
        hardware=HardwareConfig(sm_count=4, num_threads=128, warpgroup_count=1)
    )
    assert custom.NUM_GROUPS == 4
    assert custom.PROFILER_WRITE_STRIDE == 16


def test_wrapper_add_etensor_and_set_events_complete():
    wrapper = MegaKernelWrapper()

    @T.prim_func
    def main(workspace: T.Buffer((1024,), "int32")):
        evt_a = wrapper.add_etensor(StaticSemaphore, workspace, [4], None)
        evt_b = wrapper.add_etensor(StaticSemaphore, workspace, [8], None)
        wrapper.set_events_complete(False, StaticSemaphore, workspace)
        T.evaluate(evt_a.sem[0] + evt_b.sem[0])

    # Two user etensors plus the completion event appended by set_events_complete.
    assert wrapper.etensor_workspace_offset == 4 + 8 + 1
    assert len(wrapper.etensor_and_f_init_pairs) == 3
    # The completion event's init count covers the 2 user events, the
    # completion event itself, and one slot per SM.
    _, f_init = wrapper.etensor_and_f_init_pairs[-1]
    assert f_init() == 2 + 1 + HardwareConfig().sm_count
    assert wrapper.init_etensor_tile in wrapper.tile_attr
    assert wrapper.evt_etensor_init_complete is not None


def test_wrapper_set_events_complete_dynamic_has_no_completion_event():
    wrapper = MegaKernelWrapper()

    @T.prim_func
    def main(workspace: T.Buffer((1024,), "int32")):
        evt = wrapper.add_etensor(dynamic_scheduler.Semaphore, workspace, [4], None)
        wrapper.set_events_complete(True, dynamic_scheduler.Semaphore, workspace)
        T.evaluate(evt.sem[0])

    assert wrapper.evt_etensor_init_complete is None
    assert len(wrapper.etensor_and_f_init_pairs) == 1


def test_static_and_dynamic_semaphore_modules_distinct():
    assert static_scheduler.Semaphore is StaticSemaphore
    assert dynamic_scheduler.Semaphore is not StaticSemaphore
    assert issubclass(StaticSemaphore, SemaphoreBase)
    assert issubclass(dynamic_scheduler.Semaphore, SemaphoreBase)
