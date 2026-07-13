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
"""Tests for tvm.tirx.bench utilities."""

import importlib

import pytest
import torch

pytest.importorskip("triton")  # tvm.tirx.bench imports triton.profiler

from tvm.testing import env
from tvm.tirx.bench import bench

bench_module = importlib.import_module("tvm.tirx.bench")


def test_bench_cooldown_precedes_every_impl(monkeypatch):
    """cooldown_s sleeps immediately before each impl's warmup+measurement.

    2 impls x 2 rounds = 4 timed calls, so 4 sleeps (the first impl in the
    first round is included). Pins the #29 per-impl cooldown semantics.
    """
    calls = []
    sleeps = []

    def fake_timer(fn, warmup=25, rep=100):
        del warmup, rep
        fn()
        return 0.001

    monkeypatch.setattr(bench_module, "_do_bench_event", fake_timer)
    monkeypatch.setattr(bench_module.time, "sleep", sleeps.append)

    results = bench(
        {"a": lambda: calls.append("a"), "b": lambda: calls.append("b")},
        warmup=0,
        repeat=1,
        timer="event",
        cooldown_s=1.0,
        rounds=2,
    )

    assert calls == ["a", "b", "a", "b"]
    assert sleeps == [1.0, 1.0, 1.0, 1.0]
    assert results["benchmark_protocol"]["cooldown_s"] == 1.0


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_event_pure_launch():
    """New Triton-standard bench(): no-arg launch closures, event timer."""
    M, N = 256, 256
    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = torch.randn(M, N, device="cuda", dtype=torch.float16)

    funcs = {"mm": lambda: torch.mm(A, B)}
    results = bench(funcs, warmup=5, repeat=10, timer="event")
    assert "mm" in results["impls"]
    assert results["impls"]["mm"] > 0
    assert results["timer"] == "event"


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_default_timer_is_proton():
    """Omitting timer resolves to the proton default (recorded as timer='proton').

    Under pytest _do_bench_proton falls back to event timing, but bench() still
    records the resolved timer name, so this pins the timer=None -> 'proton' default.
    """
    M, N = 256, 256
    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = torch.randn(M, N, device="cuda", dtype=torch.float16)

    funcs = {"mm": lambda: torch.mm(A, B)}
    results = bench(funcs, warmup=5, repeat=10)
    assert results["timer"] == "proton"
    assert results["impls"]["mm"] > 0


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_cudagraph_proton_pure_launch():
    """New Triton-standard bench(): cudagraph_proton timer.

    Under pytest, _do_bench_cudagraph_proton falls back to event-based cudagraph
    timing (no CUPTI/Proton required in CI), so this exercises the mode wiring and
    the fallback path rather than the real Proton attribution.
    """
    M, N = 256, 256
    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = torch.randn(M, N, device="cuda", dtype=torch.float16)

    funcs = {"mm": lambda: torch.mm(A, B)}
    results = bench(funcs, warmup=5, repeat=10, timer="cudagraph_proton")
    assert results["impls"]["mm"] > 0
    assert results["timer"] == "cudagraph_proton"


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_references_pure_launch():
    """New bench(): reference builders return no-arg callables and get timed."""
    M, N = 128, 128
    A = torch.randn(M, N, device="cuda", dtype=torch.float16)
    B = torch.randn(M, N, device="cuda", dtype=torch.float16)

    funcs = {"tir": lambda: torch.mm(A, B)}

    def _addmm():
        C = torch.zeros(M, N, device="cuda", dtype=torch.float16)
        return lambda: torch.addmm(C, A, B)

    results = bench(funcs, warmup=5, repeat=10, timer="event", references={"addmm": _addmm})
    assert set(results["impls"].keys()) == {"tir", "addmm"}
    assert all(v > 0 for v in results["impls"].values())


@pytest.mark.gpu
@pytest.mark.skipif(not env.has_cuda(), reason="need cuda")
def test_bench_rejects_unknown_timer():
    """New bench() only accepts event / proton / cudagraph_proton (plain cudagraph
    was removed)."""
    A = torch.randn(8, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        bench({"mm": lambda: torch.mm(A, A)}, timer="cudagraph")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
