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
    assert results["benchmark_protocol"]["round_aggregate"] == "mean"


def test_bench_retains_round_samples_and_uses_arithmetic_mean(monkeypatch):
    values = iter([0.001, 0.002, 0.100])

    def fake_timer(_fn, warmup=25, rep=100):
        del warmup, rep
        return next(values)

    monkeypatch.setattr(bench_module, "_do_bench_event", fake_timer)

    results = bench({"tir": lambda: None}, timer="event", cooldown_s=0, rounds=3)

    assert results["round_samples"] == {"tir": [1.0, 2.0, 100.0]}
    assert results["impls"] == {"tir": 103.0 / 3.0}


def test_bench_l2_flush_buffer_matches_triton_256_mib(monkeypatch):
    captured = {}

    def fake_empty(size, *, dtype, device):
        captured.update(size=size, dtype=dtype, device=device)
        return object()

    monkeypatch.setattr(bench_module.torch, "empty", fake_empty)

    bench_module._empty_cache_for_benchmark()

    assert captured == {
        "size": 256 * 1024 * 1024 // 4,
        "dtype": torch.int,
        "device": "cuda",
    }


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


def test_bench_default_timer_is_proton(monkeypatch):
    """Omitting timer resolves to Proton and invokes only the Proton timer."""
    calls = []

    def fake_proton(fn, warmup=25, rep=100):
        calls.append((warmup, rep))
        fn()
        return 0.001

    monkeypatch.setattr(bench_module, "_do_bench_proton", fake_proton)

    results = bench({"noop": lambda: None}, cooldown_s=0)

    assert results["timer"] == "proton"
    assert results["impls"] == {"noop": 1.0}
    assert calls == [(25, 100)]


def test_bench_cudagraph_proton_wiring(monkeypatch):
    calls = []

    def fake_cudagraph_proton(fn, rep=20):
        calls.append(rep)
        fn()
        return 0.002

    monkeypatch.setattr(bench_module, "_do_bench_cudagraph_proton", fake_cudagraph_proton)

    results = bench({"noop": lambda: None}, timer="cudagraph_proton", cudagraph_rep=7, cooldown_s=0)

    assert results["impls"] == {"noop": 2.0}
    assert results["timer"] == "cudagraph_proton"
    assert calls == [7]


def test_bench_never_silently_falls_back_from_proton(monkeypatch):
    def unavailable(_fn, warmup=25, rep=100):
        del warmup, rep
        raise RuntimeError("Proton profiler session could not be created")

    monkeypatch.setattr(bench_module, "_do_bench_proton", unavailable)

    with pytest.raises(RuntimeError, match="Proton profiler session"):
        bench({"noop": lambda: None}, timer="proton", cooldown_s=0)


@pytest.mark.parametrize(
    ("timer", "alternative"),
    [("proton", "event"), ("cudagraph_proton", "event")],
)
def test_missing_proton_session_is_an_explicit_error(monkeypatch, timer, alternative):
    monkeypatch.setattr(bench_module.proton, "start", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match=rf"{timer}.*timer='{alternative}'"):
        bench_module._start_proton_session("profile", timer=timer, explicit_alternative=alternative)


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
    """Unknown timer names fail instead of changing measurement method."""
    A = torch.randn(8, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError):
        bench({"mm": lambda: torch.mm(A, A)}, timer="unknown")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
