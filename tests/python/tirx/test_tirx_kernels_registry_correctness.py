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
"""Correctness coverage for kernels registered in tirx-kernels."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

import pytest

kernel_registry = pytest.importorskip("tirx_kernels.registry")
kernel_runner = pytest.importorskip("tirx_kernels.runner")

_KERNELS = kernel_registry.discover_kernels(min_compute_capability=10, strict=True)


def _registered_kernel_config_cases():
    cases = []
    for kernel_name, mod in sorted(_KERNELS.items()):
        configs = getattr(mod, "CONFIGS", [])
        for index, config in enumerate(configs):
            label = config.get("label", f"config{index}")
            cases.append(pytest.param(kernel_name, config, id=f"{kernel_name}::{label}"))
    return cases


def _set_cuda_device_for_xdist_worker():
    try:
        import torch
    except ImportError:
        return

    if not torch.cuda.is_available():
        return
    worker = os.environ.get("PYTEST_XDIST_WORKER", "gw0")
    worker_index = int(worker[2:]) if worker.startswith("gw") and worker[2:].isdigit() else 0
    torch.cuda.set_device(worker_index % torch.cuda.device_count())


def _visible_cuda_device_count():
    try:
        import torch
    except ImportError:
        return 0

    if not torch.cuda.is_available():
        return 0
    return torch.cuda.device_count()


def _required_cuda_device_count(config):
    return int(config.get("num_processes", 1))


@contextmanager
def _registry_gpu_lock(config):
    try:
        import fcntl

        import torch
    except ImportError:
        yield
        return

    if not torch.cuda.is_available():
        yield
        return

    if int(config.get("num_processes", 1)) > 1:
        lock_name = "global"
    else:
        lock_name = f"device{torch.cuda.current_device()}"
    lock_path = Path("/tmp") / f"tirx-kernels-registry-correctness-{lock_name}.lock"
    with lock_path.open("w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            torch.cuda.empty_cache()
            yield
        finally:
            torch.cuda.empty_cache()
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@pytest.mark.parametrize(("kernel_name", "config"), _registered_kernel_config_cases())
def test_registered_tirx_kernel_correctness(kernel_name, config):
    if kernel_name == "deepgemm_fp8_fp4_mega_moe":
        pytest.skip("mega_moe multi-GPU correctness requires a dedicated scheduler")
    _set_cuda_device_for_xdist_worker()
    required_devices = _required_cuda_device_count(config)
    visible_devices = _visible_cuda_device_count()
    if required_devices > visible_devices:
        pytest.skip(
            f"requires {required_devices} CUDA devices, but only {visible_devices} are visible"
        )
    with _registry_gpu_lock(config):
        kernel_runner.run_kernel_test(kernel_name, config, registry=_KERNELS)
