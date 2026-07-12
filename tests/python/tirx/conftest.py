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
"""Suite-level hardware gate for the tirx tests.

The tirx kernels and codegen paths target Blackwell (sm_100a) — they emit
PTX/SASS (tcgen05, tmem, cp.async ``.async`` modifiers, fp8 conversions, ...)
that ptxas/NVRTC reject for older targets, and many tests execute on the
device. Running the suite on a CPU-only node or a pre-sm_100 GPU therefore
fails at compile/run time rather than skipping. Gate the whole directory on a
real sm_100a device so it skips cleanly where the hardware is absent and runs
in full where it is present.
"""

import gc
import os

import pytest

from tvm.testing import env


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


def pytest_configure(config):
    del config
    _set_cuda_device_for_xdist_worker()


@pytest.fixture(autouse=True)
def _release_cuda_cache_between_tests():
    _set_cuda_device_for_xdist_worker()
    yield
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def pytest_collection_modifyitems(config, items):
    if env.has_cuda_compute(10):
        return
    skip = pytest.mark.skip(
        reason="tirx suite requires a CUDA compute capability 10.0 (sm_100a) device"
    )
    for item in items:
        item.add_marker(skip)
