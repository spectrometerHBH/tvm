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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fail-closed environment validation for NVIDIA's official IKET runtime."""

import hashlib
import json
import os
import shutil
from importlib import metadata
from pathlib import Path

_PROFILE_ENV = "TVM_IKET_OFFICIAL_PROFILE"
_PROFILE_HINT = (
    "TVM_IKET_OFFICIAL_PROFILE=cutlass-4.6.1 "
    "run-iket profile --postprocess all -- python workload.py"
)

# Hashes are from the public 4.6.1 wheels recorded by the oracle generator.
# Only ABI-independent runtime/compiler binaries are pinned so the profile is
# usable with every Python version supported by that CUTLASS DSL release.
_OFFICIAL_PROFILES = {
    "cutlass-4.6.1": {
        "versions": {
            "nvidia-cutlass-dsl": "4.6.1",
            "nvidia-cutlass-dsl-libs-base": "4.6.1",
            "nvidia-cutlass-dsl-libs-core": "4.6.1",
            "nvidia-cutlass-dsl-libs-cu13": "4.6.1",
            "nvidia-cuda-nvdisasm": "13.3.73",
            "nvidia-cuda-nvrtc": "13.3.33",
        },
        "files": {
            "nvidia-cutlass-dsl-libs-base": {
                "nvidia_cutlass_dsl/dsl_packages/iket/libiket_cubin_info.so": (
                    "7ee839130c6bd129b04908a807c066118a459ebea644a59ecb6e41fbb323c103"
                ),
                "nvidia_cutlass_dsl/dsl_packages/iket/profiler/libsmodel_injection.so": (
                    "83be54bd06e2cd82b2f6c17bbee6c925d049acae8d880242d4a5d5509a29e122"
                ),
            },
            "nvidia-cutlass-dsl-libs-cu13": {
                "nvidia_cutlass_dsl/cu13/lib/libcute_dsl_runtime.so": (
                    "2fa9809047485ae420ca99cab0678846de692e9608a179b0020834994311dd2f"
                ),
            },
            "nvidia-cuda-nvdisasm": {
                "nvidia/cu13/bin/nvdisasm": (
                    "5842e6adf9e232c9503a804915f158a576473e542577c070da3be49390474140"
                ),
            },
            "nvidia-cuda-nvrtc": {
                "nvidia/cu13/lib/libnvrtc.so.13": (
                    "e51d197b3b0d2d9d850d29977423e6ac60661d429a59c440fc04e52b6fc6750a"
                ),
                "nvidia/cu13/lib/libnvrtc-builtins.so.13.3": (
                    "7394c640e5761d13d2bbcdbc4b4c5dbac7cb53cd5bc732d78f8a5cb38638e913"
                ),
            },
        },
    }
}


def _profile_error(message):
    return RuntimeError(
        f"Official IKET profile validation failed: {message}. "
        f"Run the workload as `{_PROFILE_HINT}`."
    )


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_run_iket_entrypoint():
    executable = shutil.which("run-iket")
    if executable is None or not os.access(executable, os.X_OK):
        raise _profile_error("the run-iket executable is unavailable")
    entry_points = metadata.distribution("nvidia-cutlass-dsl-libs-base").entry_points
    if not any(
        item.group == "console_scripts"
        and item.name == "run-iket"
        and item.value == "iket.cli.main:entrypoint"
        for item in entry_points
    ):
        raise _profile_error("the run-iket entry point does not match CUTLASS DSL 4.6.1")


def _validate_injection_environment(expected_injection_digest):
    injection_value = os.environ.get("CUDA_INJECTION64_PATH")
    injection_path = Path(injection_value) if injection_value else None
    if injection_path is None or not injection_path.is_file():
        raise _profile_error("CUDA_INJECTION64_PATH was not supplied by run-iket")
    if _sha256(injection_path) != expected_injection_digest:
        raise _profile_error("CUDA_INJECTION64_PATH does not match the locked run-iket binary")

    config_value = os.environ.get("SMODEL_INJECTION_CONFIG")
    config_path = Path(config_value) if config_value else None
    if config_path is None or not config_path.is_file():
        raise _profile_error("SMODEL_INJECTION_CONFIG was not supplied by run-iket")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as err:
        raise _profile_error("SMODEL_INJECTION_CONFIG is not valid JSON") from err
    tool_name = config.get("toolName")
    if tool_name not in ("tracker", "iket"):
        raise _profile_error("SMODEL_INJECTION_CONFIG was not generated by run-iket profile")
    if tool_name == "tracker":
        return
    instrument_path = Path(config.get("toolConfig", {}).get("appInstrument", ""))
    if not instrument_path.is_file():
        raise _profile_error("run-iket did not provide an application instrumentation manifest")


def _validate_nvrtc_13_3():
    try:
        from cuda.bindings import nvrtc

        error, major, minor = nvrtc.nvrtcVersion()
    except (ImportError, OSError, RuntimeError) as err:
        raise _profile_error("CUDA NVRTC 13.3 is unavailable") from err
    if int(error) != 0 or (int(major), int(minor)) != (13, 3):
        raise _profile_error(f"CUDA NVRTC 13.3 is required, got {int(major)}.{int(minor)}")


def validate_official_environment():
    """Validate the exact external patcher profile before a CUBIN is loaded."""
    profile_name = os.environ.get(_PROFILE_ENV)
    if profile_name not in _OFFICIAL_PROFILES:
        raise _profile_error(f"{_PROFILE_ENV} must be set to cutlass-4.6.1, got {profile_name!r}")
    profile = _OFFICIAL_PROFILES[profile_name]
    distributions = {}
    for distribution_name, expected_version in profile["versions"].items():
        try:
            distribution = metadata.distribution(distribution_name)
        except metadata.PackageNotFoundError as err:
            raise _profile_error(
                f"{distribution_name}=={expected_version} is not installed"
            ) from err
        if distribution.version != expected_version:
            raise _profile_error(
                f"{distribution_name} must be {expected_version}, got {distribution.version}"
            )
        distributions[distribution_name] = distribution

    for distribution_name, expected_files in profile["files"].items():
        distribution = distributions[distribution_name]
        for relative_path, expected_digest in expected_files.items():
            path = Path(distribution.locate_file(relative_path))
            if not path.is_file():
                raise _profile_error(f"profile binary is missing: {relative_path}")
            actual_digest = _sha256(path)
            if actual_digest != expected_digest:
                raise _profile_error(
                    f"profile binary hash mismatch for {relative_path}: {actual_digest}"
                )

    injection_digest = profile["files"]["nvidia-cutlass-dsl-libs-base"][
        "nvidia_cutlass_dsl/dsl_packages/iket/profiler/libsmodel_injection.so"
    ]
    _validate_run_iket_entrypoint()
    _validate_injection_environment(injection_digest)
    _validate_nvrtc_13_3()
