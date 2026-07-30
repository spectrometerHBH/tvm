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
"""PTX logic and shift intrinsics."""

from ._schema import device_intrinsic
from .utils import parse_str

_SHL_TYPE_INFO = {
    "b16": ("unsigned short", "h", "uint16"),
    "b32": ("unsigned int", "r", "uint32"),
    "b64": ("unsigned long long", "l", "uint64"),
}


def _shl_type_info(ptx_type):
    ptx_type = parse_str(ptx_type)
    if ptx_type not in _SHL_TYPE_INFO:
        raise ValueError(
            f"Unsupported PTX shl type {ptx_type!r}; expected {sorted(_SHL_TYPE_INFO)}"
        )
    return ptx_type, *_SHL_TYPE_INFO[ptx_type]


def _shl_helper_name(_a, _b, ptx_type):
    ptx_type, _, _, _ = _shl_type_info(ptx_type)
    return f"tvm_builtin_ptx_shl_{ptx_type}"


def _shl_signature(_a, _b, ptx_type):
    _, c_type, _, _ = _shl_type_info(ptx_type)
    return f"({c_type} a, unsigned int b)"


def _shl_body(_a, _b, ptx_type):
    ptx_type, _, constraint, _ = _shl_type_info(ptx_type)
    return (
        f"    {_SHL_TYPE_INFO[ptx_type][0]} ret;\n"
        f'    asm("shl.{ptx_type} %0, %1, %2;"'
        f' : "={constraint}"(ret) : "{constraint}"(a), "r"(b));\n'
        "    return ret;"
    )


def _shl_return_type(_a, _b, ptx_type):
    _, c_type, _, _ = _shl_type_info(ptx_type)
    return c_type


def _shl_tvm_return_type(_a, _b, ptx_type):
    _, _, _, tvm_type = _shl_type_info(ptx_type)
    return tvm_type


device_intrinsic(
    "ptx_shl",
    helper_name=_shl_helper_name,
    c_signature=_shl_signature,
    body=_shl_body,
    n_attrs=1,
    return_type=_shl_return_type,
    tvm_return_type=_shl_tvm_return_type,
)
