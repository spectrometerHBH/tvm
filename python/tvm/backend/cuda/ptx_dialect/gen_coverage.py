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
"""Thin generator 2: emit a markdown coverage table for the ptxd dialect.

Run manually::

    python -m tvm.backend.cuda.ptx_dialect.gen_coverage

Only imports :mod:`.table` (tvm-free).
"""

import sys

from .table import EFFECT_NAMES, TABLE


def generate() -> str:
    lines = [
        "# ptxd dialect coverage",
        "",
        "| instruction | modifiers | constraints | operands | returns | effect |",
        "|---|---|---|---|---|---|",
    ]
    for name in sorted(TABLE):
        e = TABLE[name]
        mods = (
            "<br>".join(
                f"`{s.name}` ∈ {{{', '.join(s.choices)}}}{' *(opt)*' if s.optional else ''}"
                for s in e.slots
            )
            or "—"
        )
        check_doc = (e.check.__doc__ or "").strip() if e.check is not None else "—"
        operands = ", ".join(f"{s.name}:{s.role}" for s in e.operands)
        returns = f"dtype of `{e.returns}`" if e.returns else "void"
        effect = EFFECT_NAMES.get(e.effect, str(e.effect))
        lines.append(f"| `{name}` | {mods} | {check_doc} | {operands} | {returns} | {effect} |")
    lines.append("")
    lines.append(f"{len(TABLE)} instruction families.")
    return "\n".join(lines) + "\n"


def main() -> None:
    sys.stdout.write(generate())


if __name__ == "__main__":
    main()
