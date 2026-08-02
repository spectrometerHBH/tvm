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
"""Thin generator 1: emit a ``.pyi`` stub for the ``T.ptxd`` namespace.

Run manually (never at import time)::

    python -m tvm.backend.cuda.ptx_dialect.gen_stubs -o python/tvm/script/tirx.pyi

The output is a *module stub for* ``tvm.script.tirx``: that module is
virtual at runtime (dialect-registry ``__getattr__``), but a ``.pyi`` file
at ``python/tvm/script/tirx.pyi`` makes Pyright/Pylance resolve it — giving
VS Code completion for ``T.ptxd.`` chains. All other members fall back to
``Any`` via the stub's module ``__getattr__``, matching today's behavior.

Each family's chain class lists every slot token flatly, so the stub also
type-checks chains the runtime would reject (e.g. repeated tokens) — the
correct trade-off for a completion-only stub; legality is enforced at trace
time.

Only imports :mod:`.table` (tvm-free), so it runs without a built tvm.
Regenerate whenever the instruction table changes (a unit test diffs the
checked-in stub against this generator).
"""

import argparse
import sys
import textwrap

from .table import TABLE, InstructionEntry, escape_token


def _chain_class(entry: InstructionEntry) -> str:
    cls = f"_Chain_{entry.name}"
    tokens = sorted({tok for slot in entry.slots for tok in slot.choices})
    lines = [f"class {cls}:"]
    doc = "; ".join(
        f"{s.name}∈{{{','.join(s.choices)}}}{' (opt)' if s.optional else ''}" for s in entry.slots
    )
    if entry.check is not None and entry.check.__doc__:
        doc = f"{doc} — {entry.check.__doc__.strip()}" if doc else entry.check.__doc__.strip()
    # *args covers the printed round-trip form (trailing modifier strings,
    # positional pred) so re-parsed scripts type-check too.
    params = ["self", *(f"{s.name}: Any" for s in entry.operands), "*args: Any"]
    if not entry.has_dst:
        params.append("pred: Any = None")
    signature = f"def __call__({', '.join(params)}) -> None"
    # 4 indent + 3 opening quotes + text + 3 closing quotes must stay <= 100,
    # because ruff format collapses a one-line docstring onto a single line.
    doc_lines = textwrap.wrap(f"`{entry.name}` — {doc or '(no modifiers)'}", width=88)
    lines.append('    """' + doc_lines[0])
    lines.extend(f"    {line}" for line in doc_lines[1:])
    lines.append('    """')
    for tok in tokens:
        lines.append(f"    {escape_token(tok)}: {cls}")
    if len(signature) > 92:  # keep the generated stub within the repo line limit
        joined = ",\n        ".join(params)
        ret = signature[signature.rindex(")") + 1 :]
        signature = f"def __call__(\n        {joined},\n    ){ret}"
    lines.append(f"    {signature}: ...")
    return "\n".join(lines)


_ASF_HEADER = """\
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
# under the License.\
"""


def generate() -> str:
    out = [
        _ASF_HEADER,
        '"""Generated stub for T.ptxd — do not edit.',
        "",
        "Regenerate:",
        "  python -m tvm.backend.cuda.ptx_dialect.gen_stubs -o python/tvm/script/tirx.pyi",
        '"""',
        "",
        "from typing import Any",
        "",
    ]
    for entry in TABLE.values():
        out.append(_chain_class(entry))
        out.append("")
    out.append("class _PTXD:")
    for name in sorted(TABLE):
        out.append(f"    {escape_token(name)}: _Chain_{name}")
    out.append("    def __getitem__(self, text: str) -> Any: ...")
    out.append("")
    out.append("ptxd: _PTXD")
    out.append("")
    out.append("# Every other tvm.script.tirx member stays dynamically typed, as before.")
    out.append("def __getattr__(name: str) -> Any: ...")
    return _ruff_format("\n".join(out) + "\n")


def _ruff_format(text: str) -> str:
    """Normalize through ``ruff format`` so the generated file is stable under pre-commit.

    Without this the repo's ruff-format hook rewrites the checked-in stub and the
    freshness test can never pass. Falls back to the raw text if ruff is absent.
    """
    import shutil  # pylint: disable=import-outside-toplevel
    import subprocess  # pylint: disable=import-outside-toplevel

    if shutil.which("ruff") is None:
        return text
    done = subprocess.run(
        ["ruff", "format", "--stdin-filename", "tirx.pyi", "-"],
        input=text,
        capture_output=True,
        text=True,
        check=False,
    )
    return done.stdout if done.returncode == 0 else text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", "--output", default=None, help="output path (default: stdout)")
    args = parser.parse_args()
    text = generate()
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
