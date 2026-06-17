#!/usr/bin/env python3
"""Promote a tir-bench run JSON to a checked-in baseline and refresh baseline.md.

`baseline.md` is the human-facing rendered view of the baseline and is what
people actually read; it MUST stay in sync with `tir.json` / `ref.json`. Doing
the promotion as a bare `cp` leaves baseline.md stale — always promote through this helper,
which copies the run JSON over the chosen baseline(s) AND regenerates baseline.md
in one step.

Usage:
    # our-kernel baseline (from `run.py --impls ours`)
    python promote_baseline.py .tir-bench/runs/<id>.json --tir

    # reference baseline (from `run.py --impls baseline`)
    python promote_baseline.py .tir-bench/runs/<id>.json --ref

    # full `run.py --impls all` run: seed both at once
    python promote_baseline.py .tir-bench/runs/<id>.json --both
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "run_json",
        type=Path,
        help="run JSON to promote (e.g. .tir-bench/runs/18-stable.json)",
    )
    ap.add_argument("--tir", action="store_true", help="refresh tir.json (our-kernel baseline)")
    ap.add_argument("--ref", action="store_true", help="refresh ref.json (reference baseline)")
    ap.add_argument(
        "--both",
        action="store_true",
        help="refresh both (use for a full --impls all run)",
    )
    args = ap.parse_args()

    if not (args.tir or args.ref or args.both):
        ap.error("pick at least one of --tir / --ref / --both")
    if not args.run_json.exists():
        ap.error(f"run JSON not found: {args.run_json}")

    targets = []
    if args.tir or args.both:
        targets.append(HERE / "tir.json")
    if args.ref or args.both:
        targets.append(HERE / "ref.json")

    for dst in targets:
        shutil.copyfile(args.run_json, dst)
        print(f"[promote] {args.run_json} -> {dst.relative_to(HERE)}")

    # Always regenerate the human-facing baseline.md so it never drifts from the
    # JSON baselines. This is the whole reason to promote through this helper.
    subprocess.run(
        [sys.executable, str(HERE / "baseline_view.py")],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(f"[promote] regenerated {(HERE / 'baseline.md').relative_to(HERE)}")


if __name__ == "__main__":
    main()
