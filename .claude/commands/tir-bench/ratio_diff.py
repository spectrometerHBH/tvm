#!/usr/bin/env python3
"""Ratio-based regression diff for tir-bench.

For each (kernel, config) we measure with multiple impls, compute the
ratio ref/ours where ref = fastest non-ours impl picked in baseline and
held fixed across runs. Higher ref/ours = ours is faster than ref =
better. Diff that ratio between baseline and current: positive ratio Δ
means we got faster vs ref (improvement), negative means slower
(regression).

Rationale: under GPU contention every impl slows by a similar factor,
so absolute-ms diffs are dominated by that noise. The ratio between
ours and a same-run reference is unchanged by uniform slowdown, so a
moving ratio is a real perf signal. Rows where the reference impl
itself drifted > 20% are flagged ⚠ — workload's environment was
unstable, so the ratio Δ is less trustworthy.

The report lists every comparable workload in a single table, sorted by
ratio Δ from most-improved to most-regressed (positive → negative).
Baseline workloads that were attempted this run but produced no comparable
measurement (failed, interfered, or missing an impl) are listed in a separate
"Not comparable in current run" section so lost coverage is never silent.

Usage:
    python ratio_diff.py [baseline.json] [current.json] [-o PATH]

Importable as `build_report(baseline_path, current)` for use from
`run.py`.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

OUR_IMPLS = {"tir", "tirx"}
DEFAULT_RATIO_THRESHOLD = 1.0


def index(payload: dict) -> dict[tuple[str, str], dict[str, float]]:
    """{(kernel, config) -> {impl -> ms}} for ok results."""
    out: dict[tuple[str, str], dict[str, float]] = {}
    for r in payload.get("results") or []:
        if r.get("status") != "ok":
            continue
        key = (r["kernel"], r.get("label") or r.get("config"))
        out[key] = dict(r.get("impls") or {})
    return out


def pick_ref(base_impls: dict[str, float]) -> str | None:
    """Pick the fastest non-ours impl from BASELINE; reused in current to
    keep ref fixed across runs."""
    refs = {i: ms for i, ms in base_impls.items() if i not in OUR_IMPLS and ms > 0}
    if not refs:
        return None
    return min(refs, key=lambda k: refs[k])


def build_report(
    baseline_path: Path | str,
    current: dict | Path | str,
    *,
    threshold_pct: float = DEFAULT_RATIO_THRESHOLD,
) -> tuple[str, int]:
    """Build the markdown report. `current` may be a loaded dict or a path.

    Returns (markdown, n_regressions_below_threshold) for run.py's exit code.
    """
    base_payload = json.loads(Path(baseline_path).read_text())
    if isinstance(current, (str, Path)):
        cur_payload = json.loads(Path(current).read_text())
        current_label = str(current)
    else:
        cur_payload = current
        current_label = "(in-memory)"

    base = index(base_payload)
    cur = index(cur_payload)

    # Status of every current-run result, including non-ok rows. index() keeps
    # only status=="ok", so a workload that failed/interfered this run is absent
    # from `cur` and would otherwise vanish from the report with no trace — the
    # only hint being a "comparable" count below the baseline size. Keep the full
    # record so we can explain *why* a baseline workload has no comparable
    # measurement this run instead of silently truncating coverage.
    cur_status: dict[tuple[str, str], dict] = {}
    for r in cur_payload.get("results") or []:
        cur_status[(r["kernel"], r.get("label") or r.get("config"))] = r

    rows: list[tuple[str, str, str, float, float, float, float, float]] = []
    skipped_no_ref: list[tuple[str, str]] = []
    # Baseline workloads attempted this run but yielding no comparable ratio
    # (failed, interfered, or ok-but-missing an impl). Workloads simply not in
    # this run's scope (e.g. a --filter subset) have no cur_status record and are
    # NOT listed, so filtered runs don't get spammed with the whole baseline.
    not_comparable: list[tuple[str, str, str]] = []
    for key, base_impls in base.items():
        ref = pick_ref(base_impls)
        ours_b = next((i for i in OUR_IMPLS if i in base_impls), None)
        if ref is None or ours_b is None:
            skipped_no_ref.append(key)
            continue
        if key not in cur:
            rec = cur_status.get(key)
            if rec is not None:  # attempted this run but not ok → surface it
                st = rec.get("status") or "?"
                err = (rec.get("error") or "").strip().splitlines()
                not_comparable.append((key[0], key[1], f"{st}: {err[0]}" if err else st))
            continue
        cur_impls = cur[key]
        if ours_b not in cur_impls or ref not in cur_impls:
            missing = ", ".join(i for i in (ours_b, ref) if i not in cur_impls)
            not_comparable.append((key[0], key[1], f"ok but missing impl(s): {missing}"))
            continue
        our_b_ms, ref_b_ms = base_impls[ours_b], base_impls[ref]
        our_c_ms, ref_c_ms = cur_impls[ours_b], cur_impls[ref]
        if min(our_b_ms, ref_b_ms, our_c_ms, ref_c_ms) <= 0:
            continue
        # ref/ours: higher = ours is faster than ref = better.
        base_ratio = ref_b_ms / our_b_ms
        cur_ratio = ref_c_ms / our_c_ms
        delta_pct = (cur_ratio - base_ratio) / base_ratio * 100.0
        ref_drift_pct = (ref_c_ms - ref_b_ms) / ref_b_ms * 100.0
        our_drift_pct = (our_c_ms - our_b_ms) / our_b_ms * 100.0
        rows.append((key[0], key[1], ref, base_ratio, cur_ratio, delta_pct,
                     ref_drift_pct, our_drift_pct))

    # Positive ratio Δ first (improvements), negative last (regressions).
    rows.sort(key=lambda r: -r[5])

    out = io.StringIO()
    def w(line: str = "") -> None:
        out.write(line + "\n")

    n_regressions = sum(1 for r in rows if r[5] <= -threshold_pct)
    n_improvements = sum(1 for r in rows if r[5] >= threshold_pct)

    w("# tir-bench ratio diff")
    w()
    w(f"- Baseline: `{baseline_path}`")
    w(f"- Current:  `{current_label}`")
    w("- Method: ref/ours per run (ref = fastest non-ours impl in baseline, "
      "fixed across runs). Higher ratio = ours is faster. Sorted by ratio Δ "
      "from improved → regressed.")
    w(f"- Summary: {len(rows)} comparable workloads; "
      f"{n_improvements} > +{threshold_pct:g}%, {n_regressions} < -{threshold_pct:g}%"
      + (f"; {len(not_comparable)} not comparable in current run (see below)"
         if not_comparable else "")
      + ". ⚠ = reference impl itself drifted >20% (less trustworthy).")
    w()

    if rows:
        w("| kernel | config | ref impl | base ref/ours | cur ref/ours | ratio Δ | ours Δ | ref Δ |")
        w("|---|---|---|---:|---:|---:|---:|---:|")
        for k, c, ref, br, cr, d, ref_d, our_d in rows:
            flag = " ⚠" if abs(ref_d) > 20 else ""
            w(f"| {k} | {c} | {ref} | {br:.3f} | {cr:.3f} | {d:+.1f}% | "
              f"{our_d:+.1f}% | {ref_d:+.1f}%{flag} |")
        w()

    if not_comparable:
        w(f"## Not comparable in current run ({len(not_comparable)})")
        w()
        w("_In baseline with a ref/ours pair, but produced no comparable "
          "measurement this run (failed, interfered, or missing an impl), so "
          "excluded from the ratio table above. Not a perf signal — usually a "
          "contention/OOM artifact — but flagged so lost coverage is never silent._")
        w()
        for k, c, reason in sorted(not_comparable):
            # OOM messages are huge single lines; keep the actionable head.
            reason = reason if len(reason) <= 160 else reason[:157] + "..."
            w(f"- `{k}/{c}` — {reason}")
        w()

    if skipped_no_ref:
        w(f"## Skipped — no comparable ref impl ({len(skipped_no_ref)})")
        w()
        for k, c in skipped_no_ref:
            w(f"- `{k}/{c}`")
        w()

    return out.getvalue(), n_regressions


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", nargs="?", default=".claude/commands/tir-bench/baseline.json")
    ap.add_argument("current", nargs="?", default=".tir-bench/latest.json")
    ap.add_argument("--threshold", type=float, default=DEFAULT_RATIO_THRESHOLD,
                    help="Ratio regression threshold in percent "
                         f"(default {DEFAULT_RATIO_THRESHOLD:g})")
    ap.add_argument("--output", "-o", type=Path, default=None,
                    help="Write report path (default: .tir-bench/reports/<current_ts>-ratio.md)")
    args = ap.parse_args()

    report, _ = build_report(args.baseline, args.current, threshold_pct=args.threshold)
    print(report)

    if args.output is not None:
        out_path = args.output
    else:
        cur_path = Path(args.current).resolve()
        reports_dir = cur_path.parent.parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        out_path = reports_dir / f"{cur_path.stem}-ratio.md"
    out_path.write_text(report)
    print(f"[ratio_diff] written: {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
