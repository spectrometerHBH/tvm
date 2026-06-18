#!/usr/bin/env python3
"""Re-check suspicious ratio rows with paired bench --impls all."""

from __future__ import annotations

import json
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LOG_ROOT = ROOT / ".tir-bench" / "logs" / "suspicious-ratio-review"
SUSPICIOUS = ROOT / ".tir-bench" / "reports" / "suspicious-ratio-review.json"
HEAD_TIR = Path("/tmp/tir-head.json")
HEAD_REF = Path("/tmp/ref-head.json")
OUR = frozenset({"tir", "tirx"})


def _load_head() -> dict[tuple[str, str], dict[str, float]]:
    tir = json.loads(HEAD_TIR.read_text())
    ref = json.loads(HEAD_REF.read_text())
    ref_idx = {
        (r["kernel"], r.get("label") or r["config"]): r.get("impls") or {}
        for r in ref.get("results", [])
    }
    out: dict[tuple[str, str], dict[str, float]] = {}
    for r in tir.get("results", []):
        if r.get("status") != "ok":
            continue
        key = (r["kernel"], r.get("label") or r["config"])
        out[key] = {**(r.get("impls") or {}), **ref_idx.get(key, {})}
    return out


def _pick_ref(impls: dict[str, float]) -> str | None:
    refs = {k: v for k, v in impls.items() if k not in OUR and v > 0}
    return min(refs, key=refs.get) if refs else None


def _trim(vals: list[float]) -> float:
    s = sorted(vals)
    return statistics.mean(s[1:-1]) if len(s) >= 3 else statistics.mean(s)


def _spread(vals: list[float]) -> float:
    t = _trim(vals)
    return (max(vals) - min(vals)) / t * 100 if t else 999.0


def _run_paired(
    kernel: str,
    config: str,
    *,
    rounds: int,
    warmup: int,
    repeat: int,
    gpu: str,
    log_dir: Path,
) -> dict[str, list[float]]:
    env = {
        **dict(__import__("os").environ),
        "TVM_PATH": str(ROOT),
        "TIRX_KERNELS_PATH": str(ROOT.parent / "tirx-kernels-staging"),
        "PYTHONPATH": f"/tmp:{ROOT.parent / 'tirx-kernels-staging'}:{ROOT / 'python'}",
        "TVM_LIBRARY_PATH": str(ROOT / "build" / "lib"),
        "TIRX_KERNELS_STRICT": "1",
        "CUDA_VISIBLE_DEVICES": gpu,
    }
    by_impl: dict[str, list[float]] = {}
    for r in range(rounds):
        jpath = log_dir / f"r{r}.json"
        cmd = [
            sys.executable,
            "-m",
            "tirx_kernels.bench",
            "--kernel",
            kernel,
            "--config",
            config,
            "--warmup",
            str(warmup),
            "--repeat",
            str(repeat),
            "--impls",
            "all",
            "--json-file",
            str(jpath),
        ]
        subprocess.run(cmd, env=env, check=True, capture_output=True, text=True)
        impls = json.loads(jpath.read_text())["results"][0]["impls"]
        for k, sec in impls.items():
            by_impl.setdefault(k, []).append(sec * 1000)
    return by_impl


def _verdict(
    old_ratio_d: float,
    new_ratio: float,
    base_ratio: float,
    ours_spr: float,
    ref_spr: float,
) -> str:
    new_ratio_d = (new_ratio / base_ratio - 1) * 100 if base_ratio else 0
    if ours_spr > 5 or ref_spr > 5:
        return "still_noisy"
    if abs(old_ratio_d) >= 5 and abs(new_ratio_d) < 3:
        return "false_alarm"
    if abs(new_ratio_d) >= 5 and abs(old_ratio_d) < 3:
        return "confirmed"
    if abs(new_ratio_d - old_ratio_d) <= 2:
        return "consistent"
    return "shifted"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default="5")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--min-abs-ratio-d", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("-o", "--out", type=Path, default=ROOT / ".tir-bench/reports/suspicious-ratio-review-results.json")
    args = ap.parse_args()

    rows = json.loads(SUSPICIOUS.read_text())
    rows = [r for r in rows if abs(r["ratio_d"]) >= args.min_abs_ratio_d]
    if args.limit:
        rows = rows[: args.limit]

    head = _load_head()
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    results = []

    for i, row in enumerate(rows):
        k, c, ref = row["kernel"], row["config"], row["ref"]
        key = (k, c)
        tag = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{k}__{c}")[:120]
        log_dir = LOG_ROOT / tag
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[{i+1}/{len(rows)}] paired review {k}/{c} ({ref})", flush=True)

        base_impls = head.get(key, {})
        ours = "tir" if "tir" in base_impls else "tirx"
        base_ratio = base_impls[ref] / base_impls[ours] if ours in base_impls and ref in base_impls else 0

        try:
            by_impl = _run_paired(
                k, c, rounds=args.rounds, warmup=args.warmup, repeat=args.repeat, gpu=args.gpu, log_dir=log_dir
            )
        except subprocess.CalledProcessError as e:
            results.append({**row, "status": "fail", "error": (e.stderr or e.stdout or str(e))[-500:]})
            continue

        if ours not in by_impl or ref not in by_impl:
            results.append({**row, "status": "missing_impl"})
            continue

        ot, orf = _trim(by_impl[ours]), _trim(by_impl[ref])
        ospr, rspr = _spread(by_impl[ours]), _spread(by_impl[ref])
        new_ratio = orf / ot if ot else 0
        new_ratio_d = (new_ratio / base_ratio - 1) * 100 if base_ratio else 0
        verdict = _verdict(row["ratio_d"], new_ratio, base_ratio, ospr, rspr)

        results.append(
            {
                **row,
                "status": "ok",
                "paired_rounds": args.rounds,
                "ours_trim_ms": ot,
                "ref_trim_ms": orf,
                "base_ratio": base_ratio,
                "old_ratio_d_pct": row["ratio_d"],
                "new_ratio": new_ratio,
                "new_ratio_d_pct": new_ratio_d,
                "ours_spread_pct": ospr,
                "ref_spread_pct": rspr,
                "verdict": verdict,
            }
        )
        print(
            f"  old Δ{row['ratio_d']:+.1f}% -> paired Δ{new_ratio_d:+.1f}% "
            f"ratio {base_ratio:.3f}->{new_ratio:.3f} spread {ospr:.1f}/{rspr:.1f}% {verdict}",
            flush=True,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    counts: dict[str, int] = {}
    for r in results:
        counts[r.get("verdict", r.get("status", "?"))] = counts.get(r.get("verdict", r.get("status", "?")), 0) + 1
    print(f"written {args.out}")
    print("summary:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
