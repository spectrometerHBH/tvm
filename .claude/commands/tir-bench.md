---
description: "Pre-commit kernel regression benchmark (auto GPU selection + parallel sweep)"
argument-hint: "[--impls all|ours|baseline] [--cpu-workers N] [--filter SUBSTR] [--baseline PATH] [--threshold PCT] [--label STR] [--util-threshold PCT]"
allowed-tools: ["Bash", "Read"]
---

# tir-bench — local kernel regression check

Run the curated workload list in `.claude/commands/tir-bench/workloads.yaml`
on every free GPU in parallel, dump JSON, and diff against the previous run.

> **NEVER ACTIVELY SELECT A GPU FOR THIS run.py — IT SELECTS GPUs AUTOMATICALLY.**
> There is no `--gpus` flag. Do not set `CUDA_VISIBLE_DEVICES` to pin cards either.
> run.py probes every visible GPU, then on each acquire scans utilization and
> picks any card below `--util-threshold` (skipping cards in active use,
> requeueing if a neighbor bursts mid-run). Manually pinning defeats this and
> can land work on a busy card. If the machine is contended, let it run — busy
> cards are skipped and re-tried automatically; just re-run later for full coverage.

**Execution model.** Each workload runs in a fresh `python -m tirx_kernels.bench`
subprocess. Many run concurrently (`--cpu-workers`, default ~4× the usable-GPU
count) so the CPU-heavy phase (`import tvm` + kernel compile) overlaps across
workloads, while a per-physical-GPU advisory flock serializes the *measurement*
phase to one workload per card. Workloads are round-robin'd onto cards with no
*foreign* tenant (our own siblings sharing a card via the flock don't count); a
foreign tenant that appears is detected via per-PID `sm%` (`nvidia-smi pmon`)
and the workload requeued.

**Modes (`--impls`).** `all` (default) benches our kernel **and** the external
reference impls (deepgemm/cublas/flashinfer/torch) and writes the ratio +
restable reports — use this to (re)populate `tir.json` + `ref.json`. `ours` benches
**only our kernel** (skips reference setup + execution) for fast per-change
iteration; it diffs absolute-ms against the baseline's recorded `tir`/`tirx`
times (the contention-robust ratio diff needs current-run reference times, so
it's skipped). `baseline` benches only the references, to refresh their times
in the baseline.

**Args forwarded to run.py:** `$ARGUMENTS`

## Steps

1. Confirm `tirx-kernels` is importable and `nvidia-smi` works:
   ```bash
   python -c "import tirx_kernels; print('ok')"
   nvidia-smi --query-gpu=index,uuid --format=csv,noheader
   ```
2. Run the benchmark:
   ```bash
   python .claude/commands/tir-bench/run.py $ARGUMENTS
   ```
3. Read back the run JSON (`.tir-bench/latest.json`) and this run's report
   folder `.tir-bench/reports/<id>/` (or `reports/latest/`): the summary
   (`summary.md`) and — if a baseline existed — the absolute-ms regression
   report (`diff.md`), the ratio-based regression report (`ratio.md`,
   ours/ref normalised — robust to GPU-contention noise; the "ref Δ"
   column flags rows where the reference impl itself drifted >20%), and
   the post-restabilization `stable-ratio.md`.
   Summarise to the user: count of regressions / improvements / failures
   + the headline row for any regression beyond the threshold.

## Baseline

The diff compares against **two checked-in baselines**, joined at read time
(there is no combined file) — they have independent update cadences:

> **Always promote through `promote_baseline.py`, never a bare `cp`.** It copies
> the run JSON over the chosen baseline(s) **and** regenerates `baseline.md` — the
> human-facing rendered view everyone actually reads. A raw `cp` leaves
> `baseline.md` stale, and `.claude/` is excluded from pre-commit so nothing
> catches it.

- **`tir.json`** — our own kernel times (`tir`/`tirx`). Refresh whenever you
  change kernels: bench only ours and promote.
  ```bash
  python .claude/commands/tir-bench/run.py --impls ours   # re-bench your kernels
  python .claude/commands/tir-bench/promote_baseline.py .tir-bench/runs/<id>.json --tir
  ```
- **`ref.json`** — external reference times (deepgemm, cublas, flashinfer,
  torch). Rarely changes; refresh only when a reference library changes.
  ```bash
  python .claude/commands/tir-bench/run.py --impls baseline   # re-bench refs only
  python .claude/commands/tir-bench/promote_baseline.py .tir-bench/runs/<id>.json --ref
  ```

A reduced-impl run holds only the matching impls, so **promotion is a plain
copy, no merge**. A full `--impls all` run can seed both at once with
`promote_baseline.py .tir-bench/runs/<id>.json --both`. Override the
join per-run with `--baseline /path/to/combined.json`.

### Which commit does a baseline correspond to?

Each baseline records provenance two ways:

- `git:` — commit SHAs of the branch that ran the sweep. A squash/rebase merge
  rewrites them, so treat as a hint.
- `kernel_tree:` — merge-stable git *tree* SHAs of the codegen dirs
  (`tir:python/tvm/tirx`, `tirx-kernels:tirx_kernels`); content-addressed, so
  stable across merges. Confirm with `git rev-parse HEAD:python/tvm/tirx` /
  `git rev-parse HEAD:tirx_kernels`. `tir.json`'s provenance tracks your kernel
  code; `ref.json`'s tracks the reference-library versions.

The authoritative "which commit set a baseline" is the commit that last touched
`tir.json` / `ref.json` (`git log -1 -- .claude/commands/tir-bench/tir.json`).

## Outputs

```
.claude/commands/tir-bench/
├── run.py                    # the script
├── workloads.yaml            # curated (kernel, config) list
├── tir.json                  # our-kernel baseline   (refresh: --impls ours)
└── ref.json                  # reference baseline    (refresh: --impls baseline)

.tir-bench/                      # runtime artifacts (relative to cwd, regenerable)
├── runs/<id>.json               # this run's full result
├── runs/<id>.log                # live orchestrator log
├── latest.json / latest.log     # symlinks → most recent
├── reports/<id>/                # one folder per run (short names inside)
│   ├── summary.md               # per-run human-readable table (with baseline/ours ratio)
│   ├── diff.md                  # absolute-ms diff vs pinned baseline (if baseline exists)
│   ├── ratio.md                 # ratio-based diff (ours/ref) — robust to contention
│   └── stable-ratio.md          # ratio diff after restabilization
├── reports/latest -> <id>/      # symlink → most recent run's report folder
└── logs/<kernel>__<config>.log
```

`<id>` is a simple incrementing run number (1, 2, 3, …) — one more than the
highest existing numeric run in `runs/`. The wall-clock time lives in each
run JSON's `started_at` / `finished_at` fields.

## Exit codes

| Code | Meaning |
|------|---------|
| 0    | All workloads OK, no regression exceeded threshold (or no baseline). |
| 2    | Config error: no workloads to run / bad YAML. |
| 3    | One or more regressions above `--threshold` percent. |

## Editing workloads

`.claude/commands/tir-bench/workloads.yaml` carries the list. To pick from
the full set of valid (kernel, config) pairs on this machine:

```bash
python -m tirx_kernels.registry --format=benchrun --cc 10 | cut -d'|' -f1,2
```
