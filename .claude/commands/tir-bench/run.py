#!/usr/bin/env python3
"""tir-bench: pre-commit regression benchmark for TIRx kernels.

Methodology mirrors tirx-bench-ci's bench-run.sh: per-GPU utilization
polling + a lock to assign at most one workload per free GPU. GPU selection
is fully automatic — there is no --gpus flag on purpose (a human pinning
cards defeats the util gate and can land work on a busy card). Differences
from bench-ci: no build phase, no SQLite, no worktrees — we test the
working tree as-is and emit JSON + a markdown regression report.

Usage:
    python run.py [--workloads PATH] [--filter SUBSTR]
                  [--baseline PATH] [--threshold PCT] [--label STR]
                  [--util-threshold PCT]

Exit codes:
    0  no regressions (or no baseline yet)
    2  config error (no workloads / bad YAML)
    3  one or more regressions exceeded the threshold
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_WORKLOADS = SCRIPT_DIR / "workloads.yaml"
DEFAULT_BASELINE = SCRIPT_DIR / "baseline.json"  # pinned reference; user `cp <new>.json baseline.json` to promote
DEFAULT_REGRESSION_THRESHOLD = 1.0
POLL_INTERVAL = 5.0       # seconds between GPU re-checks when none is free
MONITOR_INTERVAL = 0.5    # seconds between nvidia-smi polls during a workload
MAX_INTERFERED_RETRIES = 5  # workloads that hit INTERFERED get requeued up to this many times
DEFAULT_UTIL_THRESHOLD = 10.0  # % GPU util at/above which a card counts as "actively computing"
# Why util, not PID-presence: on shared boxes other tenants routinely *park*
# processes that hold tens-to-hundreds of GiB of VRAM at 0% utilization. They
# aren't competing for SMs, so co-running our bench on such a card is fine.
# Gating on "any compute-app PID present" would reject every such card and
# starve the sweep; gating on utilization lets us share idle-but-resident cards
# while still avoiding cards where a neighbor is actually burning the GPU.

# Tiny real workload used to decide whether a GPU is actually usable.
# Catches: driver hangs, ECC errors when touching memory, cuBLAS init
# failures, MIG/cgroup restrictions, fragmentation surprises — issues that
# nvidia-smi "free" status alone won't surface.
PROBE_SCRIPT = r"""
import sys
try:
    import torch
    if not torch.cuda.is_available():
        print("PROBE_FAIL: torch.cuda.is_available()=False", file=sys.stderr)
        sys.exit(1)
    a = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    b = torch.randn(512, 512, device="cuda", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    del a, b, c
    torch.cuda.empty_cache()
except Exception as e:
    print(f"PROBE_FAIL: {type(e).__name__}: {e}", file=sys.stderr)
    sys.exit(1)
print("PROBE_OK")
"""


# ── Workload loading ─────────────────────────────────────────────────────────

def load_workloads(path: Path) -> list[dict]:
    data = yaml.safe_load(path.read_text()) or {}
    defaults = data.get("defaults") or {}
    out: list[dict] = []
    for entry in data.get("workloads") or []:
        if "kernel" not in entry or "config" not in entry:
            raise ValueError(f"workload missing kernel/config: {entry}")
        out.append({**defaults, **entry})
    return out


# ── GPU pool ─────────────────────────────────────────────────────────────────

class GpuPool:
    """Hand out free GPU indices to worker threads.

    Every acquire() re-queries nvidia-smi utilization to decide who is free
    right now: a card counts as taken only if its GPU utilization is at/above
    `util_threshold` (someone is actively computing) — a card merely *holding*
    VRAM at 0% util is fair game to co-run on. So a GPU that was pegged at
    sweep start and went idle later is reusable the moment its util drops. The
    broken-card probe is a separate startup step; by the time the pool is
    built, `allowed` already excludes broken cards.
    """

    def __init__(
        self,
        allowed: set[str] | None = None,
        util_threshold: float = DEFAULT_UTIL_THRESHOLD,
    ):
        self._owned: set[str] = set()
        self._lock = threading.Lock()
        self._allowed = allowed
        self.util_threshold = util_threshold

    @staticmethod
    def _nvidia_smi(args: list[str]) -> list[str]:
        try:
            out = subprocess.run(
                ["nvidia-smi", *args, "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
        except (subprocess.TimeoutExpired, OSError):
            # Transient nvidia-smi stall under cluster load: degrade to an empty
            # reading instead of killing the whole sweep. Callers treat an empty
            # utilization map as "no occupancy info this tick"; a real co-run
            # conflict is still caught by the per-PID interference check.
            return []
        return [l.strip() for l in out.stdout.splitlines() if l.strip()]

    def _all_gpus(self) -> list[tuple[str, str]]:
        rows = self._nvidia_smi(["--query-gpu=index,uuid"])
        result = []
        for line in rows:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                result.append((parts[0], parts[1]))
        return result

    def _busy_indices(self) -> set[str]:
        """GPU indices with at least one compute-app PID (anyone's). Kept for
        the informational startup banner only — selection uses _occupied_indices
        (utilization), since a PID may just be parking idle VRAM."""
        rows = self._nvidia_smi(["--query-compute-apps=gpu_uuid"])
        busy_uuids = {l for l in rows if l}
        return {idx for idx, uuid in self._all_gpus() if uuid in busy_uuids}

    def _utils(self) -> dict[str, float]:
        """Map GPU index -> current utilization.gpu (percent)."""
        rows = self._nvidia_smi(["--query-gpu=index,utilization.gpu"])
        out: dict[str, float] = {}
        for line in rows:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 2:
                try:
                    out[parts[0]] = float(parts[1])
                except ValueError:
                    pass
        return out

    def _occupied_indices(self) -> set[str]:
        """GPU indices actively computing (util >= threshold) — i.e. a real
        tenant is burning the GPU, so we should not co-run there. Idle cards
        holding only resident VRAM read ~0% util and are NOT occupied."""
        return {idx for idx, u in self._utils().items() if u >= self.util_threshold}

    def total_visible(self) -> int:
        gpus = self._all_gpus()
        if self._allowed is not None:
            gpus = [g for g in gpus if g[0] in self._allowed]
        return len(gpus)

    def acquire(self) -> str:
        """Block until a free GPU is found; return its index string.

        Re-queries nvidia-smi utilization on every loop iteration so that a
        GPU which was pegged when the previous workload acquired now counts as
        free once the other tenant's util drops below the threshold. A card
        that only holds resident VRAM (0% util) counts as free.
        """
        while True:
            with self._lock:
                occupied = self._occupied_indices()
                for idx, _uuid in self._all_gpus():
                    if self._allowed is not None and idx not in self._allowed:
                        continue
                    if idx in self._owned or idx in occupied:
                        continue
                    self._owned.add(idx)
                    return idx
            time.sleep(POLL_INTERVAL)

    def release(self, idx: str) -> None:
        with self._lock:
            self._owned.discard(idx)


# ── Tee stdout → run log ─────────────────────────────────────────────────────

class _Tee:
    """Write to multiple streams; flush on every write so the log is live.

    Locks per write so two threads' simultaneous writes don't interleave
    bytes. For atomic *lines*, callers should still hold _log_lock around
    the full print+flush sequence — see log() below.
    """

    def __init__(self, *streams):
        self._streams = streams
        self._lock = threading.Lock()

    def write(self, s):
        with self._lock:
            for st in self._streams:
                st.write(s)
                st.flush()
        return len(s)

    def flush(self):
        with self._lock:
            for st in self._streams:
                st.flush()


# Thread-safe one-liner emitter. `print()` calls file.write() multiple times
# (once for the message, once for the trailing newline), so without this
# lock concurrent prints from worker threads can interleave halfway through
# a line. Use log() for any [tir-bench] status print from a worker thread.
_log_lock = threading.Lock()


def log(msg: str) -> None:
    with _log_lock:
        print(msg, flush=True)


# ── GPU probe ────────────────────────────────────────────────────────────────

def probe_gpu(idx: str, timeout: float = 60.0) -> tuple[bool, str]:
    """Run PROBE_SCRIPT on a single GPU. Returns (ok, error_message)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = idx
    try:
        proc = subprocess.run(
            [sys.executable, "-c", PROBE_SCRIPT],
            env=env, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"probe timed out after {timeout:.0f}s"
    except Exception as e:
        return False, repr(e)
    if proc.returncode == 0 and "PROBE_OK" in proc.stdout:
        return True, ""
    msg = (proc.stderr or proc.stdout).strip().splitlines()
    return False, msg[-1] if msg else f"exit {proc.returncode}"


def detect_usable_gpus(
    candidates: list[str], probe_timeout: float
) -> tuple[set[str], dict[str, str]]:
    """Probe candidates in parallel. Returns (usable_set, failures)."""
    usable: set[str] = set()
    failures: dict[str, str] = {}
    if not candidates:
        return usable, failures
    with ThreadPoolExecutor(max_workers=len(candidates)) as ex:
        futs = {ex.submit(probe_gpu, idx, probe_timeout): idx for idx in candidates}
        for fut in as_completed(futs):
            idx = futs[fut]
            ok, err = fut.result()
            if ok:
                usable.add(idx)
                log(f"[tir-bench]   gpu {idx}: ok")
            else:
                failures[idx] = err
                log(f"[tir-bench]   gpu {idx}: FAIL — {err}")
    return usable, failures


# ── Workload execution ───────────────────────────────────────────────────────

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _gpu_uuid_of(idx: str) -> str | None:
    """Look up the UUID for a GPU index via nvidia-smi."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == idx:
            return parts[1]
    return None


def _pids_on_gpu(uuid: str) -> set[int]:
    """Set of PIDs currently using the given GPU UUID."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return set()
    pids: set[int] = set()
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[1] == uuid:
            try:
                pids.add(int(parts[0]))
            except ValueError:
                pass
    return pids


def _pid_sm_on_gpu(gpu_index: str) -> dict[int, float]:
    """Map PID -> sm-utilization (%) for every compute process on the given
    physical GPU, via `nvidia-smi pmon`.

    This is the signal that separates a neighbor *actively burning the GPU*
    from one merely *parking resident VRAM* at 0% sm — and, crucially, it is
    per-process, so it stays meaningful while our own kernel pegs the
    device-level utilization. A single `pmon -c 1` snapshot is ~0.15s here.

    pmon `-s u` columns: gpu  pid  type  sm  mem  enc  dec  jpg  ofa  command.
    Inactive rows show "-" for pid/sm; those are skipped.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "pmon", "-i", str(gpu_index), "-c", "1", "-s", "u"],
            capture_output=True, text=True, timeout=8,
        ).stdout
    except Exception:
        return {}
    result: dict[int, float] = {}
    for line in out.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) < 4:
            continue
        try:
            pid = int(fields[1])
            sm = float(fields[3])
        except ValueError:
            continue  # pid or sm is "-" (no active process this sample)
        result[pid] = sm
    return result


def _active_strangers(gpu_index: str, our_pids: set[int], sm_threshold: float) -> dict[int, float]:
    """PIDs on `gpu_index` that are NOT ours and whose sm-util >= threshold.

    Empty result == no neighbor is actively computing right now, so an
    idle-but-resident squatter (sm 0) does not count as interference and we
    are free to share the card."""
    return {
        pid: sm
        for pid, sm in _pid_sm_on_gpu(gpu_index).items()
        if pid not in our_pids and sm >= sm_threshold
    }


def _our_process_tree(root_pid: int) -> set[int]:
    """Set of PIDs in the process tree rooted at root_pid (inclusive).

    Replaces grace-period "ours" accumulation: rather than guessing that
    every PID seen on the GPU within the first N seconds is ours, we
    actually walk the PPID chain via /proc and only call a PID "ours" if
    it's a descendant of our subprocess. Anything else on our GPU = intruder.
    """
    children: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                # /proc/PID/stat: pid (comm) state ppid ...
                # comm can contain spaces & parens, so split from the last ')'
                data = f.read()
            rparen = data.rfind(")")
            fields = data[rparen + 2:].split()
            ppid = int(fields[1])
            children.setdefault(ppid, []).append(int(entry))
        except (OSError, ValueError, IndexError):
            continue
    ours = {root_pid}
    stack = [root_pid]
    while stack:
        p = stack.pop()
        for c in children.get(p, ()):
            if c not in ours:
                ours.add(c)
                stack.append(c)
    return ours


def _run_subprocess_monitored(
    cmd: list[str],
    env: dict[str, str],
    cwd: str,
    log_path: Path,
    gpu_index: str,
    monitor_interval: float,
    sm_threshold: float,
) -> tuple[int, bool, list[int]]:
    """Spawn `cmd` on the assigned GPU and watch for *active* intruders.

    Returns (returncode, interfered, intruder_pids).

    Interference == another tenant is actually computing on our card, i.e. a
    PID that is not in our process tree has sm-utilization >= `sm_threshold`.
    A neighbor that only parks resident VRAM at 0% sm is NOT interference — we
    deliberately co-run with those (that is the whole point of the util gate).

    Two-stage protection, both using per-PID sm-util (`nvidia-smi pmon`):

    1. **Pre-spawn check**: if any stranger is already actively computing,
       someone grabbed the card between pool.acquire() and now (or an
       idle-looking card just woke up). Don't launch — return INTERFERED so
       the dispatcher requeues this workload.

    2. **Per-poll check**: at every `monitor_interval`, take the per-PID sm
       map, drop our process tree (walked via /proc PPID chain), and if any
       remaining PID is at/above the sm threshold, SIGTERM the subprocess.
       This catches a brand-new intruder *and* a resident neighbor that
       bursts its own sm mid-run — per-PID sm stays meaningful even while our
       own kernel pegs the device-level utilization.
    """
    if gpu_index:
        pre = _active_strangers(gpu_index, set(), sm_threshold)
        if pre:
            with open(log_path, "w") as lf:
                lf.write(f"RACE_LOST: pre-spawn check — active strangers {pre}\n")
            return -1, True, sorted(pre)

    with open(log_path, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=lf, stderr=subprocess.STDOUT)
    intruders: list[int] = []
    try:
        while True:
            try:
                proc.wait(timeout=monitor_interval)
                break  # subprocess exited normally
            except subprocess.TimeoutExpired:
                pass
            if not gpu_index:
                continue
            ours = _our_process_tree(proc.pid)
            active = _active_strangers(gpu_index, ours, sm_threshold)
            if active:
                intruders = sorted(active)
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise
    return proc.returncode, bool(intruders), intruders


def run_one(
    workload: dict,
    pool: GpuPool,
    log_dir: Path,
    *,
    no_monitor: bool = False,
) -> dict:
    kernel = workload["kernel"]
    config = workload["config"]
    warmup = workload.get("warmup")
    repeat = workload.get("repeat")
    timer = workload.get("timer")

    gpu = pool.acquire()
    started = now_iso()
    label = f"{kernel}/{config}"
    worker = threading.current_thread().name
    log(f"[tir-bench] {started} {worker} gpu={gpu} START {label}")

    json_tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    json_tmp.close()
    log_path = log_dir / f"{kernel}__{config}.log"

    cmd = [
        sys.executable, "-m", "tirx_kernels.bench",
        "--kernel", kernel, "--config", config,
        "--json-file", json_tmp.name,
    ]
    if warmup is not None:
        cmd += ["--warmup", str(warmup)]
    if repeat is not None:
        cmd += ["--repeat", str(repeat)]
    if timer is not None:
        cmd += ["--timer", timer]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    # Each workload gets its own scratch cwd so concurrent runs don't race on
    # proton's <proton_name>.hatchet file.
    workdir = tempfile.mkdtemp(prefix=f"tir-bench-{kernel}-{config}-")

    record: dict = {
        "kernel": kernel,
        "config": config,
        "gpu": gpu,
        "started_at": started,
    }
    interfered = False
    intruder_pids: list[int] = []
    try:
        # Pass the physical GPU index (not "" ) only when monitoring is on;
        # the monitor uses per-PID sm-util (pmon) keyed by this index.
        monitor_idx = "" if no_monitor else gpu
        returncode, interfered, intruder_pids = _run_subprocess_monitored(
            cmd, env, workdir, log_path, monitor_idx, MONITOR_INTERVAL,
            pool.util_threshold,
        )
        if interfered:
            record["status"] = "INTERFERED"
            record["intruder_pids"] = intruder_pids
            record["error"] = f"gpu {gpu}: intruder PIDs {intruder_pids}"
        elif returncode != 0:
            tail = "\n".join(log_path.read_text().splitlines()[-30:])
            record["status"] = "FAIL"
            record["error"] = f"exit {returncode}\n{tail}"
        else:
            payload = json.loads(Path(json_tmp.name).read_text())
            rows = payload.get("results") or []
            match = next(
                (r for r in rows if r.get("kernel") == kernel and r.get("label") == config),
                None,
            )
            if match is None:
                record["status"] = "FAIL"
                record["error"] = f"no matching row in bench JSON ({len(rows)} rows)"
            else:
                record.update(match)
                record.setdefault("status", "ok")
    except Exception as e:
        record["status"] = "FAIL"
        record["error"] = repr(e)
    finally:
        try:
            os.unlink(json_tmp.name)
        except FileNotFoundError:
            pass
        shutil.rmtree(workdir, ignore_errors=True)
        pool.release(gpu)

    record["finished_at"] = now_iso()
    status = record.get("status", "ok")
    impls = record.get("impls") or {}
    impl_str = ", ".join(f"{k}={v:.3f}ms" for k, v in impls.items())
    if interfered:
        # Make INTERFERED stand out — easy to spot when scrolling.
        log("[tir-bench] " + "*" * 70)
        log(f"[tir-bench] *** INTERFERED *** {worker} gpu={gpu} {label}")
        log(f"[tir-bench] ***   intruder PIDs on gpu {gpu}: {intruder_pids}")
        log("[tir-bench] ***   subprocess killed, will be requeued by dispatcher")
        log("[tir-bench] " + "*" * 70)
    else:
        log(f"[tir-bench] {record['finished_at']} {worker} gpu={gpu} {status:4s} {label} {impl_str}")
    return record


# ── Output ───────────────────────────────────────────────────────────────────

def git_label(repo: Path) -> str | None:
    if not repo.exists():
        return None
    try:
        sha = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short=8", "HEAD"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if not sha:
            return None
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return None


def _tir_repo_root() -> Path | None:
    """The git root containing this script (the tir repo, when checked in)."""
    for p in [SCRIPT_DIR, *SCRIPT_DIR.parents]:
        if (p / ".git").exists():
            return p
    return None


def _module_repo_root(import_name: str) -> Path | None:
    """Git root of an importable package, if it's a local checkout."""
    try:
        mod = __import__(import_name)
    except Exception:
        return None
    pkg_file = getattr(mod, "__file__", None)
    if not pkg_file:
        try:
            paths = list(getattr(mod, "__path__", []) or [])
            if paths:
                pkg_file = str(Path(paths[0]) / "__init__.py")
        except Exception:
            pass
    if not pkg_file:
        return None
    for p in [Path(pkg_file).resolve().parent, *Path(pkg_file).resolve().parents]:
        if (p / ".git").exists():
            return p
    return None


def collect_repo_git() -> dict[str, str | None]:
    """SHAs for the three repos involved: tir (where this script lives),
    tirx-kernels (via its installed package), tirx-bench-ci (sibling of tir)."""
    tir_root = _tir_repo_root()
    tirx_root = _module_repo_root("tirx_kernels")
    bench_ci_root: Path | None = None
    if tir_root is not None:
        candidate = tir_root.parent / "tirx-bench-ci"
        if (candidate / ".git").exists():
            bench_ci_root = candidate
    return {
        "tir": git_label(tir_root) if tir_root else None,
        "tirx-kernels": git_label(tirx_root) if tirx_root else None,
        "tirx-bench-ci": git_label(bench_ci_root) if bench_ci_root else None,
    }


def collect_kernel_fingerprint() -> dict[str, str | None]:
    """Merge-stable content fingerprints (git *tree* SHAs) of the source that
    determines kernel codegen + perf.

    The commit SHAs in ``collect_repo_git`` are rewritten by a squash/rebase
    merge, so a baseline that records only commit SHAs can't be mapped back to a
    mainline commit afterwards. A git tree SHA is content-addressed (Merkle): it
    is identical before and after a merge as long as the directory's content is
    unchanged. Confirm a checkout matches a recorded baseline with
    ``git rev-parse HEAD:<path>``.
    """
    tir_root = _tir_repo_root()
    tirx_root = _module_repo_root("tirx_kernels")

    def _tree(root: Path | None, path: str) -> str | None:
        if root is None:
            return None
        try:
            out = subprocess.run(
                ["git", "-C", str(root), "rev-parse", f"HEAD:{path}"],
                capture_output=True, text=True, timeout=5,
            )
        except (subprocess.TimeoutExpired, OSError):
            return None
        return out.stdout.strip() or None

    return {
        "tir:python/tvm/tirx": _tree(tir_root, "python/tvm/tirx"),
        "tirx-kernels:tirx_kernels": _tree(tirx_root, "tirx_kernels"),
    }


# Packages used as baselines in workloads.yaml — anything our regression
# numbers compare against, so the recorded version pins the comparison.
BASELINE_PACKAGES = ["torch", "deep_gemm", "flashinfer", "flash_attn"]


def package_provenance(import_name: str) -> dict | None:
    """Probe a Python package: version + (if editable git install) repo + SHA.

    Returns None when neither the package nor distribution metadata exists.
    """
    def _record_git(path: Path, info: dict) -> None:
        try:
            root = subprocess.run(
                ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if not root:
                return
            sha = subprocess.run(
                ["git", "-C", root, "rev-parse", "--short=8", "HEAD"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            if not sha:
                return
            dirty = subprocess.run(
                ["git", "-C", root, "status", "--porcelain"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip()
            info["git_dir"] = root
            info["git_sha"] = sha + ("-dirty" if dirty else "")
        except Exception:
            pass

    dists: list[str] = []
    try:
        from importlib.metadata import distribution as _probe_dist
        _probe_dist(import_name)
        dists.append(import_name)
    except Exception:
        pass
    try:
        from importlib.metadata import packages_distributions
        for dist_name in packages_distributions().get(import_name) or []:
            if dist_name not in dists:
                dists.append(dist_name)
    except Exception:
        pass
    if not dists:
        dists = [import_name]

    mod = None
    try:
        mod = __import__(import_name)
    except Exception as e:
        pass
    info: dict = {"importable": mod is not None}
    # Version: prefer __version__, else importlib.metadata. Top-level import
    # name and the distribution name often disagree (e.g. flash_attn ↔
    # flash-attn-4) — use packages_distributions() to bridge.
    version = getattr(mod, "__version__", None) if mod is not None else None
    if version is None:
        try:
            from importlib.metadata import version as _meta_version
            for d in dists:
                try:
                    version = _meta_version(d)
                    if version is not None:
                        info["dist"] = d
                        break
                except Exception:
                    continue
        except Exception:
            pass
    if version is not None:
        info["version"] = str(version)
    if import_name == "torch":
        cuda = getattr(getattr(mod, "version", None), "cuda", None)
        git_v = getattr(getattr(mod, "version", None), "git_version", None)
        if cuda:
            info["cuda"] = str(cuda)
        if git_v:
            info["torch_git_version"] = str(git_v)
    # PEP 610 direct_url.json: when a package was `pip install -e <path>` or
    # `pip install <path>`, pip writes the source path/URL into the dist-info.
    # This catches the editable case (the package lives outside the repo it
    # was built from, so the __file__ walk below misses it). dist resolution:
    # prefer `info["dist"]` if we set it above, else default to import_name.
    try:
        from importlib.metadata import distribution as _meta_dist
        dist = None
        for dist_name in [info.get("dist"), *dists, import_name]:
            if not dist_name:
                continue
            try:
                dist = _meta_dist(dist_name)
                info.setdefault("dist", dist.metadata["Name"])
                break
            except Exception:
                continue
        if dist is not None:
            direct_url_text = dist.read_text("direct_url.json")
            if direct_url_text:
                direct = json.loads(direct_url_text)
                url = direct.get("url") or ""
                if url.startswith("file://"):
                    src_path = Path(url[len("file://"):]).resolve()
                    info["source_dir"] = str(src_path)
                    if direct.get("dir_info", {}).get("editable"):
                        info["editable"] = True
                    _record_git(src_path, info)
    except Exception:
        pass
    if mod is None:
        return info if "version" in info or "source_dir" in info else None
    # Resolve a directory we can git-probe. Namespace packages and some
    # __init__.py-less namespaces set mod.__file__ to None — fall back to
    # __path__[0] then to a known submodule's file.
    pkg_file = getattr(mod, "__file__", None)
    if not pkg_file:
        try:
            paths = list(getattr(mod, "__path__", []) or [])
            if paths:
                pkg_file = str(Path(paths[0]) / "__init__.py")
        except Exception:
            pass
    if not pkg_file:
        # Last resort: try to import a likely submodule with a real file.
        for sub in (".cute", ".csrc", ".jit_kernels", ".jit"):
            try:
                submod = __import__(import_name + sub, fromlist=["__file__"])
                if getattr(submod, "__file__", None):
                    pkg_file = submod.__file__
                    break
            except Exception:
                continue
    if pkg_file:
        pkg_dir = Path(pkg_file).resolve().parent
        # Walk up looking for a git repo. .git can be a dir (regular clone)
        # or a file (worktree); both are fine for `git rev-parse`.
        _record_git(pkg_dir, info)
    return info


def collect_baseline_provenance() -> dict:
    return {
        name: package_provenance(name) or {"installed": False}
        for name in BASELINE_PACKAGES
    }


def write_run(
    out_dir: Path,
    stamp: str,
    results: list[dict],
    label: str | None,
    probe: dict | None = None,
) -> Path:
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": stamp,
        "label": label,
        "git": collect_repo_git(),
        "kernel_tree": collect_kernel_fingerprint(),
        "baselines": collect_baseline_provenance(),
        "probe": probe or {},
        "results": results,
    }
    path = runs_dir / f"{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


BASELINE_IMPL_BY_KERNEL = {
    "fp16_bf16_gemm": "torch-cublas",
    "fp8_blockwise_gemm": "deepgemm",
    "nvfp4_gemm": "flashinfer",
    "flash_attention4": "flashattn_sm100",
    "deepgemm_sm100_fp8_mqa_logits": "deepgemm",
    "deepgemm_sm100_fp4_mqa_logits": "deepgemm",
}


def _our_impl(row_impls: dict) -> str | None:
    """Pick our impl ('tir' or 'tirx') from a row's impls dict."""
    for name in ("tir", "tirx"):
        if name in row_impls:
            return name
    return None


def write_summary(out_dir: Path, current: dict) -> Path:
    """Human-readable per-run report, grouped by kernel.

    Times are in µs to match the existing tir-bench doc convention. Per row:
    config, one column per impl present in that kernel, baseline/ours ratio
    (against the kernel's reference impl from BASELINE_IMPL_BY_KERNEL),
    then attempt + gpu.
    """
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    stamp = current["timestamp"]
    lines: list[str] = []
    lines.append(f"# tir-bench run {stamp}")
    lines.append("")
    label = current.get("label") or "-"
    git = current.get("git") or {}
    lines.append(f"- label: `{label}`")
    lines.append(
        f"- git: tir=`{git.get('tir') or '-'}`  "
        f"tirx-kernels=`{git.get('tirx-kernels') or '-'}`  "
        f"tirx-bench-ci=`{git.get('tirx-bench-ci') or '-'}`"
    )
    statuses: dict[str, int] = {}
    for r in current.get("results") or []:
        s = r.get("status") or "?"
        statuses[s] = statuses.get(s, 0) + 1
    status_line = ", ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
    lines.append(f"- status: {status_line} (over {sum(statuses.values())} workloads)")
    lines.append("")

    baselines = current.get("baselines") or {}
    if baselines:
        lines.append("## Baseline impl provenance")
        lines.append("")
        for name, info in sorted(baselines.items()):
            if not info or info.get("installed") is False:
                lines.append(f"- `{name}`: not installed")
                continue
            bits = []
            if "version" in info:
                bits.append(f"v{info['version']}")
            if "cuda" in info:
                bits.append(f"cuda={info['cuda']}")
            if "torch_git_version" in info:
                bits.append(f"torch_git={info['torch_git_version'][:12]}")
            if "git_sha" in info:
                bits.append(f"@`{info['git_sha']}`")
            if "git_dir" in info:
                bits.append(f"({info['git_dir']})")
            lines.append(f"- `{name}`: {' '.join(bits) if bits else '?'}")
        lines.append("")

    # Group by kernel
    by_kernel: dict[str, list[dict]] = {}
    for r in current.get("results") or []:
        by_kernel.setdefault(r["kernel"], []).append(r)

    for kernel in sorted(by_kernel):
        rows = sorted(by_kernel[kernel], key=lambda r: r.get("label") or r.get("config") or "")
        # Discover all impl names that appear in this kernel
        impl_names: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for impl in (r.get("impls") or {}):
                if impl not in seen:
                    seen.add(impl)
                    impl_names.append(impl)
        impl_names.sort()
        baseline_impl = BASELINE_IMPL_BY_KERNEL.get(kernel)
        # Determine "ours" impl name once for the whole kernel (constant per kernel)
        ours_impl = None
        for r in rows:
            ours_impl = _our_impl(r.get("impls") or {})
            if ours_impl:
                break
        ratio_label = (
            f"{baseline_impl}/{ours_impl}"
            if baseline_impl and ours_impl
            else "ratio"
        )
        lines.append(f"## `{kernel}`")
        if baseline_impl and ours_impl:
            lines.append("")
            lines.append(f"_baseline impl_: `{baseline_impl}` · _ours_: `{ours_impl}` · _ratio_ = baseline/ours · `>1` means ours is faster")
        lines.append("")
        # Table header
        header = ["config", *impl_names, ratio_label, "attempt", "gpu"]
        align = ["---"] + ["---:"] * len(impl_names) + ["---:", "---:", "---:"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(align) + "|")
        for r in rows:
            cfg = r.get("label") or r.get("config") or "?"
            status = r.get("status", "ok")
            impls = r.get("impls") or {}
            row = [cfg]
            for impl in impl_names:
                ms = impls.get(impl)
                row.append(f"{ms*1000:.2f}us" if ms is not None else "—")
            # Ratio column
            ratio_cell = "—"
            if baseline_impl and ours_impl:
                base_ms = impls.get(baseline_impl)
                ours_ms = impls.get(ours_impl)
                if base_ms is not None and ours_ms is not None and ours_ms > 0:
                    ratio = base_ms / ours_ms
                    # Bold values that flag a regression risk (we're slower)
                    ratio_cell = f"**{ratio:.3f}**" if ratio < 1.0 else f"{ratio:.3f}"
            row.append(ratio_cell)
            if status != "ok":
                row[0] = f"{cfg} **[{status}]**"
            row.append(str(r.get("attempt", 1)))
            row.append(str(r.get("gpu", "-")))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    path = reports_dir / f"{stamp}-summary.md"
    path.write_text("\n".join(lines))
    return path


def _flatten(payload: dict) -> dict[tuple[str, str, str], float]:
    """{(kernel, config, impl) -> avg_ms} for all ok results."""
    out: dict[tuple[str, str, str], float] = {}
    for r in payload.get("results") or []:
        if r.get("status") != "ok":
            continue
        for impl, ms in (r.get("impls") or {}).items():
            out[(r["kernel"], r.get("label") or r.get("config"), impl)] = ms
    return out


def diff_report(baseline_path: Path, current: dict, threshold_pct: float) -> tuple[str, int]:
    base = json.loads(baseline_path.read_text())
    base_idx = _flatten(base)
    cur_idx = _flatten(current)

    regressions: list[tuple] = []
    improvements: list[tuple] = []
    unchanged: list[tuple] = []
    new_rows: list[tuple] = []

    for key, ms in cur_idx.items():
        if key not in base_idx:
            new_rows.append((key, ms))
            continue
        old = base_idx[key]
        if old <= 0:
            continue
        delta = (ms - old) / old * 100.0
        row = (key, old, ms, delta)
        if delta >= threshold_pct:
            regressions.append(row)
        elif delta <= -threshold_pct:
            improvements.append(row)
        else:
            unchanged.append(row)

    failed = [r for r in (current.get("results") or []) if r.get("status") != "ok"]

    def fmt_table(title: str, rows: list[tuple]) -> list[str]:
        if not rows:
            return []
        lines = [f"## {title} ({len(rows)})", "",
                 "| kernel | config | impl | baseline (ms) | current (ms) | Δ |",
                 "|---|---|---|---:|---:|---:|"]
        for (k, c, impl), old, new, d in sorted(rows, key=lambda r: -abs(r[3])):
            lines.append(f"| {k} | {c} | {impl} | {old:.4f} | {new:.4f} | {d:+.2f}% |")
        lines.append("")
        return lines

    md: list[str] = []
    md.append("# tir-bench regression report")
    md.append("")
    md.append(f"- Current:  `{current['timestamp']}` ({current.get('label') or '-'})")
    md.append(f"- Baseline: `{base['timestamp']}` ({base.get('label') or '-'})  from `{baseline_path.name}`")
    md.append(f"- Threshold: ±{threshold_pct:.1f}%")
    md.append("")
    md.append(
        f"**Summary** — regressions: {len(regressions)}, "
        f"improvements: {len(improvements)}, unchanged: {len(unchanged)}, "
        f"failed: {len(failed)}, new: {len(new_rows)}"
    )
    md.append("")
    md += fmt_table("Regressions", regressions)
    md += fmt_table("Improvements", improvements)

    if failed:
        md.append(f"## Failed ({len(failed)})")
        md.append("")
        for r in failed:
            first = (r.get("error") or "?").splitlines()[0]
            md.append(f"- `{r['kernel']}/{r.get('label') or r.get('config')}`: {first}")
        md.append("")

    if new_rows:
        md.append(f"## New (no baseline) ({len(new_rows)})")
        md.append("")
        md.append("| kernel | config | impl | current (ms) |")
        md.append("|---|---|---|---:|")
        for (k, c, impl), ms in new_rows:
            md.append(f"| {k} | {c} | {impl} | {ms:.4f} |")
        md.append("")

    return "\n".join(md), len(regressions)


# ── Main ─────────────────────────────────────────────────────────────────────

def _drifted_workloads(
    baseline_path: Path, current: dict, threshold_pct: float
) -> list[tuple[str, str]]:
    """Return (kernel, config) keys where |ratio Δ vs baseline| > threshold.

    The ratio is ref/ours (ref = fastest non-ours impl in baseline, fixed
    across runs) — same convention as ratio_diff.py, so a row is flagged
    when either ours or ref moved significantly. Skips workloads with no
    comparable ours/ref pair in both runs.
    """
    sys.path.insert(0, str(SCRIPT_DIR))
    from ratio_diff import OUR_IMPLS, index as ratio_index, pick_ref  # noqa: E402

    base = ratio_index(json.loads(Path(baseline_path).read_text()))
    cur = ratio_index(current)
    drifted: list[tuple[str, str]] = []
    for key, base_impls in base.items():
        ref = pick_ref(base_impls)
        ours = next((i for i in OUR_IMPLS if i in base_impls), None)
        if ref is None or ours is None or key not in cur:
            continue
        ci = cur[key]
        if ref not in ci or ours not in ci:
            continue
        ob, rb = base_impls[ours], base_impls[ref]
        oc, rc = ci[ours], ci[ref]
        if min(ob, rb, oc, rc) <= 0:
            continue
        delta = (rc / oc) - (rb / ob)
        if abs(delta) / (rb / ob) * 100.0 > threshold_pct:
            drifted.append(key)
    return drifted


def _bench_median(
    workloads: list[dict],
    reps: int,
    pool: GpuPool,
    log_dir: Path,
    *,
    no_monitor: bool,
) -> dict[tuple[str, str], dict[str, float]]:
    """Run each workload `reps` times, return per-(kernel, config) median impl times.

    Treats `INTERFERED` results as discardable but does not requeue —
    median of the surviving reps is taken. If all reps for a workload
    fail/interfere, that workload is omitted from the result.
    """
    import statistics
    samples: dict[tuple[str, str], dict[str, list[float]]] = {}
    n_jobs = len(workloads) * reps

    with ThreadPoolExecutor(
        max_workers=min(pool.total_visible(), n_jobs),
        thread_name_prefix="restable",
    ) as ex:
        futs = []
        for w in workloads:
            for _ in range(reps):
                futs.append(ex.submit(run_one, w, pool, log_dir, no_monitor=no_monitor))
        for fut in as_completed(futs):
            rec = fut.result()
            if rec.get("status") != "ok":
                continue
            key = (rec["kernel"], rec.get("label") or rec.get("config"))
            for impl, ms in (rec.get("impls") or {}).items():
                if ms is None or ms <= 0:
                    continue
                samples.setdefault(key, {}).setdefault(impl, []).append(ms)

    medians: dict[tuple[str, str], dict[str, float]] = {}
    for key, by_impl in samples.items():
        medians[key] = {impl: statistics.median(vs) for impl, vs in by_impl.items() if vs}
    return medians


def main() -> None:
    ap = argparse.ArgumentParser(description="tir-bench: pre-commit regression benchmark")
    ap.add_argument("--workloads", type=Path, default=DEFAULT_WORKLOADS,
                    help="YAML file listing kernels/configs to bench")
    ap.add_argument("--out-dir", type=Path, default=Path(".tir-bench"),
                    help="Where to store runs/, logs/, reports/, latest.json")
    ap.add_argument("--baseline", type=Path, default=None,
                    help=f"Baseline run JSON to diff against (default: {DEFAULT_BASELINE})")
    ap.add_argument("--threshold", type=float, default=DEFAULT_REGRESSION_THRESHOLD,
                    help="Regression threshold in percent slowdown "
                         f"(default {DEFAULT_REGRESSION_THRESHOLD:g})")
    ap.add_argument("--filter", type=str, default=None,
                    help="Only keep workloads whose kernel contains this substring")
    # NOTE: there is intentionally no --gpus flag. GPU selection is automatic
    # (util-gated probe + per-acquire utilization scan); a human pinning cards
    # defeats that and can land work on a busy card. See acquire()/_occupied_indices.
    ap.add_argument("--label", type=str, default=None,
                    help="Free-form label for this run (default: git short sha)")
    ap.add_argument("--no-report", action="store_true",
                    help="Skip regression report generation")
    ap.add_argument("--no-probe", action="store_true",
                    help="Skip the per-GPU probe (use nvidia-smi free-status only)")
    ap.add_argument("--probe-timeout", type=float, default=60.0,
                    help="Per-GPU probe timeout in seconds (default 60)")
    ap.add_argument("--no-monitor", action="store_true",
                    help="Don't monitor for GPU interference during workloads")
    ap.add_argument("--util-threshold", type=float, default=DEFAULT_UTIL_THRESHOLD,
                    help="%% GPU/sm utilization at/above which a card counts as "
                         "actively in use: selection skips such cards and the "
                         "monitor requeues if a neighbor crosses it mid-run. "
                         "Cards merely holding resident VRAM at lower util are "
                         f"shared (default {DEFAULT_UTIL_THRESHOLD:g})")
    ap.add_argument("--restable-threshold", type=float, default=DEFAULT_REGRESSION_THRESHOLD,
                    help="After the main sweep, re-bench any workload whose "
                         "|ratio Δ vs baseline| exceeds this %% "
                         f"(default {DEFAULT_REGRESSION_THRESHOLD:g}). "
                         "Set --restable-reps=0 to disable.")
    ap.add_argument("--restable-reps", type=int, default=5,
                    help="How many additional reps to bench each drifted "
                         "workload during the restable phase (default 5). "
                         "Set 0 to skip the phase entirely.")
    args = ap.parse_args()

    workloads = load_workloads(args.workloads)
    if args.filter:
        workloads = [w for w in workloads if args.filter in w["kernel"]]
    if not workloads:
        print("[tir-bench] no workloads to run.", file=sys.stderr)
        sys.exit(2)

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)

    # Live per-run log: tail -f this to watch progress. Use the same
    # timestamp as the run JSON so the pair is obvious in the filesystem.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_log_path = runs_dir / f"{stamp}.log"
    run_log_fh = open(run_log_path, "a", buffering=1)
    sys.stdout = _Tee(sys.stdout, run_log_fh)
    sys.stderr = _Tee(sys.stderr, run_log_fh)
    # Repoint `latest.log` symlink immediately so `tail -f .tir-bench/latest.log`
    # picks up this run before any output happens.
    latest_log = out_dir / "latest.log"
    if latest_log.exists() or latest_log.is_symlink():
        latest_log.unlink()
    latest_log.symlink_to(run_log_path.relative_to(out_dir))

    print(f"[tir-bench] live log: {run_log_path}")
    print(f"[tir-bench]   tail : tail -f {latest_log}")
    print(f"[tir-bench] run id : {stamp}")

    # ── Automatic GPU selection (no manual override on purpose) ──
    # 1. Startup probe: run a tiny fp16 matmul on every visible card
    #    (including busy ones — the probe is light, finishes fine on a
    #    contended card; this catches broken drivers / ECC). Probe failures
    #    are banned for the rest of the run.
    # 2. Per-workload acquire: re-scan utilization every time we need a card
    #    and pick any probe-OK one whose util is below --util-threshold. A
    #    card pegged at sweep start is reusable the moment its util drops; a
    #    card merely holding resident VRAM at low util is shared right away.
    listing_pool = GpuPool(util_threshold=args.util_threshold)
    in_filter = [idx for idx, _ in listing_pool._all_gpus()]
    if not in_filter:
        print("[tir-bench] no visible GPUs.", file=sys.stderr)
        sys.exit(1)
    utils_now = listing_pool._utils()
    occupied_now = sorted(listing_pool._occupied_indices() & set(in_filter), key=int)
    resident = sorted(listing_pool._busy_indices() & set(in_filter), key=int)
    util_str = " ".join(f"{i}:{utils_now.get(i, 0):.0f}%" for i in sorted(in_filter, key=int))
    print(
        f"[tir-bench] visible: {len(in_filter)} {sorted(in_filter, key=int)}; "
        f"util now [{util_str}]",
        flush=True,
    )
    print(
        f"[tir-bench] gate: util-threshold={args.util_threshold:g}% — "
        f"occupied (skip): {occupied_now if occupied_now else 'none'}; "
        f"shareable incl. idle-but-resident: "
        f"{sorted((set(in_filter) - set(occupied_now)), key=int)} "
        f"(resident-VRAM cards: {resident if resident else 'none'})",
        flush=True,
    )

    if args.no_probe:
        usable = set(in_filter)
        probe_failures: dict[str, str] = {}
    else:
        print(f"[tir-bench] probing {len(in_filter)} GPU(s) with fp16 512x512 matmul ...", flush=True)
        usable, probe_failures = detect_usable_gpus(in_filter, args.probe_timeout)

    if not usable:
        print("[tir-bench] no usable GPUs (all probes failed).", file=sys.stderr)
        for idx, err in probe_failures.items():
            print(f"[tir-bench]   gpu {idx}: {err}", file=sys.stderr)
        sys.exit(1)

    pool = GpuPool(allowed=usable, util_threshold=args.util_threshold)
    n_gpus = len(usable)

    _repo_git = collect_repo_git()
    label = args.label or _repo_git.get("tirx-kernels") or _repo_git.get("tir") or "local"
    print(
        f"[tir-bench] {len(workloads)} workloads, {n_gpus} probe-OK GPU(s) in pool, label={label}",
        flush=True,
    )

    # Results keyed by (kernel, config) so a requeued workload overwrites
    # its previous attempt — the regression report sees only the final run.
    results_by_key: dict[tuple[str, str], dict] = {}
    requeue_log: list[tuple[str, str, int, list[int]]] = []  # (kernel, config, attempt, intruder_pids)

    with ThreadPoolExecutor(max_workers=n_gpus, thread_name_prefix="bench") as ex:
        in_flight: dict = {}  # future -> (workload, attempt_no)
        for w in workloads:
            fut = ex.submit(run_one, w, pool, log_dir, no_monitor=args.no_monitor)
            in_flight[fut] = (w, 1)

        while in_flight:
            done, _ = wait(list(in_flight), return_when=FIRST_COMPLETED)
            for fut in done:
                workload, attempt = in_flight.pop(fut)
                record = fut.result()
                record["attempt"] = attempt
                key = (workload["kernel"], workload["config"])

                if (record.get("status") == "INTERFERED"
                        and attempt < MAX_INTERFERED_RETRIES):
                    # Requeue: any free worker will pick it up via acquire().
                    intruders = record.get("intruder_pids") or []
                    requeue_log.append((workload["kernel"], workload["config"], attempt, intruders))
                    log(f"[tir-bench] >>> REQUEUE {workload['kernel']}/{workload['config']} — "
                        f"attempt {attempt}/{MAX_INTERFERED_RETRIES} hit interference (intruders {intruders}), retrying <<<")
                    # Hold the INTERFERED record so the final result reflects the
                    # latest attempt; it'll get overwritten when the retry lands.
                    results_by_key[key] = record
                    new_fut = ex.submit(
                        run_one, workload, pool, log_dir, no_monitor=args.no_monitor
                    )
                    in_flight[new_fut] = (workload, attempt + 1)
                else:
                    results_by_key[key] = record
                    if record.get("status") == "INTERFERED":
                        # Out of retries — promote to FAIL so the report flags it.
                        record["status"] = "FAIL"
                        record["error"] = (
                            f"INTERFERED on all {attempt} attempts "
                            f"(last intruders: {record.get('intruder_pids')})"
                        )
                        log(f"[tir-bench] !!! GIVE UP {workload['kernel']}/{workload['config']} — "
                            f"all {attempt} attempts interfered, marking FAIL !!!")

    # End-of-run interference summary.
    if requeue_log:
        log(f"[tir-bench] interference summary: {len(requeue_log)} retry event(s)")
        for k, c, att, intr in requeue_log:
            log(f"[tir-bench]   - {k}/{c}: attempt {att} → intruders {intr}")
    else:
        log("[tir-bench] interference summary: none")

    results = list(results_by_key.values())
    results.sort(key=lambda r: (r["kernel"], r.get("label") or r.get("config")))
    probe_meta = {
        "enabled": not args.no_probe,
        "usable": sorted(usable),
        "failed": probe_failures,
    }
    run_path = write_run(out_dir, stamp, results, label, probe=probe_meta)
    current = json.loads(run_path.read_text())

    latest = out_dir / "latest.json"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(run_path.relative_to(out_dir))

    summary_path = write_summary(out_dir, current)
    print(f"[tir-bench] wrote {run_path}")
    print(f"[tir-bench] wrote {summary_path}")

    if args.no_report:
        return

    # Baseline = explicit --baseline if given, else the pinned baseline.json
    # next to the tool. To promote a fresh run, the user `cp` the new run
    # JSON over baseline.json themselves.
    baseline = args.baseline if args.baseline is not None else DEFAULT_BASELINE
    if not baseline.exists():
        print(f"[tir-bench] no baseline at {baseline} — skipping regression report")
        print(f"[tir-bench]   to set one: cp {run_path} {DEFAULT_BASELINE}")
        return

    report_md, n_regress = diff_report(baseline, current, args.threshold)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(exist_ok=True)
    report_path = reports_dir / f"{current['timestamp']}.md"
    report_path.write_text(report_md)
    print(f"[tir-bench] wrote {report_path}\n")
    print(report_md)

    # Ratio-based diff: normalises away GPU-contention noise by comparing
    # ours/ref ratio across runs (ref = fastest non-ours impl in baseline).
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        from ratio_diff import build_report as _build_ratio_report  # noqa: E402
        ratio_md, _ = _build_ratio_report(baseline, current, threshold_pct=args.threshold)
        ratio_path = reports_dir / f"{current['timestamp']}-ratio.md"
        ratio_path.write_text(ratio_md)
        print(f"[tir-bench] wrote {ratio_path}\n")
    except Exception as e:
        print(f"[tir-bench] ratio diff failed: {e}", file=sys.stderr)
        if n_regress > 0:
            sys.exit(3)
        return

    # ── Auto-restable phase ─────────────────────────────────────────────
    # Pick workloads whose ratio Δ from baseline exceeds args.restable_threshold,
    # re-run each N times, replace the result with the per-impl median, and
    # emit a stabilized ratio report. This catches outlier baseline rows
    # (one bad GPU exposure recorded in baseline.json) and outlier current
    # rows alike — the median of N reps is much less affected by either.
    if args.restable_reps <= 0:
        if n_regress > 0:
            sys.exit(3)
        return

    drifted_keys = _drifted_workloads(baseline, current, args.restable_threshold)
    if not drifted_keys:
        print(f"[tir-bench] no workloads drifted > ±{args.restable_threshold:.1f}%; "
              "skipping restable phase")
        if n_regress > 0:
            sys.exit(3)
        return

    workloads_by_key = {(w["kernel"], w["config"]): w for w in workloads}
    retest_specs = [workloads_by_key[k] for k in drifted_keys if k in workloads_by_key]
    if not retest_specs:
        print("[tir-bench] drifted keys not in original workloads list; skipping restable")
        if n_regress > 0:
            sys.exit(3)
        return

    print(
        f"[tir-bench] restabilizing {len(retest_specs)} workload(s) over "
        f"{args.restable_reps} reps each (|ratio Δ| > {args.restable_threshold:.1f}%) ..."
    )
    medians = _bench_median(
        retest_specs, args.restable_reps, pool, log_dir,
        no_monitor=args.no_monitor,
    )

    # Patch current results in-place with the stabilized per-impl medians.
    n_patched = 0
    for r in current["results"]:
        key = (r["kernel"], r.get("label") or r.get("config"))
        if key in medians:
            old_impls = dict(r.get("impls") or {})
            r["impls"] = medians[key]
            r["restabilized"] = {
                "reps": args.restable_reps,
                "old_impls": old_impls,
            }
            n_patched += 1
    print(f"[tir-bench] patched {n_patched} restabilized workload(s) into current run")

    stable_path = runs_dir / f"{current['timestamp']}-stable.json"
    stable_path.write_text(json.dumps(current, indent=2))
    print(f"[tir-bench] wrote {stable_path}")

    try:
        ratio_md, n_regress = _build_ratio_report(baseline, current, threshold_pct=args.threshold)
        ratio_path = reports_dir / f"{current['timestamp']}-stable-ratio.md"
        ratio_path.write_text(ratio_md)
        print(f"[tir-bench] wrote {ratio_path}\n")
    except Exception as e:
        print(f"[tir-bench] stable ratio diff failed: {e}", file=sys.stderr)

    if n_regress > 0:
        sys.exit(3)


if __name__ == "__main__":
    main()
