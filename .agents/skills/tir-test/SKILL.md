Run the full TIRX test suite.

## Steps

1. Install the kernel package, select the least busy GPU, and enable strict kernel-import checking:
   ```bash
   pip install -e /path/to/tirx-kernels-staging   # or sibling tirx-kernels clone
   export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
   export TVM_PATH=/path/to/tvm
   export PYTHONPATH="${TVM_PATH}/python"
   export TVM_LIBRARY_PATH="${TVM_PATH}/build/lib"
   # MUST stay set for the whole run. Without it, the kernel registry silently
   # drops any module that fails to import (warning + continue), so a framework
   # change that breaks a kernel's imports passes the tests unnoticed. With it,
   # discovery raises on the first failed import (see the import gate below).
   export TIRX_KERNELS_STRICT=1
   ```

2. Start the GPU monitor in the background so we can detect if anyone else lands on the same GPU mid-run:
   ```bash
   GPU_LOG="/tmp/tir_test_gpu_${CUDA_VISIBLE_DEVICES}.log"
   bash .agents/scripts/monitor_gpu.sh --gpu "$CUDA_VISIBLE_DEVICES" --interval 5 --log "$GPU_LOG" &
   MON_PID=$!
   trap 'kill $MON_PID 2>/dev/null' EXIT
   ```

3. Import gate — bench workloads: fail fast if any kernel listed in `workloads.yaml` fails to import:
   ```bash
   python -m tirx_kernels.tir_bench --check-imports
   ```
   A non-zero exit means a pinned workload kernel failed to import — fix it before proceeding.

4. Full kernel import gate (correctness test suite coverage):
   ```bash
   TIRX_KERNELS_STRICT=1 python -m tirx_kernels.registry --cc 10
   ```
   A non-zero exit means a kernel failed to import — this is a real failure (triage
   as category A/B below), fix it; never proceed or write it off as pre-existing.

5. Run the full test suite with xdist parallelism — keep `TIRX_KERNELS_STRICT=1`
   inlined so a kernel import error during collection is a hard failure here too:
   ```bash
   TIRX_KERNELS_STRICT=1 pytest tests/python/tirx/ -n 16
   ```

6. Stop the monitor and check for foreign GPU usage during the run:
   ```bash
   kill $MON_PID 2>/dev/null; wait $MON_PID 2>/dev/null
   grep -E 'FOREIGN USER|\[FOREIGN\]' "$GPU_LOG" || echo "no foreign GPU usage observed"
   ```

7. Report results: total passed, failed, skipped, errors — and the import-gate results from steps 3–4. If any foreign-user events are present in step 6, mention them — flaky failures should be re-evaluated on a clean GPU before being attributed to code changes.

## Failure triage rules

**CRITICAL: Never pipe test output to `tail` or `grep` when diagnosing failures. Always capture and read full logs.**

Classify every failure into one of these categories:

- **A — Environment/import error**: Module not found, missing dependency, collection error. These are not caused by code changes.
- **B — Real kernel correctness regression**: Assertion failures (cosine_sim, numerical diff), `CUDA: unspecified launch failure`, or wrong results. **These MUST be investigated and fixed if caused by current changes.**
- **C — Secondary xdist crash**: `KeyError: <WorkerController gwXX>` after a worker abort. The KeyError itself is noise — find the underlying cause (usually category B in another worker).

**Never dismiss a failure as "pre-existing" without evidence.** If a test fails:
1. Check whether the test touches code you changed.
2. If unclear, verify on the parent commit before claiming pre-existing.
3. All failures caused by current changes MUST be fixed — not deferred.
