Run the full TIRX test suite.

## Steps

1. Install the kernel package and select the least busy GPU:
   ```bash
   pip install -e /path/to/tirx-kernels-staging   # or sibling tirx-kernels checkout
   export CUDA_VISIBLE_DEVICES=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t',' -k2 -n | head -1 | cut -d',' -f1 | tr -d ' ')
   export TVM_PATH=/path/to/tvm
   export PYTHONPATH="${TVM_PATH}/python"
   export TVM_LIBRARY_PATH="${TVM_PATH}/build/lib"
   ```

2. Start the GPU monitor in the background so we can detect if anyone else lands on the same GPU mid-run:
   ```bash
   GPU_LOG="/tmp/tir_test_gpu_${CUDA_VISIBLE_DEVICES}.log"
   bash .agents/scripts/monitor_gpu.sh --gpu "$CUDA_VISIBLE_DEVICES" --interval 5 --log "$GPU_LOG" &
   MON_PID=$!
   trap 'kill $MON_PID 2>/dev/null' EXIT
   ```

3. Import gate — fail fast if any kernel no longer imports against the framework.
   `pytest tests/python/tirx/` exercises the framework, not the kernel registry, so
   a kernel that won't import is invisible to it. Force discovery of every kernel;
   `--strict` exits non-zero on the first failed import:
   ```bash
   python -m tirx_kernels.registry --cc 10 --strict
   ```
   A non-zero exit means a kernel failed to import — this is a real failure (triage
   as category A/B below), fix it; never proceed or write it off as pre-existing.

4. Run the full test suite with xdist parallelism:
   ```bash
   pytest tests/python/tirx/ -n 16
   ```

5. Stop the monitor and check for foreign GPU usage during the run:
   ```bash
   kill $MON_PID 2>/dev/null; wait $MON_PID 2>/dev/null
   grep -E 'FOREIGN USER|\[FOREIGN\]' "$GPU_LOG" || echo "no foreign GPU usage observed"
   ```

6. Report results: total passed, failed, skipped, errors — and the import-gate result from step 3. If any foreign-user events are present in step 5, mention them — flaky failures should be re-evaluated on a clean GPU before being attributed to code changes.

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
