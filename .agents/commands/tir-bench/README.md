# tir-bench (moved)

tir-bench now lives in **tirx-kernels** at `tirx_kernels/tir_bench/`.

```bash
cd /path/to/tirx-kernels-staging
export TVM_PATH=/path/to/tvm
export PYTHONPATH="$(pwd):${TVM_PATH}/python"
export TVM_LIBRARY_PATH="${TVM_PATH}/build/lib"
export TIRX_KERNELS_STRICT=1

python -m tirx_kernels.tir_bench --impls ours
```

See `tirx_kernels/tir_bench/README.md` in the kernel repo for baseline workflow and flags.
