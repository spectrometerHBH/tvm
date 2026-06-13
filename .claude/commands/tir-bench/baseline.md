# tir-bench baseline view: `baseline.json`

- Timestamp: `20260611T135029Z`
- Label:     `sparse-flashmla-head64-full-baseline-rebased+targeted-restable`
- Git:       `{'tir': 'bf634b79', 'tirx-kernels': '3aaf1663', 'tirx-bench-ci': '08a57cf3'}`
- Workloads: 168 ok, 0 failed

Each row shows our impl's time (tir/tirx) and every reference impl, with ref/ours where ref = fastest non-ours impl. Higher ratio = ours is faster.

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0404 | deepgemm | 0.0424 | 1.050 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0530 | deepgemm | 0.0555 | 1.047 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0412 | deepgemm | 0.0418 | 1.017 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0528 | deepgemm | 0.0542 | 1.025 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0411 | deepgemm | 0.0439 | 1.066 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0535 | deepgemm | 0.0575 | 1.074 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0402 | deepgemm | 0.0397 | 0.987 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0523 | deepgemm | 0.0516 | 0.987 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.0697 | deepgemm | 0.0734 | 1.053 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1087 | deepgemm | 0.1149 | 1.057 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.0711 | deepgemm | 0.0721 | 1.015 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1079 | deepgemm | 0.1095 | 1.015 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.0713 | deepgemm | 0.0769 | 1.079 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1085 | deepgemm | 0.1188 | 1.095 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 0.0696 | deepgemm | 0.0686 | 0.985 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1063 | deepgemm | 0.1050 | 0.988 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0715 | deepgemm | 0.0761 | 1.065 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0710 | deepgemm | 0.0756 | 1.065 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0726 | deepgemm | 0.0754 | 1.038 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0722 | deepgemm | 0.0749 | 1.037 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0710 | deepgemm | 0.0774 | 1.090 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0721 | deepgemm | 0.0786 | 1.090 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0694 | deepgemm | 0.0695 | 1.002 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0706 | deepgemm | 0.0707 | 1.002 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.1244 | deepgemm | 0.1322 | 1.063 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1783 | deepgemm | 0.1897 | 1.064 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.1259 | deepgemm | 0.1286 | 1.022 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1798 | deepgemm | 0.1831 | 1.018 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.1285 | deepgemm | 0.1412 | 1.098 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1800 | deepgemm | 0.1986 | 1.104 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 0.1258 | deepgemm | 0.1260 | 1.001 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1792 | deepgemm | 0.1790 | 0.999 | — |
## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0461 | deepgemm | 0.0470 | 1.020 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0604 | deepgemm | 0.0612 | 1.014 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0449 | deepgemm | 0.0446 | 0.994 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0582 | deepgemm | 0.0582 | 1.000 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0438 | deepgemm | 0.0457 | 1.043 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0594 | deepgemm | 0.0613 | 1.032 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0440 | deepgemm | 0.0436 | 0.991 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0575 | deepgemm | 0.0570 | 0.991 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.0794 | deepgemm | 0.0807 | 1.016 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1212 | deepgemm | 0.1236 | 1.020 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.0773 | deepgemm | 0.0776 | 1.003 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1189 | deepgemm | 0.1194 | 1.004 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.0790 | deepgemm | 0.0816 | 1.033 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1183 | deepgemm | 0.1235 | 1.044 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 0.0780 | deepgemm | 0.0776 | 0.995 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1197 | deepgemm | 0.1187 | 0.992 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.1273 | deepgemm | 0.1306 | 1.026 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0805 | deepgemm | 0.0822 | 1.021 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0770 | deepgemm | 0.0772 | 1.003 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0784 | deepgemm | 0.0782 | 0.998 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0768 | deepgemm | 0.0817 | 1.064 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0770 | deepgemm | 0.0823 | 1.069 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0765 | deepgemm | 0.0759 | 0.993 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0759 | deepgemm | 0.0757 | 0.997 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.1419 | deepgemm | 0.1447 | 1.020 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.2036 | deepgemm | 0.2090 | 1.027 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.1365 | deepgemm | 0.1368 | 1.002 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1946 | deepgemm | 0.1965 | 1.010 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.1358 | deepgemm | 0.1453 | 1.070 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1964 | deepgemm | 0.2089 | 1.064 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 0.1337 | deepgemm | 0.1347 | 1.008 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1921 | deepgemm | 0.1926 | 1.002 | — |
## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0046 | deepgemm | 0.0045 | 0.984 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0046 | deepgemm | 0.0045 | 0.990 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 1.000 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 1.005 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.997 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.995 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.994 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.993 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.999 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0041 | deepgemm | 0.0040 | 0.988 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.992 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.001 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.986 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0035 | 0.992 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.994 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 1.002 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.997 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0041 | deepgemm | 0.0040 | 0.992 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.998 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 1.000 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 1.000 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 1.000 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.989 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.999 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.995 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0041 | deepgemm | 0.0041 | 0.993 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.993 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 1.001 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.995 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.996 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.989 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.997 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.990 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0042 | deepgemm | 0.0042 | 0.999 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.991 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 1.002 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.997 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0038 | 0.993 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.998 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.996 | — |
## flash_attention4

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 0.0210 | flashattn_sm100 | 0.0203 | 0.969 | flashinfer=0.0261 |
| `s1024_h32kv16_causal` | tir | 0.0209 | flashattn_sm100 | 0.0200 | 0.957 | flashinfer=0.0261 |
| `s1024_h32kv32` | tir | 0.0212 | flashattn_sm100 | 0.0206 | 0.972 | flashinfer=0.0263 |
| `s1024_h32kv32_causal` | tir | 0.0226 | flashattn_sm100 | 0.0206 | 0.914 | flashinfer=0.0263 |
| `s1024_h32kv4` | tir | 0.0209 | flashattn_sm100 | 0.0201 | 0.963 | flashinfer=0.0260 |
| `s1024_h32kv4_causal` | tir | 0.0198 | flashattn_sm100 | 0.0195 | 0.987 | flashinfer=0.0259 |
| `s1024_h32kv8` | tir | 0.0208 | flashattn_sm100 | 0.0201 | 0.966 | flashinfer=0.0260 |
| `s1024_h32kv8_causal` | tir | 0.0201 | flashattn_sm100 | 0.0198 | 0.984 | flashinfer=0.0260 |
| `s2048_h32kv16` | tir | 0.0626 | flashattn_sm100 | 0.0605 | 0.967 | flashinfer=0.0774 |
| `s2048_h32kv16_causal` | tir | 0.0384 | flashattn_sm100 | 0.0392 | 1.020 | flashinfer=0.0777 |
| `s2048_h32kv32` | tir | 0.0625 | flashattn_sm100 | 0.0608 | 0.971 | flashinfer=0.0786 |
| `s2048_h32kv32_causal` | tir | 0.0430 | flashattn_sm100 | 0.0407 | 0.946 | flashinfer=0.0791 |
| `s2048_h32kv4` | tir | 0.0615 | flashattn_sm100 | 0.0599 | 0.973 | flashinfer=0.0775 |
| `s2048_h32kv4_causal` | tir | 0.0374 | flashattn_sm100 | 0.0384 | 1.028 | flashinfer=0.0767 |
| `s2048_h32kv8` | tir | 0.0607 | flashattn_sm100 | 0.0587 | 0.967 | flashinfer=0.0776 |
| `s2048_h32kv8_causal` | tir | 0.0380 | flashattn_sm100 | 0.0386 | 1.018 | flashinfer=0.0777 |
| `s4096_h32kv16` | tir | 0.2133 | flashattn_sm100 | 0.2086 | 0.978 | flashinfer=0.2708 |
| `s4096_h32kv16_causal` | tir | 0.1146 | flashattn_sm100 | 0.1163 | 1.015 | flashinfer=0.2686 |
| `s4096_h32kv32` | tir | 0.2193 | flashattn_sm100 | 0.2154 | 0.982 | flashinfer=0.2732 |
| `s4096_h32kv32_causal` | tir | 0.1240 | flashattn_sm100 | 0.1182 | 0.953 | flashinfer=0.2727 |
| `s4096_h32kv4` | tir | 0.2083 | flashattn_sm100 | 0.2036 | 0.977 | flashinfer=0.2672 |
| `s4096_h32kv4_causal` | tir | 0.1134 | flashattn_sm100 | 0.1137 | 1.003 | flashinfer=0.2678 |
| `s4096_h32kv8` | tir | 0.2117 | flashattn_sm100 | 0.2076 | 0.981 | flashinfer=0.2699 |
| `s4096_h32kv8_causal` | tir | 0.1152 | flashattn_sm100 | 0.1163 | 1.010 | flashinfer=0.2675 |
| `s8192_h32kv16` | tir | 0.7475 | flashattn_sm100 | 0.7354 | 0.984 | flashinfer=0.9061 |
| `s8192_h32kv16_causal` | tir | 0.4070 | flashattn_sm100 | 0.4058 | 0.997 | flashinfer=0.8992 |
| `s8192_h32kv32` | tir | 0.7518 | flashattn_sm100 | 0.7430 | 0.988 | flashinfer=0.9061 |
| `s8192_h32kv32_causal` | tir | 0.4270 | flashattn_sm100 | 0.4091 | 0.958 | flashinfer=0.9071 |
| `s8192_h32kv4` | tir | 0.7441 | flashattn_sm100 | 0.7213 | 0.969 | flashinfer=0.8956 |
| `s8192_h32kv4_causal` | tir | 0.4034 | flashattn_sm100 | 0.4026 | 0.998 | flashinfer=0.9040 |
| `s8192_h32kv8` | tir | 0.7394 | flashattn_sm100 | 0.7239 | 0.979 | flashinfer=0.8942 |
| `s8192_h32kv8_causal` | tir | 0.4082 | flashattn_sm100 | 0.4078 | 0.999 | flashinfer=0.9138 |
## fp16_bf16_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 0.0164 | deepgemm-cublaslt | 0.0055 | 0.331 | deepgemm-bf16=0.0075, torch-cublas=0.0055 |
| `bf16_16384x16384x16384` | tir | 5.8604 | torch-cublas | 5.6647 | 0.967 | deepgemm-bf16=6.6169, deepgemm-cublaslt=5.7109 |
| `bf16_2048x2048x2048` | tir | 0.0278 | deepgemm-cublaslt | 0.0184 | 0.663 | deepgemm-bf16=0.0188, torch-cublas=0.0185 |
| `bf16_4096x4096x4096` | tir | 0.0923 | deepgemm-bf16 | 0.0931 | 1.008 | deepgemm-cublaslt=0.0937, torch-cublas=0.0936 |
| `bf16_8192x8192x8192` | tir | 0.6647 | deepgemm-cublaslt | 0.6877 | 1.035 | deepgemm-bf16=0.6995, torch-cublas=0.6884 |
| `fp16_1024x1024x1024` | tir | 0.0166 | torch-cublas | 0.0055 | 0.332 | deepgemm-cublaslt=0.0056 |
| `fp16_16384x16384x16384` | tir | 6.4681 | deepgemm-cublaslt | 6.1511 | 0.951 | torch-cublas=6.2796 |
| `fp16_2048x2048x2048` | tir | 0.0274 | deepgemm-cublaslt | 0.0185 | 0.673 | torch-cublas=0.0185 |
| `fp16_4096x4096x4096` | tir | 0.0950 | torch-cublas | 0.0962 | 1.013 | deepgemm-cublaslt=0.0966 |
| `fp16_8192x8192x8192` | tir | 0.6961 | torch-cublas | 0.7216 | 1.037 | deepgemm-cublaslt=0.7220 |
## fp8_blockwise_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 0.0485 | deepgemm | 0.0496 | 1.024 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 0.1152 | deepgemm | 0.1167 | 1.013 | — |
| `deepgemm_m4096_n32768_k512` | tir | 0.0719 | deepgemm | 0.0754 | 1.049 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 0.0833 | deepgemm | 0.0838 | 1.005 | — |
| `deepgemm_m4096_n576_k7168` | tir | 0.0186 | deepgemm | 0.0194 | 1.040 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 0.3267 | deepgemm | 0.3286 | 1.006 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 0.0436 | deepgemm | 0.0445 | 1.020 | — |
| `smoke_1024x1024x1024` | tir | 0.0058 | deepgemm | 0.0067 | 1.139 | — |
| `stress_m8192_n7168_k4096` | tir | 0.1643 | deepgemm | 0.1657 | 1.008 | — |
## nvfp4_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 0.0066 | flashinfer | 0.0051 | 0.781 | — |
| `16384x16384x16384` | tir | 1.4647 | flashinfer | 1.6177 | 1.104 | — |
| `2048x2048x2048` | tir | 0.0088 | flashinfer | 0.0085 | 0.966 | — |
| `4096x4096x4096` | tir | 0.0303 | flashinfer | 0.0321 | 1.061 | — |
| `8192x8192x8192` | tir | 0.1879 | flashinfer | 0.1757 | 0.935 | — |
## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 0.3736 | — | nan | — | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 0.3737 | — | nan | — | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 0.3824 | — | nan | — | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 0.3727 | — | nan | — | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 0.3841 | — | nan | — | — |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 0.3867 | — | nan | — | — |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 0.4042 | — | nan | — | — |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 0.3765 | — | nan | — | — |
