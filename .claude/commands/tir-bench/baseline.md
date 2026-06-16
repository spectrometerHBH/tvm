# tir-bench baseline view: `tir.json + ref.json`

- Timestamp: `18`
- Label:     `cu132-full`
- Git:       `{'tir': '9215eddf-dirty', 'tirx-kernels': 'e2582505', 'tirx-bench-ci': '08a57cf3'}`
- Workloads: 254 ok, 0 failed

Each row shows our impl's time (tir/tirx) and every reference impl, with ref/ours where ref = fastest non-ours impl. Higher ratio = ours is faster.

## deepgemm_sm100_fp4_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0389 | deepgemm | 0.0411 | 1.056 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0505 | deepgemm | 0.0536 | 1.060 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0399 | deepgemm | 0.0409 | 1.024 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0512 | deepgemm | 0.0526 | 1.029 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0400 | deepgemm | 0.0427 | 1.066 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0513 | deepgemm | 0.0553 | 1.078 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0391 | deepgemm | 0.0384 | 0.984 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0503 | deepgemm | 0.0494 | 0.983 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.0680 | deepgemm | 0.0715 | 1.053 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1039 | deepgemm | 0.1107 | 1.065 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.0684 | deepgemm | 0.0697 | 1.019 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1051 | deepgemm | 0.1064 | 1.012 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.0687 | deepgemm | 0.0748 | 1.088 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1059 | deepgemm | 0.1160 | 1.096 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 0.0674 | deepgemm | 0.0661 | 0.980 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1037 | deepgemm | 0.1022 | 0.985 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0682 | deepgemm | 0.0734 | 1.075 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0683 | deepgemm | 0.0733 | 1.073 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0690 | deepgemm | 0.0720 | 1.042 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0690 | deepgemm | 0.0716 | 1.038 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0688 | deepgemm | 0.0753 | 1.094 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0693 | deepgemm | 0.0758 | 1.094 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0675 | deepgemm | 0.0676 | 1.002 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0674 | deepgemm | 0.0676 | 1.003 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.1221 | deepgemm | 0.1302 | 1.067 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1756 | deepgemm | 0.1877 | 1.069 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.1229 | deepgemm | 0.1262 | 1.026 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1779 | deepgemm | 0.1815 | 1.020 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.1229 | deepgemm | 0.1357 | 1.104 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1779 | deepgemm | 0.1968 | 1.106 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 0.1209 | deepgemm | 0.1202 | 0.994 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1734 | deepgemm | 0.1730 | 0.998 | — |
## deepgemm_sm100_fp4_paged_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.992 | — |
| `b16_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 0.0043 | deepgemm | 0.0042 | 0.985 | — |
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0045 | deepgemm | 0.0045 | 0.996 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0045 | deepgemm | 0.0045 | 1.003 | — |
| `b16_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0039 | 1.044 | — |
| `b16_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0040 | 1.046 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0035 | deepgemm | 0.0036 | 1.042 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0034 | deepgemm | 0.0036 | 1.053 | — |
| `b16_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.994 | — |
| `b16_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.995 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0038 | 0.996 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0038 | 0.998 | — |
| `b16_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.989 | — |
| `b16_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.994 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.990 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.998 | — |
| `b1_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.995 | — |
| `b1_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.995 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.991 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.997 | — |
| `b1_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 0.0035 | deepgemm | 0.0037 | 1.052 | — |
| `b1_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 0.0035 | deepgemm | 0.0036 | 1.052 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0035 | deepgemm | 0.0037 | 1.053 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0033 | deepgemm | 0.0036 | 1.067 | — |
| `b1_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.995 | — |
| `b1_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.995 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.993 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.999 | — |
| `b1_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.990 | — |
| `b1_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.996 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.992 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.995 | — |
| `b2_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.992 | — |
| `b2_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.992 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.994 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.998 | — |
| `b2_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 0.0034 | deepgemm | 0.0036 | 1.062 | — |
| `b2_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0039 | 1.048 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0035 | deepgemm | 0.0037 | 1.047 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0034 | deepgemm | 0.0036 | 1.051 | — |
| `b2_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.995 | — |
| `b2_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.995 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.992 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.996 | — |
| `b2_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.997 | — |
| `b2_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.995 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.994 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.996 | — |
| `b4_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.996 | — |
| `b4_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.996 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0038 | 0.993 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0038 | 0.997 | — |
| `b4_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 0.0034 | deepgemm | 0.0036 | 1.051 | — |
| `b4_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0039 | 1.046 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0034 | deepgemm | 0.0035 | 1.050 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0038 | deepgemm | 0.0039 | 1.048 | — |
| `b4_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.995 | — |
| `b4_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.996 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.989 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.999 | — |
| `b4_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.992 | — |
| `b4_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.994 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.990 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.998 | — |
| `b8_n1_mp128_ps32_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.996 | — |
| `b8_n1_mp128_ps32_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.995 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.995 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.996 | — |
| `b8_n1_mp1_ps32_h64_d128_bf16_fixed` | tirx | 0.0035 | deepgemm | 0.0036 | 1.047 | — |
| `b8_n1_mp1_ps32_h64_d128_f32_fixed` | tirx | 0.0035 | deepgemm | 0.0037 | 1.062 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0034 | deepgemm | 0.0036 | 1.058 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0034 | deepgemm | 0.0036 | 1.058 | — |
| `b8_n1_mp32_ps32_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.994 | — |
| `b8_n1_mp32_ps32_h64_d128_f32_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.996 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0035 | 0.993 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.997 | — |
| `b8_n1_mp8_ps32_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.992 | — |
| `b8_n1_mp8_ps32_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.993 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.992 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.991 | — |
## deepgemm_sm100_fp8_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s2048_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0419 | deepgemm | 0.0434 | 1.036 | — |
| `s2048_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0555 | deepgemm | 0.0577 | 1.041 | — |
| `s2048_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0412 | deepgemm | 0.0408 | 0.992 | — |
| `s2048_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0549 | deepgemm | 0.0535 | 0.974 | — |
| `s2048_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0410 | deepgemm | 0.0435 | 1.061 | — |
| `s2048_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0534 | deepgemm | 0.0564 | 1.055 | — |
| `s2048_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0407 | deepgemm | 0.0404 | 0.992 | — |
| `s2048_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0543 | deepgemm | 0.0535 | 0.985 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.0734 | deepgemm | 0.0763 | 1.039 | — |
| `s2048_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1103 | deepgemm | 0.1165 | 1.056 | — |
| `s2048_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.0702 | deepgemm | 0.0702 | 1.000 | — |
| `s2048_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1082 | deepgemm | 0.1081 | 0.999 | — |
| `s2048_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.0716 | deepgemm | 0.0759 | 1.060 | — |
| `s2048_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1093 | deepgemm | 0.1166 | 1.066 | — |
| `s2048_skv8192_h64_d128_f32_dense_cp` | tirx | 0.0713 | deepgemm | 0.0709 | 0.995 | — |
| `s2048_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1113 | deepgemm | 0.1104 | 0.992 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_cp` | tirx | 0.0750 | deepgemm | 0.0778 | 1.037 | — |
| `s4096_skv4096_h64_d128_bf16_compressed_nocp` | tirx | 0.0716 | deepgemm | 0.0775 | 1.082 | — |
| `s4096_skv4096_h64_d128_bf16_dense_cp` | tirx | 0.0706 | deepgemm | 0.0711 | 1.008 | — |
| `s4096_skv4096_h64_d128_bf16_dense_nocp` | tirx | 0.0712 | deepgemm | 0.0718 | 1.008 | — |
| `s4096_skv4096_h64_d128_f32_compressed_cp` | tirx | 0.0716 | deepgemm | 0.0776 | 1.084 | — |
| `s4096_skv4096_h64_d128_f32_compressed_nocp` | tirx | 0.0711 | deepgemm | 0.0770 | 1.083 | — |
| `s4096_skv4096_h64_d128_f32_dense_cp` | tirx | 0.0699 | deepgemm | 0.0703 | 1.006 | — |
| `s4096_skv4096_h64_d128_f32_dense_nocp` | tirx | 0.0715 | deepgemm | 0.0705 | 0.986 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_cp` | tirx | 0.1317 | deepgemm | 0.1382 | 1.049 | — |
| `s4096_skv8192_h64_d128_bf16_compressed_nocp` | tirx | 0.1858 | deepgemm | 0.1979 | 1.065 | — |
| `s4096_skv8192_h64_d128_bf16_dense_cp` | tirx | 0.1280 | deepgemm | 0.1279 | 0.999 | — |
| `s4096_skv8192_h64_d128_bf16_dense_nocp` | tirx | 0.1849 | deepgemm | 0.1911 | 1.033 | — |
| `s4096_skv8192_h64_d128_f32_compressed_cp` | tirx | 0.1301 | deepgemm | 0.1387 | 1.066 | — |
| `s4096_skv8192_h64_d128_f32_compressed_nocp` | tirx | 0.1842 | deepgemm | 0.1980 | 1.075 | — |
| `s4096_skv8192_h64_d128_f32_dense_cp` | tirx | 0.1278 | deepgemm | 0.1284 | 1.004 | — |
| `s4096_skv8192_h64_d128_f32_dense_nocp` | tirx | 0.1861 | deepgemm | 0.1877 | 1.009 | — |
## deepgemm_sm100_fp8_paged_mqa_logits

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `b16_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0045 | deepgemm | 0.0044 | 0.991 | — |
| `b16_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0045 | deepgemm | 0.0045 | 0.999 | — |
| `b16_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.994 | — |
| `b16_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.993 | — |
| `b16_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.990 | — |
| `b16_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.991 | — |
| `b16_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.992 | — |
| `b16_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.994 | — |
| `b1_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.989 | — |
| `b1_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.991 | — |
| `b1_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.001 | — |
| `b1_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.995 | — |
| `b1_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.989 | — |
| `b1_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.989 | — |
| `b1_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 1.004 | — |
| `b1_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.999 | — |
| `b2_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.991 | — |
| `b2_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.999 | — |
| `b2_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0038 | deepgemm | 0.0038 | 0.996 | — |
| `b2_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.995 | — |
| `b2_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.989 | — |
| `b2_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 1.001 | — |
| `b2_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.991 | — |
| `b2_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0036 | 0.999 | — |
| `b4_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.997 | — |
| `b4_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.997 | — |
| `b4_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 0.989 | — |
| `b4_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.995 | — |
| `b4_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.992 | — |
| `b4_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.998 | — |
| `b4_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.996 | — |
| `b4_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.997 | — |
| `b8_n1_mp128_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0039 | 0.992 | — |
| `b8_n1_mp128_ps64_h64_d128_f32_fixed` | tirx | 0.0043 | deepgemm | 0.0043 | 1.004 | — |
| `b8_n1_mp1_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.993 | — |
| `b8_n1_mp1_ps64_h64_d128_f32_fixed` | tirx | 0.0036 | deepgemm | 0.0036 | 0.997 | — |
| `b8_n1_mp32_ps64_h64_d128_bf16_fixed` | tirx | 0.0036 | deepgemm | 0.0037 | 1.003 | — |
| `b8_n1_mp32_ps64_h64_d128_f32_fixed` | tirx | 0.0039 | deepgemm | 0.0039 | 0.996 | — |
| `b8_n1_mp8_ps64_h64_d128_bf16_fixed` | tirx | 0.0040 | deepgemm | 0.0040 | 0.996 | — |
| `b8_n1_mp8_ps64_h64_d128_f32_fixed` | tirx | 0.0037 | deepgemm | 0.0037 | 1.000 | — |
## flash_attention4

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `s1024_h32kv16` | tir | 0.0200 | flashattn_sm100 | 0.0201 | 1.004 | flashinfer=0.0255 |
| `s1024_h32kv16_causal` | tir | 0.0195 | flashattn_sm100 | 0.0194 | 0.995 | flashinfer=0.0249 |
| `s1024_h32kv32` | tir | 0.0202 | flashattn_sm100 | 0.0203 | 1.001 | flashinfer=0.0256 |
| `s1024_h32kv32_causal` | tir | 0.0201 | flashattn_sm100 | 0.0200 | 0.991 | flashinfer=0.0249 |
| `s1024_h32kv4` | tir | 0.0193 | flashattn_sm100 | 0.0196 | 1.016 | flashinfer=0.0248 |
| `s1024_h32kv4_causal` | tir | 0.0182 | flashattn_sm100 | 0.0193 | 1.061 | flashinfer=0.0249 |
| `s1024_h32kv8` | tir | 0.0196 | flashattn_sm100 | 0.0197 | 1.003 | flashinfer=0.0248 |
| `s1024_h32kv8_causal` | tir | 0.0187 | flashattn_sm100 | 0.0190 | 1.019 | flashinfer=0.0250 |
| `s2048_h32kv16` | tir | 0.0565 | flashattn_sm100 | 0.0571 | 1.011 | flashinfer=0.0732 |
| `s2048_h32kv16_causal` | tir | 0.0355 | flashattn_sm100 | 0.0374 | 1.055 | flashinfer=0.0734 |
| `s2048_h32kv32` | tir | 0.0575 | flashattn_sm100 | 0.0580 | 1.009 | flashinfer=0.1552 |
| `s2048_h32kv32_causal` | tir | 0.0389 | flashattn_sm100 | 0.0390 | 1.003 | flashinfer=0.0734 |
| `s2048_h32kv4` | tir | 0.0545 | flashattn_sm100 | 0.0551 | 1.011 | flashinfer=0.0732 |
| `s2048_h32kv4_causal` | tir | 0.0345 | flashattn_sm100 | 0.0364 | 1.055 | flashinfer=0.0728 |
| `s2048_h32kv8` | tir | 0.0556 | flashattn_sm100 | 0.0561 | 1.010 | flashinfer=0.0732 |
| `s2048_h32kv8_causal` | tir | 0.0352 | flashattn_sm100 | 0.0373 | 1.059 | flashinfer=0.0730 |
| `s4096_h32kv16` | tir | 0.2038 | flashattn_sm100 | 0.2058 | 1.010 | flashinfer=0.2615 |
| `s4096_h32kv16_causal` | tir | 0.1129 | flashattn_sm100 | 0.1161 | 1.029 | flashinfer=0.2651 |
| `s4096_h32kv32` | tir | 0.2129 | flashattn_sm100 | 0.2138 | 1.004 | flashinfer=0.2656 |
| `s4096_h32kv32_causal` | tir | 0.1193 | flashattn_sm100 | 0.1189 | 0.996 | flashinfer=0.2662 |
| `s4096_h32kv4` | tir | 0.2015 | flashattn_sm100 | 0.2036 | 1.010 | flashinfer=0.2631 |
| `s4096_h32kv4_causal` | tir | 0.1095 | flashattn_sm100 | 0.1127 | 1.030 | flashinfer=0.2608 |
| `s4096_h32kv8` | tir | 0.2005 | flashattn_sm100 | 0.2025 | 1.010 | flashinfer=0.2639 |
| `s4096_h32kv8_causal` | tir | 0.1090 | flashattn_sm100 | 0.1122 | 1.030 | flashinfer=0.2658 |
| `s8192_h32kv16` | tir | 0.8191 | flashattn_sm100 | 0.8293 | 1.012 | flashinfer=1.0203 |
| `s8192_h32kv16_causal` | tir | 0.4285 | flashattn_sm100 | 0.4451 | 1.039 | flashinfer=1.0122 |
| `s8192_h32kv32` | tir | 0.8417 | flashattn_sm100 | 0.8403 | 0.998 | flashinfer=1.0321 |
| `s8192_h32kv32_causal` | tir | 0.4480 | flashattn_sm100 | 0.4527 | 1.010 | flashinfer=1.0360 |
| `s8192_h32kv4` | tir | 0.8337 | flashattn_sm100 | 0.8377 | 1.005 | flashinfer=1.0155 |
| `s8192_h32kv4_causal` | tir | 0.4237 | flashattn_sm100 | 0.4382 | 1.034 | flashinfer=1.0190 |
| `s8192_h32kv8` | tir | 0.8077 | flashattn_sm100 | 0.8117 | 1.005 | flashinfer=1.0066 |
| `s8192_h32kv8_causal` | tir | 0.4298 | flashattn_sm100 | 0.4284 | 0.997 | flashinfer=1.0343 |
## fp16_bf16_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bf16_1024x1024x1024` | tir | 0.0063 | torch-cublas | 0.0053 | 0.845 | deepgemm-bf16=0.0072, deepgemm-cublaslt=0.0054 |
| `bf16_16384x16384x16384` | tir | 6.2843 | torch-cublas | 6.1744 | 0.983 | deepgemm-bf16=7.2544, deepgemm-cublaslt=6.1915 |
| `bf16_2048x2048x2048` | tir | 0.0158 | torch-cublas | 0.0154 | 0.969 | deepgemm-bf16=0.0180, deepgemm-cublaslt=0.0156 |
| `bf16_4096x4096x4096` | tir | 0.0925 | deepgemm-bf16 | 0.0923 | 0.998 | deepgemm-cublaslt=0.0938, torch-cublas=0.0933 |
| `bf16_8192x8192x8192` | tir | 0.7789 | deepgemm-bf16 | 0.8096 | 1.039 | deepgemm-cublaslt=0.8210, torch-cublas=0.8306 |
| `fp16_1024x1024x1024` | tir | 0.0062 | torch-cublas | 0.0053 | 0.860 | deepgemm-cublaslt=0.0054 |
| `fp16_16384x16384x16384` | tir | 6.5401 | torch-cublas | 6.4054 | 0.979 | deepgemm-cublaslt=6.4326 |
| `fp16_2048x2048x2048` | tir | 0.0161 | torch-cublas | 0.0158 | 0.980 | deepgemm-cublaslt=0.0158 |
| `fp16_4096x4096x4096` | tir | 0.0961 | deepgemm-cublaslt | 0.0977 | 1.016 | torch-cublas=0.0979 |
| `fp16_8192x8192x8192` | tir | 0.8313 | deepgemm-cublaslt | 0.8553 | 1.029 | torch-cublas=0.8852 |
## fp8_blockwise_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `deepgemm_m4096_n2112_k7168` | tir | 0.0491 | deepgemm | 0.0496 | 1.011 | — |
| `deepgemm_m4096_n24576_k1536` | tir | 0.1159 | deepgemm | 0.1166 | 1.006 | — |
| `deepgemm_m4096_n32768_k512` | tir | 0.0725 | deepgemm | 0.0764 | 1.053 | — |
| `deepgemm_m4096_n4096_k7168` | tir | 0.0841 | deepgemm | 0.0846 | 1.006 | — |
| `deepgemm_m4096_n576_k7168` | tir | 0.0180 | deepgemm | 0.0185 | 1.029 | — |
| `deepgemm_m4096_n7168_k16384` | tir | 0.3227 | deepgemm | 0.3311 | 1.026 | — |
| `deepgemm_m4096_n7168_k2048` | tir | 0.0441 | deepgemm | 0.0447 | 1.014 | — |
| `smoke_1024x1024x1024` | tir | 0.0057 | deepgemm | 0.0062 | 1.093 | — |
| `stress_m8192_n7168_k4096` | tir | 0.1608 | deepgemm | 0.1611 | 1.002 | — |
## nvfp4_gemm

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `1024x1024x1024` | tir | 0.0051 | cublaslt_nvfp4 | 0.0042 | 0.832 | flashinfer=0.0043 |
| `16384x16384x16384` | tir | 1.6704 | flashinfer | 1.6149 | 0.967 | cublaslt_nvfp4=1.6281 |
| `2048x2048x2048` | tir | 0.0082 | cublaslt_nvfp4 | 0.0075 | 0.913 | flashinfer=0.0076 |
| `4096x4096x4096` | tir | 0.0287 | flashinfer | 0.0291 | 1.016 | cublaslt_nvfp4=0.0299 |
| `8192x8192x8192` | tir | 0.1835 | flashinfer | 0.1740 | 0.948 | cublaslt_nvfp4=0.1832 |
## sparse_flashmla_prefill_head128_phase1

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_regular_dqk512_hq128_s4096_kv32768_topk2048` | tirx | 1.8099 | — | nan | — | — |
| `bench_regular_dqk512_hq128_s4096_kv65536_topk2048` | tirx | 1.9908 | — | nan | — | — |
| `bench_regular_dqk512_hq128_s4096_kv8192_topk2048` | tirx | 1.7790 | — | nan | — | — |
| `bench_regular_dqk576_hq128_s4096_kv32768_topk2048` | tirx | 1.9050 | — | nan | — | — |
| `bench_regular_dqk576_hq128_s4096_kv65536_topk2048` | tirx | 2.0986 | — | nan | — | — |
| `bench_regular_dqk576_hq128_s4096_kv8192_topk2048` | tirx | 1.8995 | — | nan | — | — |
## sparse_flashmla_prefill_head64_phase1

| config | ours impl | ours (ms) | ref impl | ref (ms) | ref/ours | other impls |
|---|---|---:|---|---:|---:|---|
| `bench_dqk512_hq64_s4096_kv32768_topk512` | tirx | 0.3758 | — | nan | — | — |
| `bench_dqk512_hq64_s4096_kv49152_topk512` | tirx | 0.3759 | — | nan | — | — |
| `bench_dqk512_hq64_s4096_kv65536_topk512` | tirx | 0.3850 | — | nan | — | — |
| `bench_dqk512_hq64_s4096_kv8192_topk512` | tirx | 0.3630 | — | nan | — | — |
| `bench_dqk576_hq64_s4096_kv32768_topk512` | tirx | 0.3929 | — | nan | — | — |
| `bench_dqk576_hq64_s4096_kv49152_topk512` | tirx | 0.3938 | — | nan | — | — |
| `bench_dqk576_hq64_s4096_kv65536_topk512` | tirx | 0.4007 | — | nan | — | — |
| `bench_dqk576_hq64_s4096_kv8192_topk512` | tirx | 0.3768 | — | nan | — | — |
