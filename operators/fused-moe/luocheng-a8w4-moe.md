# Luocheng runtime A8W4 MoE on MI308X

## What

PyHIP `fused_moe_a8w4` packs MXFP4 expert weights at load time and decodes them to FP8 in registers for the gate/up and down GEMMs. M13 combines expert sorting, input quantization and location-ID emission on eligible prefill shapes. The reported operator time covers route preparation, quantization, both GEMMs and routed-output reduction. [M13 change](https://github.com/luocheng25/pyhip/commit/a641da286511f3f8b703f90b3f9a1ea66b18301a) · [Kernel design](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/kernels.md)

## Data

Each shape ran on one MI308X (`gfx942`) using the indicated TP-local weights. The call takes BF16 `[M,H]` activations, int32 route IDs and FP32 route weights, both `[M,top-k]`, and returns BF16 `[M,H]`. It uses SiLU, per-row FP8 E4M3FNUZ quantization with scale `amax / 240`, and MXFP4 E2M1 weights with one E8M0 scale per 32 K elements. Qwen routes 10 of 512 experts plus fixed shared slot ID 512. MiMo uses uniform routing with 8 of 256 experts. [Benchmark shapes and protocol](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/tests/ops/moe/test_moe_a8w4.py#L1450) · [Operator contract](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_a8w4.py#L605)

| Model | TP shape | Hidden H | Global I | Local I | Padded local I | Routed experts | Fused shared | Expert slots | Routed top-k | Executed top-k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next | 2 | 2,560 | 640 | 320 | 320 | 512 | 1 | 513 | 10 | 11 |
| Qwen3.8-Flash-Next | 4 | 2,560 | 640 | 160 | 192 | 512 | 1 | 513 | 10 | 11 |
| MiMo-V2.6-Flash | 4 | 4,096 | 2,048 | 512 | 512 | 256 | 0 | 256 | 8 | 8 |
| MiMo-V2.6-Flash | 8 | 4,096 | 2,048 | 256 | 256 | 256 | 0 | 256 | 8 | 8 |

The author compared M13 at `a641da286511` with frozen PyHIP A8W4 at `7d881a240f`. Each shape covers powers-of-two token counts from 1 to 32,768 plus M=48. GPU-event medians use 30 full calls after 3 warmups, cycling 4 input groups and flushing 512 MiB before each sample. The author reports the frozen baseline median across 3 runs. [Published runnable protocol and timings](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md#4-性能与精度)

The Qwen arms also timed AITER `43ce3d6bf2` via `aiter.fused_moe` with FP8 `QuantType.per_Token`, using the same generated BF16 source weights, inputs and routes as PyHIP. Qwen TP4 pads both arms to I=192. The harness checks the dispatcher hit the exact `qwen38_flash_next_fp8_ptpc_tuned_fmoe.csv` row. MiMo has frozen PyHIP and M13 results. [AITER revision](https://github.com/ROCm/aiter/commit/43ce3d6bf292bfae4e21466c902c7d2f852a979d) · [Tuned-row check](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/tests/ops/moe/test_moe_a8w4.py#L1559)

## Benchmarking and Profiling

The table samples M=1 decode, M=512 streaming and M=8,192 prefill. The geometric mean and AITER ranges use all 17 token counts per shape. [All 68 per-shape timings with parameters](moe-a8w4-timings.csv)

| Model and TP shape | M=1 (µs) frozen → M13 | M=512 (µs) frozen → M13 | M=8,192 (µs) frozen → M13 | Frozen / M13 geometric mean | M13 / tuned AITER time range |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-Flash-Next TP2 | 18.04 → 18.09 (+0.28%) | 390.39 → 330.38 (−15.37%) | 1,930.55 → 1,722.48 (−10.78%) | 1.092× | 0.597–0.909× |
| Qwen3.8-Flash-Next TP4 | 14.79 → 14.77 (−0.14%) | 294.02 → 222.18 (−24.43%) | 1,347.11 → 1,215.43 (−9.77%) | 1.123× | 0.575–0.842× |
| MiMo-V2.6-Flash TP4 | 28.14 → 28.35 (+0.75%) | 509.97 → 424.13 (−16.83%) | 3,025.11 → 2,668.14 (−11.80%) | 1.120× | — |
| MiMo-V2.6-Flash TP8 | 19.94 → 19.85 (−0.45%) | 316.73 → 244.68 (−22.75%) | 1,760.32 → 1,522.20 (−13.53%) | 1.157× | — |

The author's follow-up measured M13 at 1.000× versus M12 across 68 cases. In four separate same-card rounds versus the faster of R1 and Tingqli, 6/66 cases missed the 2% per-case target. All six were MiMo TP8 (M=16, 512, 1,024, 2,048, 4,096 and 8,192), with up to 4.8% above the target. [M13 performance and acceptance record](https://github.com/luocheng25/pyhip/commit/a641da286511f3f8b703f90b3f9a1ea66b18301a)

### Stage measurements

| Stage | Shape and tokens M | Reported change and GPU time |
|---|---|---|
| Gate/up | MiMo streaming, M=512. TP and local I unspecified. | M11 pairs gate/up per wave, 191 → 153 µs (−19.9%) for the kernel. [M11](https://github.com/luocheng25/pyhip/commit/492d90e4f1b69b25b8ae19cf67b2d96b3a463ad0) |
| Gate/up | Qwen TP2, M=8,192 prefill | R2 before M12 took 1,110 µs versus R1 at 994 µs. M12 reports 8–19% lower prefill gate/up time with paired 16-byte loads and two-tile prefetch. [M12](https://github.com/luocheng25/pyhip/commit/9681cb98cf28de854e59d2dcf6aab3a332115ea6) |
| Down | MiMo TP8 M=2,048 and TP4 M=1,024 | Splitting H across workgroups on TP8 took 232 → 207 µs (−10.8%). Prefill down dispatch with BM64 on TP4 took 303 → 289 µs (−4.6%). [M11](https://github.com/luocheng25/pyhip/commit/492d90e4f1b69b25b8ae19cf67b2d96b3a463ad0) · [M12](https://github.com/luocheng25/pyhip/commit/9681cb98cf28de854e59d2dcf6aab3a332115ea6) |
| Input row quantization | MiMo TP8, M=8,192 | M12 interleaved 16-byte lane loads and optimized the maximum reduction, 75 → 38 µs (−49.3%) for input X. [M12](https://github.com/luocheng25/pyhip/commit/9681cb98cf28de854e59d2dcf6aab3a332115ea6) |
| Routing prep | Qwen TP2, MiMo TP8 and MiMo TP4, each M=2,048 | M13 replaces separate Opus sorting and quantization. Whole-operator time ratios were 0.979×, 0.988× and 1.000× respectively. Dispatch requires H ≥ 2,560 divisible by 512 and M × executed top-k ≤ 24,576. [M13](https://github.com/luocheng25/pyhip/commit/a641da286511f3f8b703f90b3f9a1ea66b18301a) |

## Validation

- The author checked A8W4 outputs against an independent FP32 reference with a 4.5% maximum-error bound relative to the maximum absolute reference value and `calc_diff ≤ 1e-3`. Published maximum errors were 1.33% for decode, 1.19% for streaming and 2.26% for prefill. Atomic accumulation can change low bits. [Reported quality](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md#5-测试覆盖)
- The source reports bitwise-equal M12 gate/up and quantization outputs in its same-process A/B. M13's fused prep retained duplicate expert routes. The earlier Opus path kept one route per repeated expert. [M12 A/B](https://github.com/luocheng25/pyhip/commit/9681cb98cf28de854e59d2dcf6aab3a332115ea6) · [Routing tests](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/tests/ops/moe/test_moe_a8w4.py#L867)
- These measurements are from the pinned PyHIP study. ATOM DeepSeek-V4.1-Flash TP4 uses H=5,120, global I=2,304, local I=576, 385 slots, executed top-k=7 (6 routed plus 1 shared) and SwiGLU clamp 10.0. This H exceeds the fast PyHIP decode guard of H ≤ 4,096. [DeepSeek configuration](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/tests/models/deepseek_v41/fixtures/config.json#L52) · [PyHIP dispatch guard](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_a8w4.py#L615)
