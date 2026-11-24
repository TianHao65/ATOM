# Luocheng runtime A8W4 MoE — Qwen3.8-Flash-Next TP2

## What

This historical MI308X experiment compares the original PyHIP implementation recorded in the study and the M13 optimization, and includes the size-matched AITER FP8 PTPC arm. It measures the complete `fused_moe_a8w4` operator, including its routing preparation, activation quantization, gate/up and down stages. The FlyDSL implementation packs MXFP4 weights during loading and decodes them into FP8 in registers during computation. [PyHIP M13](https://github.com/luocheng25/pyhip/commit/a641da286511f3f8b703f90b3f9a1ea66b18301a) · [Kernel design](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/kernels.md)

## Data

| GPU | H | Global I | Local I | Padded I | Routed experts | Fused shared | Expert slots | Routed top-k | Executed top-k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MI308X (gfx942), TP2 shape | 2560 | 640 | 320 | 320 | 512 | 1 | 513 | 10 | 11 |

Single-GPU operator benchmark using the TP2 weight shape, BF16 inputs/outputs and SiLU. M is the number of input tokens. The study used powers of two from 1 to 32,768 plus 48. The median is from 30 timed calls after three warmups, cycling four inputs and flushing 512 MiB before each call. Timing encloses the full call's GPU work. [Benchmark harness](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/tests/ops/moe/test_moe_a8w4.py#L1450)

The author records the frozen PyHIP baseline as `7d881a240f` and reports the median of three benchmark runs. [Published benchmark and frozen baseline](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md#4-性能与精度)

The comparison arm called `aiter.fused_moe` with FP8 `QuantType.per_Token`. Both arms used the same generated BF16 source weights, activation inputs and routing IDs/weights. The test asserted that the dispatcher used the row from `qwen38_flash_next_fp8_ptpc_tuned_fmoe.csv` at the exact shape above. The published table includes FlyDSL batch-1, split-K, and prefill dispatch, CK two-stage, and ASM one-stage depending on M. [AITER reference revision](https://github.com/ROCm/aiter/commit/43ce3d6bf292bfae4e21466c902c7d2f852a979d) · [Bench and tuned-row checks](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/tests/ops/moe/test_moe_a8w4.py#L1450)

## Benchmarking and Profiling

Across these 17 rows, the geometric mean of frozen / M13 latency is **1.092×**, calculated from the published per-case medians. The published M13 / tuned AITER FP8 PTPC latency ratios range from **0.597× to 0.909×**.

| M | Frozen A8W4 (µs) | M13 A8W4 (µs) | Frozen / M13 | AITER FP8 PTPC (µs) | M13 / AITER |
|---:|---:|---:|---:|---:|---:|
| 1 | 18.04 | 18.09 | 0.997× | 23.51 | 0.769× |
| 2 | 27.16 | 27.2 | 1.001× | 36.90 | 0.736× |
| 4 | 40.19 | 40.25 | 0.998× | 62.40 | 0.645× |
| 8 | 65.03 | 65.13 | 0.999× | 108.94 | 0.597× |
| 16 | 100.45 | 100.88 | 0.997× | 166.63 | 0.605× |
| 32 | 167.79 | 168.15 | 0.997× | 247.46 | 0.680× |
| 48 | 255.92 | 226.86 | 1.128× | 308.11 | 0.735× |
| 64 | 261.68 | 209.31 | 1.249× | 345.50 | 0.606× |
| 128 | 317.8 | 263.67 | 1.204× | 426.29 | 0.619× |
| 256 | 343.32 | 291.47 | 1.176× | 443.34 | 0.658× |
| 512 | 390.39 | 330.38 | 1.181× | 461.91 | 0.715× |
| 1,024 | 478.8 | 428.67 | 1.117× | 481.77 | 0.890× |
| 2,048 | 670.6 | 606.7 | 1.105× | 712.75 | 0.850× |
| 4,096 | 1152.72 | 1020.67 | 1.131× | 1130.05 | 0.903× |
| 8,192 | 1930.55 | 1722.48 | 1.121× | 1958.77 | 0.882× |
| 16,384 | 3581.26 | 3193.17 | 1.122× | 3611.59 | 0.885× |
| 32,768 | 6979.96 | 6354.57 | 1.099× | 6988.19 | 0.909× |

The small-M path covers 1–32 tokens. Streaming covers the sampled 48–1024 cases, and prefill covers M ≥ 2048. At M=2048 the M13 single-launch routing prep plus input quantization ran at 0.979× the GPU time of the separate Opus sorting and quantization path in the author’s additional same-process A/B. [Source timing table](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md#4-性能与精度)

## Validation

- The source study checked timed A8W4 outputs against an independent FP32 reference. Across the published shapes, maximum errors were 1.33% in decode, 1.19% in streaming, and 2.26% in prefill, against a 4.5% max-error bound and `calc_diff ≤ 1e-3`. Atomic accumulation can change the low bits. [Source validation](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md#5-测试覆盖)
- Current DeepSeek-V4.1 Flash TP4 uses H=5120, global I=2304, local logical I=576, 385 fused slots and 7 routes per token (6 routed + 1 shared), with a SwiGLU clamp of 10.0. The PyHIP fast decode dispatch is restricted to H ≤ 4096. A DeepSeek comparison needs a size- and activation-matched run through AITER `fused_moe` and its own tuning rows. [DeepSeek fixture](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/tests/models/deepseek_v41/fixtures/config.json#L52) · [PyHIP dispatch](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_a8w4.py#L615)
