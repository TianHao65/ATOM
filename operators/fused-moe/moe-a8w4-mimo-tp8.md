# Luocheng runtime A8W4 MoE — MiMo-V2.6-Flash TP8

## What

This historical MI308X experiment compares the original PyHIP implementation recorded in the study with the M13 optimization. It measures the complete `fused_moe_a8w4` operator, including its routing preparation, activation quantization, gate/up and down stages. The FlyDSL implementation packs MXFP4 weights during loading and decodes them into FP8 in registers during computation. [PyHIP M13](https://github.com/luocheng25/pyhip/commit/a641da286511f3f8b703f90b3f9a1ea66b18301a) · [Kernel design](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/kernels.md)

## Data

| GPU | H | Global I | Local I | Padded I | Routed experts | Fused shared | Expert slots | Routed top-k | Executed top-k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MI308X (gfx942), TP8 shape | 4096 | 2048 | 256 | 256 | 256 | 0 | 256 | 8 | 8 |

Single-GPU operator benchmark using the TP8 weight shape, BF16 inputs/outputs and SiLU. M is the number of input tokens. The study used powers of two from 1 to 32,768 plus 48. The median is from 30 timed calls after three warmups, cycling four inputs and flushing 512 MiB before each call. Timing encloses the full call's GPU work. [Benchmark harness](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/tests/ops/moe/test_moe_a8w4.py#L1450)

The author records the frozen PyHIP baseline as `7d881a240f` and reports the median of three benchmark runs. [Published benchmark and frozen baseline](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md#4-性能与精度)

The published run used uniformly sampled routed experts. The source does not include an AITER FP8 PTPC arm for this shape because its pinned AITER table lacked a matching MiMo row. The baseline column below is frozen PyHIP A8W4.

## Benchmarking and Profiling

Across these 17 rows, the geometric mean of frozen / M13 latency is **1.157×**, calculated from the published per-case medians.

| M | Frozen A8W4 (µs) | M13 A8W4 (µs) | Frozen / M13 |
|---:|---:|---:|---:|
| 1 | 19.94 | 19.85 | 1.010× |
| 2 | 28.1 | 27.76 | 1.012× |
| 4 | 43.81 | 40 | 1.094× |
| 8 | 70.34 | 64.65 | 1.087× |
| 16 | 113.16 | 100.26 | 1.129× |
| 32 | 178.51 | 141.81 | 1.260× |
| 48 | 201.33 | 173.65 | 1.160× |
| 64 | 218.14 | 178.09 | 1.224× |
| 128 | 238.4 | 194.47 | 1.225× |
| 256 | 255.27 | 203.83 | 1.251× |
| 512 | 316.73 | 244.68 | 1.296× |
| 1,024 | 417.47 | 347.53 | 1.201× |
| 2,048 | 652.78 | 548.77 | 1.190× |
| 4,096 | 1004.56 | 878.69 | 1.144× |
| 8,192 | 1760.32 | 1522.2 | 1.155× |
| 16,384 | 3289.01 | 2855.43 | 1.152× |
| 32,768 | 6435.16 | 5685.13 | 1.132× |

The M=1 path runs the regular decode kernels. M=2–48 tests reuse weights across routes to the same expert, M=64–1024 use streaming, and M ≥ 2048 use prefill. At local I=256, streaming Down can accumulate directly into the output. The separate M=2048 prep A/B reported a 0.988× time ratio. [Source timing table](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md#4-性能与精度)

In a separate four-round same-card comparison against the faster R1 or Tingqli implementation, this TP8 configuration had six cases above the 2% per-case target. M=16, 512, 1,024, 2,048, 4,096 and 8,192 had target-normalized ratios of 1.005, 1.008, 1.029, 1.023, 1.048 and 1.007. These ratios use the target as the denominator, which includes the 2% allowance. [M13 acceptance record](https://github.com/luocheng25/pyhip/commit/a641da286511f3f8b703f90b3f9a1ea66b18301a)

## Validation

- The source study checked timed A8W4 outputs against an independent FP32 reference. Across the published shapes, maximum errors were 1.33% in decode, 1.19% in streaming, and 2.26% in prefill, against a 4.5% max-error bound and `calc_diff ≤ 1e-3`. Atomic accumulation can change the low bits. [Source validation](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md#5-测试覆盖)
- Current DeepSeek-V4.1 Flash TP4 uses H=5120, global I=2304, local logical I=576, 385 fused slots and 7 routes per token (6 routed + 1 shared), with a SwiGLU clamp of 10.0. The PyHIP fast decode dispatch is restricted to H ≤ 4096. A DeepSeek comparison needs a size- and activation-matched run through AITER `fused_moe` and its own tuning rows. [DeepSeek fixture](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/tests/models/deepseek_v41/fixtures/config.json#L52) · [PyHIP dispatch](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_a8w4.py#L615)
