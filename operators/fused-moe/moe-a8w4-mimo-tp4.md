# Luocheng runtime A8W4 MoE — MiMo-V2.6-Flash TP4

## What

This historical MI308X experiment compares the original PyHIP implementation recorded in the study with the M13 optimization. It measures the complete `fused_moe_a8w4` operator, including its routing preparation, activation quantization, gate/up and down stages. The FlyDSL implementation packs MXFP4 weights during loading and decodes them into FP8 in registers during computation. [PyHIP M13](https://github.com/luocheng25/pyhip/commit/a641da286511f3f8b703f90b3f9a1ea66b18301a) · [Kernel design](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/kernels.md)

## Data

| GPU | H | Global I | Local I | Padded I | Routed experts | Fused shared | Expert slots | Routed top-k | Executed top-k |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MI308X (gfx942), TP4 shape | 4096 | 2048 | 512 | 512 | 256 | 0 | 256 | 8 | 8 |

Single-GPU operator benchmark using the TP4 weight shape, BF16 inputs/outputs and SiLU. M is the number of input tokens. The study used powers of two from 1 to 32,768 plus 48. The median is from 30 timed calls after three warmups, cycling four inputs and flushing 512 MiB before each call. Timing encloses the full call's GPU work. [Benchmark harness](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/tests/ops/moe/test_moe_a8w4.py#L1450)

The author records the frozen PyHIP baseline as `7d881a240f` and reports the median of three benchmark runs. [Published benchmark and frozen baseline](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md#4-性能与精度)

The published run used uniformly sampled routed experts. The source does not include an AITER FP8 PTPC arm for this shape because its pinned AITER table lacked a matching MiMo row. The baseline column below is frozen PyHIP A8W4.

## Benchmarking and Profiling

Across these 17 rows, the geometric mean of frozen / M13 latency is **1.120×**, calculated from the published per-case medians.

| M | Frozen A8W4 (µs) | M13 A8W4 (µs) | Frozen / M13 |
|---:|---:|---:|---:|
| 1 | 28.14 | 28.35 | 0.992× |
| 2 | 42.44 | 41.97 | 1.010× |
| 4 | 70.08 | 65.15 | 1.078× |
| 8 | 114.72 | 104.12 | 1.102× |
| 16 | 181.95 | 157.66 | 1.154× |
| 32 | 317.69 | 236.48 | 1.342× |
| 48 | 320.37 | 297.84 | 1.077× |
| 64 | 353.23 | 309.02 | 1.145× |
| 128 | 388.53 | 342.63 | 1.133× |
| 256 | 404.24 | 370.94 | 1.090× |
| 512 | 509.97 | 424.13 | 1.203× |
| 1,024 | 663.4 | 591.08 | 1.120× |
| 2,048 | 1050.22 | 916.65 | 1.146× |
| 4,096 | 1678.85 | 1477.84 | 1.136× |
| 8,192 | 3025.11 | 2668.14 | 1.134× |
| 16,384 | 5688.87 | 5106.33 | 1.113× |
| 32,768 | 11127 | 10079.1 | 1.104× |

The M=1 path runs the regular decode kernels. M=2–48 tests reuse weights across routes to the same expert, M=64–1024 use streaming, and M ≥ 2048 use prefill. At local I=512, the streaming path retains the route output and reduction because the atomic epilogue was slower. The separate M=2048 prep A/B reported a 1.00× time ratio. [Source timing table](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md#4-性能与精度)

## Validation

- The source study checked timed A8W4 outputs against an independent FP32 reference. Across the published shapes, maximum errors were 1.33% in decode, 1.19% in streaming, and 2.26% in prefill, against a 4.5% max-error bound and `calc_diff ≤ 1e-3`. Atomic accumulation can change the low bits. [Source validation](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_gemm_2stage/README.md#5-测试覆盖)
- Current DeepSeek-V4.1 Flash TP4 uses H=5120, global I=2304, local logical I=576, 385 fused slots and 7 routes per token (6 routed + 1 shared), with a SwiGLU clamp of 10.0. The PyHIP fast decode dispatch is restricted to H ≤ 4096. A DeepSeek comparison needs a size- and activation-matched run through AITER `fused_moe` and its own tuning rows. [DeepSeek fixture](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/tests/models/deepseek_v41/fixtures/config.json#L52) · [PyHIP dispatch](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_a8w4.py#L615)
