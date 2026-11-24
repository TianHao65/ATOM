# Luocheng runtime A8W4 MoE routing prep

## What

M13 dispatches `a8w4_stream_prep` for eligible prefill shapes, combining expert sorting, input row quantization and location ID emission in one launch. The dispatch requires H ≥ 2,560, H divisible by 512, and at most 24,576 routes (M × top-k). The full operator continues through gate/up, intermediate quantization, down and routed output reduction. [M13 source and A/B](https://github.com/luocheng25/pyhip/commit/a641da286511f3f8b703f90b3f9a1ea66b18301a)

## Data

The author made a same-process comparison on MI308X. The original path used Opus sorting followed by input quantization. The optimized arm used the fused prep and measured the complete A8W4 operator.

| Parallel shape | H | Local I | E | Executed top-k |
|---|---:|---:|---:|---:|
| Qwen3.8 TP2 | 2,560 | 320 | 513 | 11 |
| MiMo TP8 | 4,096 | 256 | 256 | 8 |
| MiMo TP4 | 4,096 | 512 | 256 | 8 |

## Benchmarking and Profiling

| Parallel shape | M | Fused prep path / original path GPU time |
|---|---:|---:|
| Qwen3.8 TP2 | 2,048 | 0.979× |
| Qwen3.8 TP2 | 8,192 | 1.02× |
| MiMo TP8 | 2,048 | 0.988× |
| MiMo TP8 | 4,096 | 1.01× |
| MiMo TP8 | 8,192 | 1.03× |
| MiMo TP8 | 16,384 | 1.06× |
| MiMo TP4 | 2,048 | 1.00× |

Large-M rows are development comparisons that motivated the final route limit. The M13 review reported 1.000× incremental geometric-mean speedup over M12 across all 68 cases. Its earlier M11 profile of Qwen TP2 M=8,192 measured 52 µs of sorting versus 38 µs in R1. [M13 results](https://github.com/luocheng25/pyhip/commit/a641da286511f3f8b703f90b3f9a1ea66b18301a) · [Earlier sorting breakdown](https://github.com/luocheng25/pyhip/commit/492d90e4f1b69b25b8ae19cf67b2d96b3a463ad0)

## Validation

- The author reports bitwise-equal row quantization, matched sorting format and identical per-case maximum numerical errors in both arms over 12 tested shapes. The fused prep retains repeated expert IDs for a token. The Opus path keeps one repeated route. [Coverage and duplicate-route semantics](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/tests/ops/moe/test_moe_a8w4.py#L867)
- This is expert-ID sorting and quantization inside MoE. The attention indexer selects tokens from logits and has a separate report.
