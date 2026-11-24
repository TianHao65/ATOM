# Luocheng runtime A8W4 MoE per-row quantization

## What

M12 optimized the per-row E4M3FNUZ activation quantizer used on input X and the gate/up intermediate. The code interleaves 16-byte lane loads, uses `maxnumf` for the maximum reduction, and removes redundant clamping. [M12 change](https://github.com/luocheng25/pyhip/commit/9681cb98cf28de854e59d2dcf6aab3a332115ea6)

## Data

The author reported stage measurements on MI308X from standalone PyHIP A8W4. Qwen3.8 TP2 used H=2,560, local I=320, 513 slots and top-k=11. MiMo TP8 used H=4,096, local I=256, 256 slots and top-k=8. Both cases have M=8,192.

## Benchmarking and Profiling

| Parallel shape | Comparison | Quantization time (µs) |
|---|---|---:|
| Qwen3.8 TP2 | Earlier R2 stage breakdown versus R1 | 2 × 49 versus 2 × 27 |
| MiMo TP8 | M12 input X row quantization before and after | 75 → 38 (−49.3%) |

The MiMo number measures the input-quantization kernel. The Qwen numbers are the prior breakdown across the two quantization launches. [Reported stage measurements](https://github.com/luocheng25/pyhip/commit/9681cb98cf28de854e59d2dcf6aab3a332115ea6)

## Validation

- The author reports bitwise-equal quantization outputs in the M12 same-process comparison. Unit coverage includes zero, rounding midpoints, small scales and exact IEEE division behavior. [Test source](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/tests/ops/moe/test_moe_a8w4.py)
- The source compares PyHIP milestones. An ATOM original/optimized quantization comparison on DeepSeek H=5,120 remains a separate operator experiment.
