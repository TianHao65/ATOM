# Luocheng dense A8W4 GEMM on MI308X

## What

Luocheng's M0 FlyDSL GEMM packs MXFP4 weights once, quantizes activations per token on each call, and decodes weights into FP8 in registers before MFMA. One tile configuration serves the full M range. The baseline uses AITER FP8 PTPC with the same generated MXFP4 source weights converted to per-channel FP8. [M0 commit](https://github.com/luocheng25/aiter/commit/ba9dd6a3493bb2b6313e5c52664300f8f9f75438) · [Implementation](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/aiter/ops/flydsl/gemm_a8w4_ptpc.py)

## Data

Historical Qwen3.8-Flash-Next single-operator benchmark on one MI308X (gfx942, 80 CUs), FlyDSL 0.5.3, and an AITER worktree based on `162173e89`. Hidden size is 2,560. For `Y[M,N] = X[M,K] × W[N,K]ᵀ`, all six TP-local weight shapes run with `M = 2⁰, 2¹, …, 2¹⁵` (1 to 32,768 tokens, 96 cases). [Benchmark shapes](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/op_tests/test_gemm_a8w4_ptpc.py#L54)

| TP shape | Projection | Local N | Local K |
|---|---|---:|---:|
| TP2 | GDN `in_proj_qkvz` | 8,192 | 2,560 |
| TP2 | Attention QKV with Q gate | 6,656 | 2,560 |
| TP2 | GDN or attention output | 2,560 | 3,072 |
| TP4 | GDN `in_proj_qkvz` | 4,096 | 2,560 |
| TP4 | Attention QKV with Q gate | 3,584 | 2,560 |
| TP4 | GDN or attention output | 2,560 | 1,536 |

Inputs and output are BF16. A8W4 uses dynamic per-token E4M3FNUZ activations (FP32 scale `amax/240` for nonzero rows), packed E2M1 weights with one E8M0 scale per 32 K values, and FP32 accumulation. The GEMM tile is `64×128×128` (M×N×K), with four wave64s per workgroup and FP8 `16×16×32` MFMA. The weight preparation runs before timing. [Design and kernel parameters](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/aiter/ops/flydsl/kernels/gemm_a8w4_ptpc/README.md#2-总体设计)

## Benchmarking and Profiling

Both paths time the full call, including activation quantization, with one AITER `run_perftest` invocation per arm and shape. The harness uses its defaults of 101 iterations and two warmup calls with graph timing off. The FP8 PTPC path calls `per_token_quant_hip` plus `gemm_a8w8_bpreshuffle`. It selects tuned CK rows for TP2 except the output projection at M=32,768. TP4 uses default CK. [Timing loop](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/op_tests/test_gemm_a8w4_ptpc.py#L677) · [Timing defaults](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/aiter/test_common.py#L243)

| M range | A8W4 / FP8 PTPC latency (geomean) | Latency ratio range | Throughput, FP8 PTPC → A8W4 |
|---|---:|---:|---:|
| 1–32 | 2.43× | 1.89–3.60× | 0.99 → 0.22 TB/s |
| 64–2,048 | 1.32× | 0.97–2.60× | 179 → 136 TFLOPS |
| 4,096–32,768 | 1.12× | 1.05–1.17× | 379 → 338 TFLOPS |

M0 was slower in 95 of 96 cases. For M ≤ 32 its standalone quantizer is followed by only N/128 GEMM workgroups (20–64 across these shapes). The author publishes pooled TP2/TP4 statistics from one self-test pass. Per-shape latencies and an independent repeat are unavailable. [Reported measurements](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/aiter/ops/flydsl/kernels/gemm_a8w4_ptpc/README.md#4-性能与精度)

## Validation

- The author reports all 96 timed cases passing independent FP32 references built from each arm's quantized operands. The gate required finite outputs, `|error| ≤ 0.01 × (|reference| + row_max_abs_reference)`, and `calc_diff ≤ 1e-5`. The A8W4 reference applies the weight preparation rule that zeroes MXFP4 groups with an E8M0 scale delta of at least 12 from the channel maximum. [Reference and grading](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/op_tests/test_gemm_a8w4_ptpc.py#L143)
- Profiler checks on six basic cases matched the declared kernel names. Additional coverage includes tile boundaries, bit-exact activation quantization, noncontiguous TP weight slices, and graph replay. The reported performance applies to these Qwen3.8 TP-local shapes. [Coverage](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/aiter/ops/flydsl/kernels/gemm_a8w4_ptpc/README.md#5-测试覆盖)
