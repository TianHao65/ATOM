# Luocheng dense A8W4 GEMM — Qwen3.8 TP2 and TP4

## What

The M0 FlyDSL operator prepares MXFP4 weights in a packed layout at load time. The timed call quantizes BF16 activations to FP8 per token, decodes weights in registers, and uses one BM64/BN128/TK128 FP8 MFMA GEMM. The reference uses the AITER PTPC FP8 path (`per_token_quant_hip` and `gemm_a8w8_bpreshuffle`) on the same generated source weights. [M0 commit](https://github.com/luocheng25/aiter/commit/ba9dd6a3493bb2b6313e5c52664300f8f9f75438) · [Design and kernel details](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/aiter/ops/flydsl/kernels/gemm_a8w4_ptpc/README.md)

## Data

MI308X (gfx942), FlyDSL 0.5.3. Both arms used the same AITER worktree based on `162173e89`, BF16 inputs and outputs, and 16 powers-of-two M values from 1 to 32,768. M is the number of rows in X. The six distinct (N, K) shapes below produce 96 single-GPU cases. [Exact shapes and reference](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/op_tests/test_gemm_a8w4_ptpc.py#L54)

| Parallel shape | Projection | N | K |
|---|---|---:|---:|
| TP2 | GDN input qkvz | 8,192 | 2,560 |
| TP2 | Attention qkv with Q gate | 6,656 | 2,560 |
| TP2 | GDN/attention output | 2,560 | 3,072 |
| TP4 | GDN input qkvz | 4,096 | 2,560 |
| TP4 | Attention qkv with Q gate | 3,584 | 2,560 |
| TP4 | GDN/attention output | 2,560 | 1,536 |

## Benchmarking and Profiling

The author timed both complete calls, including activation quantization, within each case using AITER `run_perftest`. The source publishes latency ratios pooled over TP2 and TP4. The reference selected tuned CK rows for TP2 except the output shape at M=32,768, and used default CK for TP4. [Timing loop](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/op_tests/test_gemm_a8w4_ptpc.py#L677)

| M range | A8W4 / AITER FP8 PTPC latency (geomean) | Ratio range | A8W4 throughput | AITER throughput |
|---|---:|---:|---:|---:|
| 1–32 | 2.43× | 1.89–3.60× | 0.22 TB/s | 0.99 TB/s |
| 64–2,048 | 1.32× | 0.97–2.60× | 136 TFLOPS | 179 TFLOPS |
| 4,096–32,768 | 1.12× | 1.05–1.17× | 338 TFLOPS | 379 TFLOPS |

M0 was slower on 95 of 96 cases. Small M launches a separate quantization kernel and assigns N/128 workgroups to the GEMM. The benchmark does not publish per-TP latency summaries or an independently repeated performance run. [Reported performance and limits](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/aiter/ops/flydsl/kernels/gemm_a8w4_ptpc/README.md#4-性能与精度)

## Validation

- The author reports all 96 timed outputs passing independent FP32 reference checks. The pointwise bound was 1% of the reference value plus 1% of that row's maximum magnitude, with `calc_diff ≤ 1e-5`. The reference applies the same zeroing for MXFP4 group scale delta ≥ 12. [Test harness](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/op_tests/test_gemm_a8w4_ptpc.py)
- These are Qwen3.8 projection shapes. An ATOM DeepSeek comparison needs the exact live projection geometry and matched FP8 reference. The AITER operator here is a standalone dense GEMM. [Pinned AITER entry](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/aiter/ops/flydsl/gemm_a8w4_ptpc.py)
