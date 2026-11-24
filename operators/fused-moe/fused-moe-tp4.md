# Fused MoE on MI308X TP4

## What

Tune the DeepSeek-V4.1-Flash-1017 TP4 Base rank-local FP8 PTPC `aiter.fused_moe` call used on both sides of the Ballot A/B. The pinned AITER default runs clamped SiLU through the gfx942 FlyDSL implementation. Two exact-shape CSV rows select faster configurations for the 4-token Decode and 8,192-token Prefill buckets. [Serving ATOM revision](https://github.com/TianHao65/ATOM/commit/e309e005dfa83dd323dfb959b040d0af8bb6bceb) · [AITER revision](https://github.com/ROCm/aiter/commit/ad0d5e208eb2d49ab658c3c161aa5c3502dfc2a1) · [Runtime tuning CSV](dsv41_mi308_tp4_ptpc_tuned_fmoe.csv)

## Data

| Parameter | Runtime value used by both arms |
|---|---|
| Device and parallelism | MI308X gfx942, 80 CUs, one rank of TP4 Base, DP1, EP off |
| Hidden and expert intermediate | H = 5,120, global I = 2,304, logical and allocated rank-local I = 576 |
| Experts and routes | Router E = 384 and Top-6. Packed FusedMoE E = 385 and Top-7. Shared slot ID 384 has weight 1.0 on every token. |
| FP8 weights | Preshuffled gate/up shape `[385,1152,5120]`, down `[385,5120,576]`, FP32 per-channel scales |
| Activation and layout | BF16 input and output, dynamic E4M3FNUZ FP8 A/W for `QuantType.per_Token`, SiLU clamp 10.0, G1U1 separated gate/up, stage-1 router weighting off, no expert mask, biases or padding arguments |
| Token shapes | Decode graph buffers M = 1, 2, 4, 8, 16, 32, 48, 64. The 100K/24 trace has Prefill buffers M = 2,048 with 1,952 live rows and M = 8,192 with 8,192 live rows. |

ATOM stages the FP8 shared expert during loading and copies it into the final slot after the routed MXFP4 weights are converted to FP8 PTPC. The `per_Token` target keeps 576 channels per TP rank. The native MXFP4 target can allocate a different intermediate width for its layout. Rank 0 serving traces show the router's `384` template width and MoE sorting's 385-expert grid. The C1 gate/up grid is `[36,7,1]` and down is `[80,7,1]` in both arms. [Model config](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/tests/models/deepseek_v41/fixtures/config.json#L25) · [Packed expert slots](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/atom/model_ops/moe.py#L3683) · [FP8 target allocation](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/atom/model_ops/moe.py#L3049) · [Serving traces](https://github.com/TianHao65/ATOM/blob/3712bf1cde8ffdb7ac7ec2f1ed95d1928d15c6dd/README.md#benchmarking-and-profiling)

## Benchmarking and Profiling

All candidates entered through `aiter.fused_moe` in image `cfd987217fc4` with AITER `ad0d5e20`. Each pair used the same synthetic seed, preshuffled weights, BF16 inputs, six distinct routed IDs, fixed shared ID and live-token count. GPU events timed graph replay after warmup. The sweep screened 62 candidate/shape pairs with five paired windows where output checks passed. Two selected rows repeated the comparison with freshly captured graphs and nine alternating windows. A row had to clear a 2% paired-median improvement threshold on the sampled shape or shapes that use its key. [Full measured sweep](fused-moe-tp4-measurements.csv)

| Tokens / live rows | CSV bucket | ATOM original P50 (µs) | Tuned P50 (µs) | Median paired delta | Selected configuration |
|---|---:|---:|---:|---:|---|
| 1 / 1 | 1 | 28.74 | — | — | Default batch-1 |
| 2 / 2 | 2 | 43.03 | — | — | Default |
| 4 / 4 | 4 | 73.75 | 71.12 | −3.67% | `16_16_16_False_True:default:0` |
| 8 / 8 | 8 | 156.91 | — | — | Default |
| 16 / 16 | 16 | 284.01 | — | — | Default |
| 32 / 32 | 32 | 448.42 | — | — | Default |
| 48 / 48 | 64 | 629.48 | — | — | Default |
| 64 / 64 | 64 | 756.92 | — | — | Default |
| 2,048 / 1,952 | 2,048 | 1,821.85 | — | — | Default |
| 8,192 / 8,192 | 8,192 | 4,414.89 | 4,309.79 | −2.12% | `64_128_256_True:default:0` |

The table uses revalidation windows for the selected rows and explicit-default screening windows for the others. AITER rounds M up to the next power of two at these sizes. M48 and M64 therefore share bucket 64 and no candidate cleared the improvement threshold for both.

At M4 the delivered CSV changes the observed gate/up and down kernels from `moe_2stage_gateup_splitk_0` and `moe_2stage_down_splitk_0` to `moe_2stage_gateup_batch1_0` and `moe_2stage_down_batch1_0`. At M8192 both configurations emit `moe_2stage_gateup_prefill_1x4_0` and `moe_2stage_down_prefill_1x4_0`. The lookup and stage geometry change the prefill K tile from 128 to 256 with a common kernel symbol. [Selected kernel trace audit](fused-moe-tp4-kernels.json)

### Historical runtime A8W4 prototype on ATOM

The Nov 22 experiment used MI308X TP4, H=5120, global I=2304, logical rank-local I=576, padded W4 I=640, E384 and Top-6. The model ran its FP8 shared branch separately. Original ATOM used load-time MXFP4-to-FP8 PTPC. The prototype kept MXFP4 weights and used Triton runtime A8W4. Both serving arms used SwiGLU clamp 10.0. [ATOM base revision](https://github.com/ROCm/ATOM/commit/5f6bdb75c19a0e4bc57512ef425973d53f0f54f4) · [Historical samples and trace hashes](evidence/deepseek-moe-a8w4-20261122.json)

The isolated probe times post-router work, including route preparation, gate/up, down and combine. M1 used the checkpoint's layer 20 rank-0 weight shard and 12 matched route sets. M8192 used generated packed weights. The FP8 reference used the FlyDSL backend entry directly.

| M | FP8 PTPC (µs) | Runtime A8W4 (µs) | FP8-MFMA C1 prototype (µs) | Rebased C1 prototype (µs) |
|---:|---:|---:|---:|---:|
| 1 | 29.83 | 292.06 | 144.42 | 89.64 |
| 8,192 | 3,980.40 | 38,279.66 | | |

The rebased C1 prototype matched its staged reference on the tested input. In separate 2,048/16 C1 serving traces, rank-0 gate/up plus down rose from 45.337 to 338.131 ms in Prefill and from 2.172 to 12.458 ms per Decode step. Seven timed matched requests had median ETE of 1.601 → 2.041 s (+27.47%) with runtime A8W4. [Historical timing method and stage breakdown](evidence/deepseek-moe-a8w4-20261122.json)

Luocheng's M13 fast Decode has an H ≤ 4096 guard. The historical 384/6 ATOM prototype and the M13 studies use separate kernel paths from the current 385/7 PTPC tuning rows. [M13 dispatch](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/src/pyhip/ops/moe/flydsl/moe_a8w4.py#L615)

## Validation

- The runtime emitted lookup keys for gfx942, 80 CUs, the padded token bucket, H5120, I576, E385, Top-7, SiLU, BF16 output, E4M3FNUZ A and W, `per_Token`, G1U1 and stage-1 weighting off. The existing AITER tables have no exact row for this key. Rows for E384/Top-6, I640 or another architecture/quantization use different keys. [AITER lookup](https://github.com/ROCm/aiter/blob/ad0d5e208eb2d49ab658c3c161aa5c3502dfc2a1/aiter/fused_moe.py#L4572)
- Synthetic outputs were screened against original ATOM dispatch on the same live rows with normalized RMSE ≤ 2.5% and maximum absolute difference ≤ 0.05. M4 passed revalidation at 0.379% normalized RMSE with maximum absolute difference 0.015625. M8192 matched bitwise. A separate process loaded the saved CSV through the normal selector and repeated the check with fresh inputs. The timings cover rank-local kernels with synthetic inputs. Serving trace attribution comes from the earlier checkpoint run. Full-model quality and end-to-end gains for these rows are pending.
- The selected device was idle before each benchmark and ended at 0% utilization and its original 297,611,264-byte VRAM baseline, with no remaining process. Across 167 ownership samples neither benchmark overlapped another GPU 7 workload.

To use the rows, start a fresh worker in the matching AITER revision with `AITER_CONFIG_FMOE=/path/to/current-resolved-fmoe.csv:/path/to/dsv41_mi308_tp4_ptpc_tuned_fmoe.csv`. The first file should contain the stock and per-model rows normally resolved by the worker. The two files must have distinct tuning keys. [CSV resolver](https://github.com/ROCm/aiter/blob/ad0d5e208eb2d49ab658c3c161aa5c3502dfc2a1/aiter/jit/core.py#L818)
