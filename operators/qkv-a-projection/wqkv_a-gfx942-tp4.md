# Q/KV input projection (`wqkv_a`) on MI308X TP4

## What

The merged replicated projection maps 5,120 input features to the 1,280-dimensional Q latent and the 512-dimensional KV latent. ATOM original loads native block-scaled FP8 weights into BF16 and uses a BF16 GEMM. With PTPC, it keeps FP8 weights and runs FP8 × FP8 GEMM with BF16 output. [Projection implementation](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/atom/models/deepseek_v41/attention.py#L293) · [Native load-time PTPC patch](https://github.com/TianHao65/ATOM/commit/d42d306aa19db441d21f84dad09b3c613b227767)

## Data

Both historical PTL52 comparisons used MI308X, TP4 Base C1, image `cfd987217fc4` and AITER `ad0d5e20`. Each used a separate 100K/24 profile. One held the native checkpoint `08c8657` fixed and toggled the ATOM load-time PTPC patch. The other ran original `08c8657` and offline-converted Bobo FP4PTPC `97cc9b32` through ATOM `5f6bdb75`. Each pair has its own original timings.

## Benchmarking and Profiling

Rank 0 per-call P50 sums the GEMM and any identified split-K reduction. The final column includes any separately launched input quantization. Prefill uses the first 20 full-row attention layers of the 8,192-token chunk. Decode uses 24 one-token graph replays with up to 40 selected layers per step.

| ATOM original → optimized | Phase | Per-rank M × N × K | Calls per arm | GEMM P50 (µs) | GEMM + separate quant P50 (µs) |
|---|---|---:|---:|---:|---:|
| Native checkpoint, ATOM load-time PTPC | Prefill | 8,192 × 1,792 × 5,120 | 20 | 649.822 → 319.380 (-50.85%) | 649.822 → 363.799 (-44.02%) |
| Native checkpoint, ATOM load-time PTPC | Decode | 1 × 1,792 × 5,120 | 960 | 12.399 → 8.039 (-35.16%) | 12.399 → 12.798 (+3.22%) |
| Native checkpoint → Bobo FP4PTPC on ATOM | Prefill | 8,192 × 1,792 × 5,120 | 20 | 806.022 → 410.481 (-49.07%) | 806.022 → 466.460 (-42.13%) |
| Native checkpoint → Bobo FP4PTPC on ATOM | Decode | 1 × 1,792 × 5,120 | 960 | 15.319 → 9.159 (-40.21%) | 15.319 → 14.518 (-5.23%) |

Standalone input quantization is included in the rightmost column. At C1 it offsets the native PTPC GEMM gain, leaving 12.399 → 12.798 µs (+3.22%) for the measured chain.

## Validation

The trace audit attributed projection kernels by attention markers and HIP launch correlation. Shapes come from TP4 model geometry and phase annotations. The native load-time conversion check matched all 290 converted weight/scale pairs to Bobo’s offline FP4PTPC checkpoint. These are serving-trace slices.

Rank 0 original trace SHA-256 values.

- Native checkpoint, ATOM load-time PTPC. Original `75e224dfd754e69d87496ed54c7e62c3d9feab76c724a68c355f6b818d4976d0`. Optimized `122e3db8a98c4173a3aca9347ebc49855471645db7902051f0a5ff9383e09e6d`.
- Native checkpoint → Bobo FP4PTPC on ATOM. Original `cee437a2a4404954370b787b506158cfc7134d1bab75d5a25de2c4fcb3752129`. Optimized `1a5b24f346a30738882af1ba322d4d337b6c4a920d1cfec93c5155e8e85dd654`.
