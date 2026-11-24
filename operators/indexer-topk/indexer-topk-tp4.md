# Stable indexer top-k on MI308X

## What

The DeepSeek-V4.1 ATOM indexer selects 512 positions from each FP32 score row. The original dispatch uses the AITER FlyDSL selector for the 16,384-column candidate rows and HIP radix selection for the longer rows. Silo's Ballot kernel uses wave ballots and popcount for radix counting. The ATOM adapter selects an unsplit or split launch from the physical row width. [ATOM selector call](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/atom/model_ops/indexer/score.py#L231) · [Silo AITER source](https://github.com/ROCm/aiter/commit/7f36447ac7a235d15d8de7b4598c2c167d85da67) · [Serving Ballot adapter](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/atom/model_ops/deepseek_v41/ballot_topk.py#L31)

## Data

One MI308X (gfx942) on PTL52, PyTorch 2.10.0 with ROCm 7.2.4, and AITER `ad0d5e2` from the original ATOM benchmark image `cfd987217fc4`. Inputs use the physical row widths and baseline dispatch from the DeepSeek-V4.1-Flash-1017 TP4 Base C1 100K trace. The isolated operator uses batch 1, one query token per request, and `stable=True`. Both arms receive the same FP32 logits from the same synthetic FP8 paged scorer run with 8-token pages. The live column counts use the 100K input boundary.

Bobo's release notes identify SGLang `baa9befa`, AITER `58b5015d` and ATOM `988052da` (`feat/dsv41-pr3518-port`). The listed `*_atom_tp4.sh` launchers establish an ATOM serving configuration. The pinned SGLang source has its own paged top-k v2 dispatch behind `use_topk_v2`. The Bobo SGL cells are blank pending verified dispatch and matched operator timing for the delivery image. [SGLang paged top-k dispatch](https://github.com/sgl-project/sglang/blob/baa9befa381511cd4d56f83980951e5a45fffc6d/python/sglang/srt/layers/attention/dsv4/indexer.py#L446) · [Delivery measurements](../../releases/deepseek-v41-1017-tp4-20261119.md)

## Benchmarking and Profiling

Each arm was warmed and captured as one selector call per graph. Reported latency is the median of eight GPU-event means over 200 replays per arm, with arm order reversed on alternating rounds. The serving Ballot adapter includes its split scratch zeroing in the timing. The aggregate weights each isolated result by its observed number of calls per Decode step.

| Physical columns | Live columns | Calls per step | ATOM original (µs) | SGL Bobo (µs) | AITER Silo (µs) | Silo speedup vs ATOM |
|---:|---:|---:|---:|---:|---:|---:|
| 16,384 | 16,384 | 4 | 26.77 | | 16.31 | 1.64× |
| 51,200 | 50,000 | 3 | 46.54 | | 22.90 | 2.03× |
| 102,400 | 100,000 | 1 | 78.47 | | 27.24 | 2.88× |
| **Frequency-weighted sum** | | **8** | **325.15** | | **161.20** | **2.02×** |

The isolated weighted sum falls by **50.42%**. The separate full-model profile measured rank-0 Decode K512 GPU work at 387.89 → 238.75 µs per step (−38.45%). The trace sums the K512 kernel durations inside the serving graph. [Per-round isolated results](indexer-topk-tp4.json) · [Serving profile and original traces](https://github.com/TianHao65/ATOM/blob/3712bf1cde8ffdb7ac7ec2f1ed95d1928d15c6dd/README.md#benchmarking-and-profiling)

## Validation

- Both arms returned the exact ascending stable index sequence on all three shapes with FP8 indexer, repeated-score and all-zero inputs. The reference breaks score ties by the lowest index. High scores outside the live row confirmed that padding is excluded.
- The serving A/B independently matched all 39,904 same-logits index comparisons. Across separate starts, end-to-end SSE hashes matched 3/8 timed requests. [Serving validation](https://github.com/TianHao65/ATOM/blob/3712bf1cde8ffdb7ac7ec2f1ed95d1928d15c6dd/README.md#validation)
- Ownership sampling found zero foreign use during the isolated timing. GPU utilization was 0% and VRAM was 297,611,264 bytes both immediately before launch and after cleanup. The isolated container was removed.
