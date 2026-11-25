# Luocheng dense A8W4 on DeepSeek-V4.1-Flash-1017-FP4PTPC

## What

DeepSeek's TP4 dense projections use FP8 weights. Luocheng's `flydsl_gemm_a8w4_ptpc` requires packed E2M1 weights and one E8M0 scale per 32 K elements. The kernel quantizes BF16 activations per token to E4M3FNUZ, decodes the packed weights to FP8 in registers, and writes BF16 from FP32 accumulators. Both the original M0 and current M9 have the same weight format requirement. [M0](https://github.com/luocheng25/aiter/blob/ba9dd6a3493bb2b6313e5c52664300f8f9f75438/aiter/ops/flydsl/gemm_a8w4_ptpc.py#L124) · [M9](https://github.com/luocheng25/aiter/blob/4f98da2531efc78a51583900a249536626a76eaa/aiter/ops/flydsl/gemm_a8w4_ptpc.py#L257)

## Data

Checkpoint `ginsongsong/DeepSeek-V4.1-Flash-1017-FP4PTPC@97cc9b32` on PTL52. Shapes are per TP rank for `Y[M,N] = X[M,K] × W[N,K]ᵀ`, with M equal to the scheduled token rows. The backbone has 40 layers, hidden size 5,120, 64 attention heads of width 512, and Q/O LoRA ranks 1,280/1,024. [Checkpoint config](https://huggingface.co/ginsongsong/DeepSeek-V4.1-Flash-1017-FP4PTPC/blob/97cc9b32c970864edce0f3fc186d3042b04e3a36/config.json) · [ATOM projection setup](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/atom/models/deepseek_v41/attention.py#L285)

| TP4 call | Instances in backbone | Local N | Local K | Weight and scale |
|---|---:|---:|---:|---|
| `attn.wqkv_a` (`wq_a` + `wkv`, replicated) | 40 | 1,792 | 5,120 | E4M3, FP32 per output channel |
| `attn.wq_b` (column parallel) | 40 | 8,192 | 1,280 | E4M3, FP32 per output channel |
| `attn.wo_a` (two groups per rank) | 40 | 1,024 per group | 4,096 | E4M3, E8M0 per 32×32 block |
| `attn.wo_b` (row parallel) | 40 | 5,120 | 2,048 | E4M3, FP32 per output channel |
| `attn.indexer.wq_b` (replicated) | 8 | 4,096 | 1,280 | E4M3, FP32 per output channel |
| `engram.wkv` (replicated at layers 1 and 14) | 2 | 25,600 | 6,144 | E4M3, FP32 per output channel |

The shared expert also stores E4M3 with FP32 channel scales. Its local gate and up projections each have `N=576, K=5120`, and down has `N=5120, K=576`. ATOM's FP8 PTPC MoE fuses the shared slot with the 384 routed experts after load-time PTPC quantization of the routed matrices. The actual MXFP4 weights available in the checkpoint for runtime unpacking are the routed-expert matrices. [Engram projection](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/atom/model_ops/engram/device/layer.py#L41) · [Shared-expert dispatch](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/atom/models/deepseek_v41/moe.py#L19)

## Benchmarking and Profiling

ATOM's FP8 PTPC linear dispatch uses AITER `gemm_a8w8_bpreshuffle` on the standard path. The attention and indexer `wq_b` calls can reuse the same FP8 activation and FP32 token scale from `DualRMSNormPerToken`. Luocheng's public entry takes BF16 and quantizes it within the call. A dense A/B on this checkpoint stops at the weight format check. An A8W4 dense run would require requantizing checkpoint weights to MXFP4 and validating the resulting model. [ATOM FP8 dispatch](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/atom/model_ops/linear.py#L1663) · [ATOM fused norm](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/atom/models/deepseek_v41/attention.py#L299) · [Luocheng entry](https://github.com/luocheng25/aiter/blob/4f98da2531efc78a51583900a249536626a76eaa/aiter/ops/flydsl/gemm_a8w4_ptpc.py#L311)

## Validation

Read the safetensors headers of all 48 local shards. The 96,085 tensor names match the checkpoint index. The backbone has 290 FP32 channel-scale tensors across the attention projections, Engram projections, and shared experts. The dense result is a compatibility check with GPU measurements pending a compatible dense weight format. [Checkpoint index](https://huggingface.co/ginsongsong/DeepSeek-V4.1-Flash-1017-FP4PTPC/blob/97cc9b32c970864edce0f3fc186d3042b04e3a36/model.safetensors.index.json)
