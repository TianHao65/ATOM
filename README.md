# DeepSeek-V4.1 Flash operator reports

## MI308X TP4 Base

| Operator | Report |
|---|---|
| Attention indexer top-k | [ATOM original and Silo Ballot](operators/indexer-topk/indexer-topk-tp4.md) |
| Fused MoE | [Exact-shape AITER tuning, kernel selection and CSV](operators/fused-moe/fused-moe-tp4.md) |
| Q/KV input projection | [`wqkv_a` BF16 and FP8 PTPC](operators/qkv-a-projection/wqkv_a-gfx942-tp4.md) |
| Attention Q projection | [`wq_b` BF16 and FP8 PTPC](operators/q-b-projection/wq_b-gfx942-tp4.md) |
| Attention output projection | [`wo_b` BF16 and FP8 PTPC](operators/o-b-projection/wo_b-gfx942-tp4.md) |

The standalone MoE uses the ATOM call's rank-local `H=5120, I=576, experts=385, top-k=7`. The shared expert occupies slot 384 after the 384 routed experts. The packaged AITER CSV includes the two rows that passed revalidation through the runtime loader. [Tuned rows](operators/fused-moe/dsv41_mi308_tp4_ptpc_tuned_fmoe.csv)

## Luocheng historical operator tests

| Operator | Model or parallel configuration | Report |
|---|---|---|
| Runtime A8W4 MoE | Qwen3.8-Flash-Next TP2 | [Original PyHIP, M13 and tuned AITER FP8](operators/fused-moe/moe-a8w4-qwen38-tp2.md) |
| Runtime A8W4 MoE | Qwen3.8-Flash-Next TP4 | [Original PyHIP, M13 and tuned AITER FP8](operators/fused-moe/moe-a8w4-qwen38-tp4.md) |
| Runtime A8W4 MoE | MiMo-V2.6-Flash TP4 | [Original PyHIP and M13](operators/fused-moe/moe-a8w4-mimo-tp4.md) |
| Runtime A8W4 MoE | MiMo-V2.6-Flash TP8 | [Original PyHIP and M13](operators/fused-moe/moe-a8w4-mimo-tp8.md) |
| MoE gate/up | Published profiling shapes | [Gate/up variants](operators/moe-gate-up/moe-a8w4-gate-up.md) |
| MoE down | Published profiling shapes | [Down variants](operators/moe-down/moe-a8w4-down.md) |
| MoE row quantization | Published profiling shapes | [Activation quantization variants](operators/moe-row-quantization/moe-a8w4-row-quant.md) |
| MoE routing prep | Published TP2/TP4/TP8 shapes | [Sorting and input quantization](operators/moe-routing-prep/moe-a8w4-route-prep.md) |
| Dense A8W4 GEMM | Qwen3.8-Flash-Next TP2/TP4 | [M0 and AITER FP8 PTPC](operators/dense-gemm/dense-gemm-a8w4-qwen38.md) |

## End-to-end references

- The Silo A/B used 100K input, 150 output and 89,984 cached tokens per request. Its profile uses 100K/24. [End-to-end report and original Perfetto evidence](https://github.com/TianHao65/ATOM/blob/3712bf1cde8ffdb7ac7ec2f1ed95d1928d15c6dd/README.md)
- The delivery notes report 100K/1,500 cache-off results with and without DSpark. [Bobo delivery results and revisions](releases/deepseek-v41-1017-tp4-20261119.md)
