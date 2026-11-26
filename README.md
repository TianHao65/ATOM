# MI308X operator reports

| Report | Tested workloads |
|---|---|
| [Indexer top-k](operators/indexer-topk/indexer-topk-tp4.md) | DeepSeek-V4.1-Flash-1017 TP4 Base, stable K=512, ATOM original and Silo Ballot |
| [Silo Ballot end-to-end latency breakdown](operators/indexer-topk/silo-ballot-e2e-breakdown.md) | 100K/150 serving latency and paired 100K/24 Prefill and Decode profiles, TP4 C1 |
| [Luocheng runtime A8W4 MoE](operators/fused-moe/luocheng-a8w4-moe.md) | DeepSeek-V4.1-Flash-1017-FP4PTPC TP4 |
| [Luocheng dense A8W4 GEMM](operators/dense-gemm/dense-gemm-a8w4.md) | DeepSeek-V4.1-Flash-1017-FP4PTPC TP4 weight formats and projection shapes |
