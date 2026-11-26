# MI308X operator reports

| Report | Tested workloads |
|---|---|
| [Indexer top-k](operators/indexer-topk/indexer-topk-tp4.md) | DeepSeek-V4.1-Flash-1017 TP4 Base, stable K=512, ATOM baseline and AITER Ballot |
| [DeepSeek-V4.1-Flash model latency breakdown](operators/indexer-topk/indexer-topk-e2e-breakdown.md) | TP4 C1, 100K/150 serving latency and 100K/24 whole-model module shares, ATOM baseline and AITER Ballot |
| [Luocheng runtime A8W4 MoE](operators/fused-moe/luocheng-a8w4-moe.md) | DeepSeek-V4.1-Flash-1017-FP4PTPC TP4 |
| [Luocheng dense A8W4 GEMM](operators/dense-gemm/dense-gemm-a8w4.md) | DeepSeek-V4.1-Flash-1017-FP4PTPC TP4 weight formats and projection shapes |
