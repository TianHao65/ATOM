# Luocheng runtime A8W4 MoE down projection

## What

M12 split the prefill down projection across two or four workgroups when the existing sort tiles offered too little parallel work. The wide, non-atomic BM64 streaming path can use the prefill down kernel and route reduction. [M12 change and notes](https://github.com/luocheng25/pyhip/commit/9681cb98cf28de854e59d2dcf6aab3a332115ea6)

## Data

MI308X, MiMo-V2.6-Flash shape, H=4,096, global I=2,048, E=256, top-k=8, SiLU. The TP8 shape has local I=256. The TP4 shape has local I=512. The source reports development-time stage measurements within the PyHIP operator.

## Benchmarking and Profiling

| Parallel shape | Token count | Change | Down GPU time (µs) |
|---|---:|---|---:|
| TP8 | 2,048 | Split the hidden dimension across workgroups | 232 → 207 (−10.8%) |
| TP4 | 1,024 | Dispatch BM64 non-atomic streaming down through prefill down | 303 → 289 (−4.6%) |

The author retained the prefill route output and sorted sum after testing a slower prefill atomic variant. For streaming at I=512, the atomic epilogue added about 4 µs due to register pressure. [M11 streaming result](https://github.com/luocheng25/pyhip/commit/492d90e4f1b69b25b8ae19cf67b2d96b3a463ad0)

## Validation

- The author validated these variants against the A8W4 FP32 reference using a 4.5% maximum-error bound and `calc_diff ≤ 1e-3`. Tests exercise split factors 1, 2 and 4. [Kernel tests](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/tests/ops/moe/test_moe_a8w4.py#L845)
- The measured MiMo shapes use fewer expert slots and a different top-k than DeepSeek. The ATOM DeepSeek down comparison needs the same H, local/padded I, 385 slots, and top-k=7 in both arms. [DeepSeek config](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/tests/models/deepseek_v41/fixtures/config.json#L52)
