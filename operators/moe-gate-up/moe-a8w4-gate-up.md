# Luocheng runtime A8W4 MoE gate/up

## What

M11 changed the streaming gate/up to one paired gate/up block per wave. M12 read paired weight rows with a 16-byte load, prefetched weights two tiles ahead in prefill, and set a fixed four-wave occupancy for the BN128/TK128 prefill tile. Both paths decode the packed MXFP4 weights in registers before FP8 MFMA. [M11 streaming changes](https://github.com/luocheng25/pyhip/commit/492d90e4f1b69b25b8ae19cf67b2d96b3a463ad0) · [M12 prefill changes](https://github.com/luocheng25/pyhip/commit/9681cb98cf28de854e59d2dcf6aab3a332115ea6)

## Data

These are MI308X stage timings from the author's standalone A8W4 MoE development runs. For the Qwen3.8 TP2 sample, H=2,560, local I=320, E=513, executed top-k=11. The MiMo streaming sample used H=4,096, E=256 and top-k=8. Its short timing note omits the TP and local I.

## Benchmarking and Profiling

| Phase and M | Baseline | Gate/up GPU time (µs) | Scope |
|---|---|---:|---|
| Qwen3.8 TP2, M=8,192 prefill | R2 before M12 versus R1 | 1,110 versus 994 | Original stage breakdown used to target M12 |
| MiMo, M=512 streaming | R2 before and after M11 | 191 → 153 (−19.9%) | Single gate/up kernel |

The M12 author reports 8–19% lower prefill gate/up time after the paired loads and prefetch change. The published commit does not list a per-shape timing matrix for this stage. End-to-end operator measurements appear in the separate MoE reports. [Stage breakdowns and measurement scope](https://github.com/luocheng25/pyhip/commit/9681cb98cf28de854e59d2dcf6aab3a332115ea6)

## Validation

- The M12 notes report bitwise-equal gate/up outputs in the author's same-process comparison. The benchmark harness also checked whole-operator results. [A8W4 tests](https://github.com/luocheng25/pyhip/blob/a641da286511f3f8b703f90b3f9a1ea66b18301a/tests/ops/moe/test_moe_a8w4.py)
- The current ATOM DeepSeek TP4 gate/up needs H=5,120, 385 fused expert slots and top-k=7 with its checkpoint clamp. The historical author measurements do not cover this dimension set. [DeepSeek config](https://github.com/TianHao65/ATOM/blob/e309e005dfa83dd323dfb959b040d0af8bb6bceb/tests/models/deepseek_v41/fixtures/config.json#L52)
