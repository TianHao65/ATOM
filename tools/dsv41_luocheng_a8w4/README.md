# DeepSeek-V4.1 FP4PTPC runtime A8W4

ATOM uses Luocheng's PyHIP M13 kernels for routed experts when
`ATOM_DSV41_LUOCHENG_MOE=a8w4`. This experiment targets MI308X TP4 with
DeepSeek-V4.1-Flash-1017-FP4PTPC revision `97cc9b32`. The routed weights
remain packed E2M1 with E8M0 scales until the forward pass. FP8 PTPC
shared experts run through the existing standalone shared-expert path.

The routed call uses H=5120, I=576 per rank, 384 experts, top-k=6,
per-token E4M3FNUZ activations and SwiGLU clamp 10.0. The parent ATOM
configuration with online routed-expert quantization uses 385 experts
and top-k=7 after fusing the FP8 PTPC shared expert.

## Dependencies

Apply the patches to experiment checkouts of AITER
`ad0d5e208eb2d49ab658c3c161aa5c3502dfc2a1` and PyHIP
`a641da286511f3f8b703f90b3f9a1ea66b18301a`. The serving image uses
PyTorch 2.10.0, ROCm 7.2.4 and FlyDSL 0.5.0.

```bash
git -C "$DSV41_AITER_SRC" apply --unidiff-zero \
    "$DSV41_ATOM_SRC/tools/dsv41_luocheng_a8w4/aiter-runtime-a8w4.patch"
git -C "$DSV41_AITER_SRC" apply --unidiff-zero \
    "$DSV41_ATOM_SRC/tools/dsv41_luocheng_a8w4/aiter-fused-shared.patch"
git -C "$DSV41_PYHIP_SRC" apply --unidiff-zero \
    "$DSV41_ATOM_SRC/tools/dsv41_luocheng_a8w4/pyhip-dsv41-moe.patch"
git -C "$DSV41_PYHIP_SRC" apply --unidiff-zero \
    "$DSV41_ATOM_SRC/tools/dsv41_luocheng_a8w4/pyhip-fused-shared.patch"
export PYTHONPATH="$DSV41_ATOM_SRC:$DSV41_AITER_SRC:$DSV41_PYHIP_SRC/src"
```

| Serving arm | `ATOM_DSV41_LUOCHENG_MOE` | `--online_quant_config` |
|---|---|---|
| Original ATOM | `off` | `{"layer_quant_config":{"*.ffn.experts*":"ptpc_fp8"}}` |
| Matched routed shape | `ptpc_split` | `{"layer_quant_config":{"*.ffn.experts*":"ptpc_fp8"}}` |
| Luocheng A8W4 | `a8w4` | Omit argument |
| Luocheng A8W4 with mixed-format Decode | `a8w4_fused` | Omit argument |

The ATOM adapter pre-packs weights during model load and calls
`aiter.fused_moe` with the prepared weights. The supplied PyHIP patch
extends decode to H=5120, applies the model's gate and up clamp before
SiLU multiplication, and supports masked decode routes in graph replay.
In the split arms, the shared expert's FP8 down GEMM pads K from 576 to
640 with zeros to support the gfx942 preshuffled kernel. The logical
intermediate width and channel scales stay at their checkpoint values.

## Mixed-format Decode

`a8w4_fused` uses the original ATOM TP4 routing layout with 385 expert
slots and a seventh route for shared expert ID 384. The shared tensors
keep their FP8 values and FP32 channel scales. Gate/up and down select
MXFP4 unpack or direct FP8 load by expert ID. Down folds all seven
outputs into the same reduction. This path fuses M=1 in two launches.
M>1 uses the six routed MXFP4 experts and the FP8 shared calculation
under the same `aiter.fused_moe` entry point.

In a graph replay probe on a single MI308X with the checkpoint's layer-20
TP4 rank-0 weights, the final M1 build took 31.17 µs against 48.50 µs for
the split A8W4 arm and 28.91 µs for the original fused FP8 arm. The probe
uses H5120, I576, shared slot 384 and SwiGLU clamp 10.0. It times
post-routing compute on one rank. A graph replay with six masked routed
IDs and a prefilled shared ID returned zero. The M2 and M64 fallbacks
matched an explicit split call bitwise.
