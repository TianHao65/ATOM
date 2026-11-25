#!/usr/bin/env python3
"""Compare fused and split shared-expert C1 paths on DeepSeek TP4 rank 0."""

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from aiter import ActivationType, QuantType, gemm_a8w8_bpreshuffle, get_hip_quant
from aiter import silu_and_mul
from aiter.fused_moe import fused_moe, prepare_runtime_a8w4_shared_weights
from aiter.ops.shuffle import shuffle_weight
from pyhip.ops.moe.flydsl import moe_a8w4
from safetensors import safe_open

from bench_routed_moe import (
    EXPERTS,
    FP8,
    HIDDEN,
    INTER,
    LIMIT,
    LOADER,
    TOPK,
    emit,
    errors,
    graph_timings,
    load_checkpoint,
    make_inputs,
    prepare_ptpc,
    reference,
)

LAYER = 20
RANK = 0
PTPC_PADDED_INTER = 640
CHECKPOINT_REVISION = "97cc9b32c970864edce0f3fc186d3042b04e3a36"


def load_shared(model):
    """Load exact TP-local FP8 weights and normalize E4M3FN to gfx942 FNUZ."""
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    prefix = f"layers.{LAYER}.ffn.shared_experts."
    tp_slice = slice(RANK * INTER, (RANK + 1) * INTER)
    projections = {}
    for name in ("w1", "w3", "w2"):
        weight_key = prefix + name + ".weight"
        scale_key = prefix + name + ".weight_scale"
        with safe_open(model / index[weight_key], framework="pt", device="cpu") as shard:
            weight = shard.get_tensor(weight_key)
        with safe_open(model / index[scale_key], framework="pt", device="cpu") as shard:
            scale = shard.get_tensor(scale_key)
        outer = HIDDEN if name == "w2" else INTER * 4
        inner = INTER * 4 if name == "w2" else HIDDEN
        if (
            weight.dtype != torch.float8_e4m3fn
            or weight.shape != (outer, inner)
            or scale.dtype != torch.float32
            or scale.shape != (outer, 1)
        ):
            raise ValueError(f"Unexpected checkpoint shape or dtype in {weight_key}")
        if name == "w2":
            weight = weight[:, tp_slice]
        else:
            weight = weight[tp_slice]
            scale = scale[tp_slice]
        bits = weight.contiguous().view(torch.uint8).cuda()
        # E4M3FN negative zero is a NaN byte in FNUZ. For all nonzero
        # finite values, FNUZ decodes to half the corresponding FN value.
        bits.masked_fill_(bits == 0x80, 0)
        projections[name] = (bits, scale.cuda().mul_(2).reshape(-1).contiguous())
    gate, gate_scales = projections["w1"]
    up, up_scales = projections["w3"]
    down, down_scales = projections["w2"]
    return (
        torch.cat((gate, up)).view(FP8),
        torch.cat((gate_scales, up_scales)),
        down.view(FP8),
        down_scales,
    )


def append_shared_weight(routed, routed_scales, shared, shared_scales):
    packed = shuffle_weight(shared, layout=(16, 16))
    weights = torch.cat((routed.view(torch.uint8), packed.view(torch.uint8)[None]))
    weights = weights.view(FP8)
    weights.is_shuffled = True
    return weights, torch.cat((routed_scales, shared_scales[None]))


def add_shared_route(weights, ids):
    shared_ids = ids.new_full((ids.shape[0], 1), EXPERTS)
    shared_weights = weights.new_ones((weights.shape[0], 1))
    return torch.cat((weights, shared_weights), dim=1), torch.cat((ids, shared_ids), dim=1)


def dequantized_row_fp8(x):
    x = x.float()
    scale = x.abs().amax(dim=-1, keepdim=True) / 240.0
    scale = torch.where(scale == 0, 1.0, scale)
    return (x / scale).clamp(-240.0, 240.0).to(FP8).float() * scale


def shared_reference(x, shared, quantize_activations):
    w13, s13, w2, s2 = shared
    x = dequantized_row_fp8(x) if quantize_activations else x.float()
    gate, up = (x @ (w13.float() * s13[:, None]).T).chunk(2, dim=-1)
    inter = F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
    if quantize_activations:
        inter = dequantized_row_fp8(inter)
    return inter @ (w2.float() * s2[:, None]).T


def closed_graph_check(call):
    eager = call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay = call()
    graph.replay()
    torch.cuda.synchronize()
    return torch.equal(replay, eager)


def make_split_projection(shared):
    w13, s13, w2, s2 = shared
    w13 = shuffle_weight(w13, layout=(16, 16))
    down_bits = F.pad(w2.view(torch.uint8), (0, PTPC_PADDED_INTER - INTER))
    w2 = shuffle_weight(down_bits.view(FP8), layout=(16, 16))
    quantize = get_hip_quant(QuantType.per_Token)
    s13, s2 = s13[:, None], s2[:, None]

    def run_shared(x):
        x_fp8, x_scale = quantize(x, quant_dtype=FP8)
        gate_up = gemm_a8w8_bpreshuffle(x_fp8, w13, x_scale, s13, dtype=torch.bfloat16)
        inter = x.new_empty((x.shape[0], INTER))
        silu_and_mul(inter, gate_up, LIMIT)
        inter_fp8, inter_scale = quantize(inter, quant_dtype=FP8)
        return gemm_a8w8_bpreshuffle(inter_fp8, w2, inter_scale, s2, dtype=torch.bfloat16)

    return run_shared


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--cases", type=int, default=4)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--groups", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42117)
    parser.add_argument("--fallback-rows", type=int, nargs="*", default=[2, 64])
    args = parser.parse_args()
    if min(args.cases, args.repeat, args.groups) < 1 or any(v < 2 for v in args.fallback_rows):
        parser.error("Cases, repeats and groups must be positive, and fallback needs at least two rows")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one verified idle MI308X GPU")
    props = torch.cuda.get_device_properties(0)
    if not props.gcnArchName.startswith("gfx942") or props.multi_processor_count != 80:
        raise RuntimeError(f"Expected 80-CU gfx942, got {props.gcnArchName}/{props.multi_processor_count}")
    torch.backends.cuda.matmul.allow_tf32 = False
    emit(
        "setup", checkpoint=CHECKPOINT_REVISION, layer=LAYER, tp=4, rank=RANK,
        h=HIDDEN, i=INTER, e=EXPERTS + 1, topk=TOPK + 1, shared_id=EXPERTS,
        shared_down_k_split=PTPC_PADDED_INTER, clamp=LIMIT, arch=props.gcnArchName,
        cu=props.multi_processor_count, cases=args.cases, repeat=args.repeat,
        groups=args.groups, seed=args.seed, fallback_rows=args.fallback_rows,
        scope="Rank-local MoE after Top-k, excluding routing and TP all-reduce.",
    )

    native, router = load_checkpoint(args.model, LOADER, LAYER, RANK)
    shared = load_shared(args.model)
    shared_w13, shared_s13, shared_w2, shared_s2 = shared
    w13, s13, w2, s2 = native
    routed = moe_a8w4.prepare_a8w4_weights(w13, s13, w2, s2)
    shared_prepared = prepare_runtime_a8w4_shared_weights(*shared)
    ptpc_w13, ptpc_s13 = append_shared_weight(*prepare_ptpc(w13, s13), shared_w13, shared_s13)
    ptpc_w2, ptpc_s2 = append_shared_weight(*prepare_ptpc(w2, s2), shared_w2, shared_s2)
    emit("weights", routed_mxfp4=[list(t.shape) for t in native],
         shared_fp8=[list(t.shape) for t in shared], padded_inter=routed.inter_padded)

    def original(x, weights, ids):
        return fused_moe(
            x, ptpc_w13, ptpc_w2, weights, ids,
            activation=ActivationType.Silu, quant_type=QuantType.per_Token,
            w1_scale=ptpc_s13, w2_scale=ptpc_s2, swiglu_limit=LIMIT,
        )

    def routed_call(x, weights, ids):
        return fused_moe(
            x, None, None, weights, ids,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            runtime_a8w4_weights=routed, swiglu_limit=LIMIT,
        )

    def mixed(x, weights, ids):
        return fused_moe(
            x, None, None, weights, ids,
            activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
            runtime_a8w4_weights=routed, runtime_a8w4_shared_weights=shared_prepared,
            swiglu_limit=LIMIT,
        )

    shared_call = make_split_projection(shared)
    shared_stream = torch.cuda.Stream()

    def split(x, weights, ids):
        main = torch.cuda.current_stream()
        shared_stream.wait_stream(main)
        routed_out = routed_call(x, weights, ids)
        with torch.cuda.stream(shared_stream):
            shared_out = shared_call(x)
        main.wait_stream(shared_stream)
        return routed_out + shared_out

    all_timings = {}
    for case in range(args.cases):
        seed = args.seed + case
        x, weights, ids = make_inputs(1, router, seed)
        weights7, ids7 = add_shared_route(weights, ids)
        ref_routed, _ = reference(x, weights, ids, native)
        ref_shared = shared_reference(x, shared, quantize_activations=False)
        ref_shared_ptpc = shared_reference(x, shared, quantize_activations=True)
        ref_total = ref_routed + ref_shared
        shared_only_gates = weights7.clone()
        shared_only_gates[:, :-1] = 0
        masked_shared_ids = ids7.clone()
        masked_shared_ids[:, -1] = -1
        masked_row_ids = ids7.clone()
        masked_row_ids[:, :-1] = -1

        arms = {
            "atom_fused_fp8": lambda: original(x, weights7, ids7),
            "luocheng_split_shared": lambda: split(x, weights, ids),
            "luocheng_fused_mixed": lambda: mixed(x, weights7, ids7),
        }
        orig, split_out, mixed_out = [fn() for fn in arms.values()]
        mixed_path = list(moe_a8w4.last_path)
        mixed_shared = mixed(x, shared_only_gates, ids7)
        masked = mixed(x, weights7, masked_shared_ids)
        route_only = routed_call(x, weights, ids)
        def masked_row():
            return mixed(x, weights7, masked_row_ids)
        masked_row_zero = int(masked_row().count_nonzero()) == 0
        masked_row_replay = closed_graph_check(masked_row)
        checks = {
            "shared_vs_exact_fp8_weights": errors(mixed_shared, ref_shared),
            "shared_vs_fp8_quantized_activation_reference": errors(mixed_shared, ref_shared_ptpc),
            "masked_shared_vs_routed_only": errors(masked, route_only.float()),
            "mixed_vs_fp32": errors(mixed_out, ref_total),
            "original_vs_fp32": errors(orig, ref_total),
            "split_vs_fp32": errors(split_out, ref_total),
            "mixed_vs_original": errors(mixed_out, orig.float()),
            "mixed_vs_split": errors(mixed_out, split_out.float()),
        }
        if not all(result["finite"] for result in checks.values()):
            raise RuntimeError("Nonfinite output in C1 reference check")
        # The validation allows normal FP8 tie rounding differences between
        # the torch reference and gfx942 instructions.
        passed = (
            checks["shared_vs_fp8_quantized_activation_reference"]["normalized_rmse"] <= 0.01
            and checks["masked_shared_vs_routed_only"]["normalized_rmse"] <= 0.005
            and masked_row_zero
            and masked_row_replay
        )
        exact_graph_replay = closed_graph_check(arms["luocheng_fused_mixed"])
        emit("validation", seed=seed, ids=ids7.tolist(), gates=weights7.tolist(), path=mixed_path,
             passed=passed, exact_graph_replay=exact_graph_replay, masked_row_zero=masked_row_zero,
             masked_row_graph_replay=masked_row_replay, **checks)
        if not passed or not exact_graph_replay:
            raise RuntimeError(f"Mixed decode validation failed at seed {seed}")
        started = time.time()
        timings = graph_timings(arms, args.repeat, args.groups)
        emit("timing", seed=seed, started_unix=started, finished_unix=time.time(), **timings)
        for name, result in timings.items():
            all_timings.setdefault(name, []).extend(result["window_us"])

    for rows in args.fallback_rows:
        x, weights, ids = make_inputs(rows, router, args.seed)
        weights7, ids7 = add_shared_route(weights, ids)
        expected = routed_call(x, weights, ids) + shared_call(x)
        def fallback():
            return mixed(x, weights7, ids7)
        actual = fallback()
        metrics = errors(actual, expected.float())
        parity = closed_graph_check(fallback)
        passed = metrics["finite"] and metrics["normalized_rmse"] <= 0.005 and parity
        emit("fallback", m=rows, path=list(moe_a8w4.last_path), passed=passed,
             exact_graph_replay=parity, error=metrics)
        if not passed:
            raise RuntimeError(f"Shared fallback validation failed at M={rows}")

    emit("summary", median_us={name: statistics.median(values) for name, values in all_timings.items()})


if __name__ == "__main__":
    main()
