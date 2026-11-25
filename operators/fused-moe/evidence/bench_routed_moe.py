#!/usr/bin/env python3
"""Compare rank-local FP8 PTPC and Luocheng A8W4 on real DeepSeek experts."""

import argparse
import json
import os
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from aiter import ActivationType, QuantType, get_hip_quant
from aiter.fused_moe import fused_moe
from aiter.ops.shuffle import shuffle_weight
from pyhip.ops.moe.flydsl import moe_a8w4
from pyhip.testing.moe_a8w4 import dequant_mxfp4
from safetensors import safe_open


EXPERTS, TOPK, HIDDEN, INTER = 384, 6, 5120, 576
LIMIT, ROUTED_SCALE = 10.0, 1.5
FP8 = torch.float8_e4m3fnuz
LOADER = Path(__file__).resolve().parent


def emit(phase, **data):
    print(json.dumps(dict(phase=phase, **data), allow_nan=False), flush=True)


def load_checkpoint(model, loader_dir, layer, rank):
    sys.path.insert(0, str(loader_dir))
    from load_tp4_layer import load_tp4_layer

    (w13, s13), (w2, s2) = load_tp4_layer(model, layer=layer, rank=rank)
    stored_inter = w13.shape[1] // 2
    native = (
        torch.cat((w13[:, :INTER], w13[:, stored_inter : stored_inter + INTER]), dim=1),
        torch.cat((s13[:, :INTER], s13[:, stored_inter : stored_inter + INTER]), dim=1),
        w2[..., : INTER // 2].contiguous(),
        s2[..., : INTER // 32].contiguous(),
    )
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    gate_prefix = f"layers.{layer}.ffn.gate"
    router = []
    for key in (f"{gate_prefix}.weight", f"{gate_prefix}.bias"):
        with safe_open(model / index[key], framework="pt", device="cpu") as shard:
            router.append(shard.get_tensor(key).cuda().float())
    return native, router


def prepare_ptpc(packed, codes):
    experts, rows, packed_cols = packed.shape
    cols = packed_cols * 2
    weight = torch.empty((experts, rows, cols), dtype=FP8, device=packed.device)
    scale = torch.empty((experts, rows), dtype=torch.float32, device=packed.device)
    quantize = get_hip_quant(QuantType.per_Token)
    for start in range(0, experts, 8):
        end = min(start + 8, experts)
        dense = dequant_mxfp4(packed[start:end], codes[start:end]).to(torch.bfloat16)
        q, s = quantize(dense.reshape(-1, cols), quant_dtype=FP8)
        shuffled = shuffle_weight(q.view(end - start, rows, cols), layout=(16, 16))
        weight[start:end].view(torch.uint8).copy_(shuffled.view(torch.uint8))
        scale[start:end].copy_(s.reshape(end - start, rows))
    weight.is_shuffled = True
    return weight, scale


def make_inputs(tokens, router, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((tokens, HIDDEN), dtype=torch.bfloat16, device="cuda", generator=generator)
    gate_weight, gate_bias = router
    scores = F.softplus(x.float() @ gate_weight.T).sqrt()
    ids = (scores + gate_bias).topk(TOPK, dim=-1).indices
    weights = scores.gather(1, ids)
    weights = (weights / weights.sum(dim=-1, keepdim=True)) * ROUTED_SCALE
    return x, weights.contiguous(), ids.to(dtype=torch.int32).contiguous()


def reference(x, route_weights, ids, native):
    """FP32 GEMMs and combine on BF16 inputs and losslessly unpacked MXFP4."""
    w13, s13, w2, s2 = native
    x = x.float()
    flat_ids = ids.flatten().long()
    order = flat_ids.argsort(stable=True)
    tokens, slots = order // TOPK, order % TOPK
    counts = flat_ids.bincount(minlength=EXPERTS).tolist()
    output = torch.zeros_like(x)
    start = 0
    for expert, count in enumerate(counts):
        if not count:
            continue
        end = start + count
        t, s = tokens[start:end], slots[start:end]
        gateup = x[t] @ dequant_mxfp4(w13[expert], s13[expert]).T
        gate, up = gateup.chunk(2, dim=-1)
        activation = F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
        down = activation @ dequant_mxfp4(w2[expert], s2[expert]).T
        output.index_add_(0, t, down * route_weights[t, s, None])
        start = end
    return output, counts


def errors(actual, expected):
    actual = actual.float()
    if not bool(torch.isfinite(actual).all()):
        return {"finite": False}
    delta = actual - expected
    return {
        "finite": True,
        "normalized_rmse": (delta.norm() / expected.norm().clamp_min(1e-12)).item(),
        "max_abs": delta.abs().amax().item(),
        "max_abs_reference": expected.abs().amax().item(),
        "cosine": F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
    }


def graph_timings(functions, repeat, groups):
    graphs = []
    for name, call in functions.items():
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = call()
        graphs.append((name, graph, output))
        for _ in range(3):
            graph.replay()
    torch.cuda.synchronize()
    measurements = {name: [] for name in functions}
    for group in range(groups):
        order = graphs if group % 2 == 0 else reversed(graphs)
        for name, graph, _ in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeat):
                graph.replay()
            end.record()
            end.synchronize()
            measurements[name].append(start.elapsed_time(end) * 1000 / repeat)
    return {
        name: {"window_us": values, "median_us": statistics.median(values)}
        for name, values in measurements.items()
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--loader-dir", type=Path, default=LOADER)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 64, 512, 2048, 8192])
    parser.add_argument("--seed", type=int, default=42117)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--groups", type=int, default=5)
    args = parser.parse_args()
    if min(*args.tokens, args.repeat, args.groups) < 1:
        parser.error("token counts, repeats and groups must be positive")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one verified idle GPU")
    props = torch.cuda.get_device_properties(0)
    if not props.gcnArchName.startswith("gfx942"):
        raise RuntimeError(f"Expected gfx942, found {props.gcnArchName}")
    torch.backends.cuda.matmul.allow_tf32 = False
    emit(
        "setup", model=str(args.model), layer=args.layer, tp=4, rank=args.rank,
        e=EXPERTS, topk=TOPK, h=HIDDEN, i=INTER, swiglu_limit=LIMIT,
        seed=args.seed, tokens=args.tokens, repeat=args.repeat, groups=args.groups,
        device=props.name, arch=props.gcnArchName, cus=props.multi_processor_count,
        aiter_config_fmoe=os.environ.get("AITER_CONFIG_FMOE"),
        input_scope="seeded BF16 hidden states, real layer router and routed weights",
        timing_scope="post-topk rank-local routed expert forward including sorting and quantization",
    )
    native, router = load_checkpoint(args.model, args.loader_dir, args.layer, args.rank)
    w13, s13, w2, s2 = native
    w13_ptpc, s13_ptpc = prepare_ptpc(w13, s13)
    w2_ptpc, s2_ptpc = prepare_ptpc(w2, s2)
    a8w4 = moe_a8w4.prepare_a8w4_weights(*native)
    emit("weights", native=[list(t.shape) for t in native], a8w4_padded_inter=a8w4.inter_padded)
    inputs, route_weights, route_ids = make_inputs(max(args.tokens), router, args.seed)

    for tokens in args.tokens:
        x, weights, ids = inputs[:tokens], route_weights[:tokens], route_ids[:tokens]

        def ptpc():
            return fused_moe(
                x, w13_ptpc, w2_ptpc, weights, ids,
                activation=ActivationType.Silu, quant_type=QuantType.per_Token,
                w1_scale=s13_ptpc, w2_scale=s2_ptpc, swiglu_limit=LIMIT,
            )

        def candidate():
            return fused_moe(
                x, None, None, weights, ids,
                activation=ActivationType.Silu, quant_type=QuantType.per_1x32,
                runtime_a8w4_weights=a8w4, swiglu_limit=LIMIT,
            )

        emit("launch", m=tokens)
        original_out, candidate_out = ptpc(), candidate()
        path = list(moe_a8w4.last_path)
        want, counts = reference(x, weights, ids, native)
        metrics = {
            "fp8_ptpc_vs_ref": errors(original_out, want),
            "a8w4_vs_ref": errors(candidate_out, want),
            "a8w4_vs_fp8_ptpc": errors(candidate_out, original_out.float()),
        }
        emit("validation", m=tokens, active_experts=sum(c > 0 for c in counts),
             max_routes_per_expert=max(counts), a8w4_path=path, **metrics)
        if not all(arm["finite"] for arm in metrics.values()):
            raise RuntimeError(f"Nonfinite output at M={tokens}")
        timings = graph_timings({"fp8_ptpc": ptpc, "a8w4": candidate}, args.repeat, args.groups)
        ratio = timings["a8w4"]["median_us"] / timings["fp8_ptpc"]["median_us"]
        emit("timings", m=tokens, a8w4_over_fp8_ptpc=ratio, **timings)


if __name__ == "__main__":
    main()
