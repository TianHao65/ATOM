#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Paired BF16 sparse attention microbench for DeepSeek-V4.1-Flash on gfx942."""

import argparse
import hashlib
import inspect
import json
import pathlib
import statistics
from collections import Counter

import torch
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src

from upstream.bench_sparse_attn_v4_gfx942 import HipSparseAttnV4


DEVICE = "cuda"
DTYPE = torch.bfloat16
TP = 4
SEED = 20261127
DECODE_POSITION = 99_999
PREFILL_CHUNKS = ((89_856, 8192, 8192), (98_048, 1952, 2048))
MAX_ABS_ERROR = 0.02
ARMS = ("triton", "hip", "triton_irope", "hip_irope")


def csr(lengths, padded_rows, capacity, parts):
    cursor = 0
    ptr = [0]
    for length in lengths:
        cursor += length
        ptr.append(cursor)
    ptr.extend([cursor] * (padded_rows - len(lengths)))
    values = torch.full((capacity,), -1, dtype=torch.int32, device=DEVICE)
    if parts:
        values[:cursor] = torch.cat(parts)
    indptr = torch.tensor(ptr, dtype=torch.int32, device=DEVICE)
    return values, indptr


def make_inputs(config, stage, ratio, start, live_rows, padded_rows, seed):
    heads = config["num_attention_heads"] // TP
    dim = config["head_dim"]
    topk = config["index_topk"] if ratio else 0
    window = config["sliding_window"]
    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    def randn(*shape, dtype=DTYPE, scale=0.5):
        return torch.randn(*shape, device=DEVICE, dtype=dtype, generator=generator) * scale

    q = randn(padded_rows, heads, dim)
    sink = randn(heads, dtype=torch.float32, scale=0.25)
    positions = torch.full((padded_rows,), start, dtype=torch.int64, device=DEVICE)
    pool_rows = (start + 1) // ratio if ratio else 0
    csa_indices = torch.randint(
        max(pool_rows, 1), (live_rows, topk), device=DEVICE,
        dtype=torch.int32, generator=generator,
    )
    if stage == "decode":
        csa, cptr = csr(
            [topk] * live_rows, padded_rows, padded_rows * topk, [csa_indices.flatten()],
        )
        ring = torch.arange(live_rows * window, dtype=torch.int32, device=DEVICE)
        swa, sptr = csr([window] * live_rows, padded_rows, padded_rows * window, [ring])
        return (
            q, randn(max(pool_rows, 1), dim), csa, cptr,
            randn(live_rows * window, dim), swa, sptr, sink, positions,
        )
    if stage == "decoder_last":
        # Last-layer QueryRows selects one query from a 128-row source.
        prefix, pptr = csr([topk], 1, window * (topk + window), [csa_indices.flatten()])
        local = torch.arange(window, dtype=torch.int32, device=DEVICE)
        extend, eptr = csr([window], 1, window * window, [local])
        return (
            q, randn(pool_rows, dim), prefix, pptr,
            randn(window, dim), extend, eptr, sink, positions,
        )

    # The BF16 serving layout appends past window rows to each compressed prefix.
    # Current-window KV comes from this forward's second source.
    positions[:live_rows] += torch.arange(live_rows, device=DEVICE)
    local_rows = torch.arange(live_rows, dtype=torch.int32, device=DEVICE)
    window_rows = torch.arange(window, dtype=torch.int32, device=DEVICE)
    local_indices = local_rows[:, None] - (window - 1) + window_rows[None, :]
    history_lengths = [0] * live_rows
    prefix_parts = [csa_indices.flatten()]
    if stage == "encoder":
        csa_history = (pool_rows + window + local_indices)[local_indices < 0]
        history_lengths = [max(window - 1 - i, 0) for i in range(live_rows)]
        rows_with_history = min(window - 1, live_rows)
        history = csa_history.split(history_lengths[:rows_with_history])
        prefix_parts = [
            torch.cat((csa_indices[i], history[i])) for i in range(rows_with_history)
        ]
        prefix_parts.append(csa_indices[rows_with_history:].flatten())
    prefix, pptr = csr(
        [topk + n for n in history_lengths], padded_rows, padded_rows * (topk + window), prefix_parts,
    )
    extend, eptr = csr(
        [min(i + 1, window) for i in range(live_rows)], padded_rows, padded_rows * window,
        [local_indices[local_indices >= 0]],
    )
    return (
        q, randn(pool_rows + window, dim), prefix, pptr,
        randn(padded_rows, dim), extend, eptr, sink, positions,
    )


def error_stats(actual, expected):
    actual = actual.float()
    expected = expected.float()
    difference = (actual - expected).abs()
    return {
        "max_abs": difference.max().item(),
        "mean_abs": difference.mean().item(),
        "rmse": difference.square().mean().sqrt().item(),
        "bf16_exact_fraction": (difference == 0).float().mean().item(),
    }


def graph_replay(fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    return graph.replay


def event_us(fn, iterations):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def test_case(hip, inputs, rope, split_count, live_rows, capture, rounds, iterations):
    *attention, positions = inputs
    q = attention[0]
    scale = q.shape[-1] ** -0.5
    outs = {arm: torch.empty_like(q) for arm in ARMS}
    rope_args = {
        "cos_cache": rope.cos_cache, "sin_cache": rope.sin_cache,
        "positions": positions, "rope_dim": rope.rope_dim,
    }

    def triton_attention():
        sparse_attn_v4_paged_2src(*attention, scale, out=outs["triton"])

    def hip_attention():
        hip(*attention, scale, out=outs["hip"], num_splits=split_count)

    def triton_irope():
        out = outs["triton_irope"]
        sparse_attn_v4_paged_2src(*attention, scale, out=out)
        rope(out.unsqueeze(0), positions, inverse=True)

    def hip_irope():
        hip(*attention, scale, out=outs["hip_irope"], num_splits=split_count, **rope_args)

    dispatch = dict(zip(ARMS, (triton_attention, hip_attention, triton_irope, hip_irope)))
    for fn in dispatch.values():
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    if capture:
        dispatch = {arm: graph_replay(fn) for arm, fn in dispatch.items()}
        torch.cuda.synchronize()
    correctness = {
        "attention": error_stats(outs["hip"], outs["triton"]),
        "attention_irope": error_stats(outs["hip_irope"], outs["triton_irope"]),
        "padded_rows_zero": all(not out[live_rows:].count_nonzero().item() for out in outs.values()),
    }
    passed = (
        correctness["attention"]["max_abs"] <= MAX_ABS_ERROR
        and correctness["attention_irope"]["max_abs"] <= MAX_ABS_ERROR
        and correctness["padded_rows_zero"]
    )
    if not passed:
        raise AssertionError(f"Sparse attention correctness check failed: {correctness}")

    times = {arm: [] for arm in ARMS}
    for round_num in range(rounds):
        order = ARMS if round_num % 2 == 0 else ARMS[::-1]
        for arm in order:
            times[arm].append(event_us(dispatch[arm], iterations))
    medians = {arm: statistics.median(samples) for arm, samples in times.items() if samples}
    return {"correctness": correctness, "round_means_us": times, "median_us": medians}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=pathlib.Path, required=True)
    parser.add_argument("--library", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--phase", choices=("all", "decode", "prefill"), default="all")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--decode-replays", type=int, default=200)
    parser.add_argument("--prefill-iterations", type=int, default=10)
    args = parser.parse_args()
    config_bytes = args.model_config.read_bytes()
    config = json.loads(config_bytes)
    config = config.get("text_config", config)
    props = torch.cuda.get_device_properties(0)
    if props.gcnArchName.split(":")[0] != "gfx942":
        raise RuntimeError(f"Expected gfx942, got {props.gcnArchName}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expose the single assigned GPU to this process.")
    scaling = config["rope_scaling"]
    rotary = config["qk_rope_head_dim"]
    ropes = {
        False: RotaryEmbedding(rotary, 102400, base=config["rope_theta"]).to(DEVICE),
        True: RotaryEmbedding(
            rotary, 102400, base=config["compress_rope_theta"],
            original_length=scaling["original_max_position_embeddings"], factor=scaling["factor"],
            beta_fast=scaling["beta_fast"], beta_slow=scaling["beta_slow"],
        ).to(DEVICE),
    }
    ratios = config["compress_ratios"][:config["num_hidden_layers"]]
    decoder_start = max(config["kv_source_layer_ids"])
    layer_counts = {
        "decode": Counter(ratios),
        "encoder": Counter(ratios[:decoder_start]),
        "decoder_tail": Counter(ratios[decoder_start:-1]),
        "decoder_last": Counter(ratios[-1:]),
    }
    window = config["sliding_window"]
    hip = HipSparseAttnV4(str(args.library.resolve()))
    results = {
        "atom_source": inspect.getsourcefile(inspect.unwrap(sparse_attn_v4_paged_2src)),
        "model_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "gpu": props.name, "arch": props.gcnArchName, "num_cus": props.multi_processor_count,
        "pytorch": torch.__version__, "hip": torch.version.hip,
        "tp": TP, "heads_per_rank": config["num_attention_heads"] // TP,
        "head_dim": config["head_dim"], "topk": config["index_topk"],
        "window": window, "rope_dim": rotary,
        "rope_theta": config["rope_theta"], "compress_rope_theta": config["compress_rope_theta"],
        "rope_scaling": scaling, "generator_seed_base": SEED,
        "rounds": args.rounds, "decode_replays_per_round": args.decode_replays,
        "prefill_iterations_per_round": args.prefill_iterations,
        "max_abs_error_threshold": MAX_ABS_ERROR, "synthetic_kv_inputs": True,
        "cases": [],
    }
    cases = []
    if args.phase in ("all", "decode"):
        cases += [("decode", DECODE_POSITION, n, n) for n in (1, 4, 8)]
        cases += [("decode", DECODE_POSITION, 1, 4)]
    if args.phase in ("all", "prefill"):
        cases += [("encoder", *chunk) for chunk in PREFILL_CHUNKS]
        cases += [
            ("decoder_tail", DECODE_POSITION + 1 - window, window, window),
            ("decoder_last", DECODE_POSITION, 1, 1),
        ]
    for idx, (stage, start, live, padded) in enumerate(cases):
        decode = stage == "decode"
        for ratio, layers in sorted(layer_counts[stage].items()):
            inputs = make_inputs(config, stage, ratio, start, live, padded, SEED + idx)
            q, _, cidx, _, _, sidx, _, *_ = inputs
            avg_keys = (cidx.numel() + sidx.numel()) // padded
            splits = hip.splits_for(padded, q.shape[1], avg_keys, hip.num_cus)
            rounds = 0 if decode and live != padded else args.rounds
            row = {
                "phase": "decode" if decode else "prefill", "stage": stage,
                "compress_ratio": ratio, "layers": layers,
                "first_position": start, "live_rows": live, "allocated_rows": padded,
                "hip_splits": splits, "prefix_allocated_entries": cidx.numel(),
                "extend_allocated_entries": sidx.numel(),
            }
            row.update(test_case(
                hip, inputs, ropes[bool(ratio)], splits, live, decode, rounds,
                args.decode_replays if decode else args.prefill_iterations,
            ))
            print(json.dumps(row, sort_keys=True), flush=True)
            results["cases"].append(row)
            del inputs, q, cidx, sidx
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
