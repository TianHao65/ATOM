#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Check the ATOM adapter's split output aliasing and padded CSR graph replay."""

import argparse
import json
from pathlib import Path

import torch
from atom.model_ops.deepseek_v41.hip_sparse_attention import (
    hip_sparse_attention,
    supports_hip_sparse_attention,
)
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding
from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src

from bench_atom_sparse_attention import MAX_ABS_ERROR, error_stats, make_inputs


def verify(config, rope, stage, start, live, padded, ratio, empty=False):
    *inputs, positions = make_inputs(config, stage, ratio, start, live, padded, 20261127)
    q = inputs[0]
    if empty:
        inputs[3].zero_()
        inputs[6].zero_()
    # A kernel may address these capacity entries only if it reads beyond CSR.
    for indices, indptr in ((inputs[2], inputs[3]), (inputs[5], inputs[6])):
        indices[indptr[-1].item():].fill_(1 << 30)
    scale = q.shape[-1] ** -0.5
    base = sparse_attn_v4_paged_2src(*inputs, scale)
    rope(base.unsqueeze(0), positions, inverse=True)
    original = q.clone()
    assert supports_hip_sparse_attention(q)
    decode = stage == "decode"

    def launch():
        q.copy_(original)
        return hip_sparse_attention(*inputs, scale, decode=decode, rope=rope, positions=positions)

    out = launch()
    alias = out.data_ptr() == q.data_ptr()
    compared = error_stats(out, base)
    if alias or decode:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = launch()
        graph.replay()
        graph.replay()
        replayed = error_stats(out, base)
    else:
        replayed = None
    padded_zero = not out[live if not empty else 0:].count_nonzero().item()
    result = {
        "stage": stage,
        "ratio": ratio,
        "live": live,
        "padded": padded,
        "empty": empty,
        "aliased": alias,
        "padded_zero": padded_zero,
        "eager": compared,
        "replay": replayed,
    }
    if (
        not compared["max_abs"] <= MAX_ABS_ERROR
        or (replayed is not None and not replayed["max_abs"] <= MAX_ABS_ERROR)
        or not padded_zero
        or alias != (decode or stage == "decoder_last")
    ):
        raise AssertionError(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.model_config.read_text())
    config = config.get("text_config", config)
    scaling = config["rope_scaling"]
    rope = RotaryEmbedding(
        64, 102400, base=config["compress_rope_theta"],
        original_length=scaling["original_max_position_embeddings"], factor=scaling["factor"],
        beta_fast=scaling["beta_fast"], beta_slow=scaling["beta_slow"],
    ).to("cuda")
    cases = [
        ("decode", 99999, 1, 1, 2),
        ("decode", 99999, 1, 4, 1),
        ("decode", 99999, 1, 4, 1, True),
        ("encoder", 98048, 1952, 2048, 2),
        ("decoder_tail", 99872, 128, 128, 1),
        ("decoder_last", 99999, 1, 1, 1),
    ]
    result = []
    for case in cases:
        row = verify(config, rope, *case)
        print(json.dumps(row), flush=True)
        result.append(row)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
