"""Read the native checkpoint's routed experts as one padded TP4 rank."""

import json
import mmap
import struct
from collections import defaultdict
from pathlib import Path


EXPERTS = 384
HIDDEN = 5120
INTERMEDIATE = 2304
TP = 4
LOCAL = INTERMEDIATE // TP
PADDED = 640
GROUP = 32


def _header(shard):
    size = struct.unpack("<Q", shard.read(8))[0]
    return 8 + size, json.loads(shard.read(size))


def load_tp4_layer(model_root, layer=20, rank=0):
    """Return ((w13, s13), (w2, s2)) as contiguous CUDA uint8 tensors.

    w13 and s13 store separate padded gate and up halves. All checkpoint
    bytes, including raw E8M0 scale encodings, are copied without conversion.
    """
    import numpy as np
    import torch

    if not 0 <= rank < TP or not 0 <= layer < 40:
        raise ValueError("Expected a TP4 rank between 0 and 3 and a layer between 0 and 39")
    root = Path(model_root)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    w13 = torch.zeros((EXPERTS, 2 * PADDED, HIDDEN // 2), dtype=torch.uint8, device="cpu")
    s13 = torch.zeros((EXPERTS, 2 * PADDED, HIDDEN // GROUP), dtype=torch.uint8, device="cpu")
    w2 = torch.zeros((EXPERTS, HIDDEN, PADDED // 2), dtype=torch.uint8, device="cpu")
    s2 = torch.zeros((EXPERTS, HIDDEN, PADDED // GROUP), dtype=torch.uint8, device="cpu")
    gate_up, gate_up_scales = w13.numpy(), s13.numpy()
    down, down_scales = w2.numpy(), s2.numpy()
    requests = defaultdict(list)
    model_rows = slice(rank * LOCAL, (rank + 1) * LOCAL)
    model_packed = slice(model_rows.start // 2, model_rows.stop // 2)
    model_scales = slice(model_rows.start // GROUP, model_rows.stop // GROUP)

    for expert in range(EXPERTS):
        ss = f"layers.{layer}.ffn.experts.{expert}"
        for proj, dst_row in (("w1", 0), ("w3", PADDED)):
            local_rows = slice(dst_row, dst_row + LOCAL)
            requests[index[f"{ss}.{proj}.weight"]].append((
                f"{ss}.{proj}.weight", "I8", (INTERMEDIATE, HIDDEN // 2),
                (model_rows, slice(None)), gate_up[expert, local_rows],
            ))
            requests[index[f"{ss}.{proj}.scale"]].append((
                f"{ss}.{proj}.scale", "F8_E8M0", (INTERMEDIATE, HIDDEN // GROUP),
                (model_rows, slice(None)), gate_up_scales[expert, local_rows],
            ))
        requests[index[f"{ss}.w2.weight"]].append((
            f"{ss}.w2.weight", "I8", (HIDDEN, INTERMEDIATE // 2),
            (slice(None), model_packed), down[expert, :, : LOCAL // 2],
        ))
        requests[index[f"{ss}.w2.scale"]].append((
            f"{ss}.w2.scale", "F8_E8M0", (HIDDEN, INTERMEDIATE // GROUP),
            (slice(None), model_scales), down_scales[expert, :, : LOCAL // GROUP],
        ))

    for filename, entries in requests.items():
        with (root / filename).open("rb") as shard:
            data_start, header = _header(shard)
            with mmap.mmap(shard.fileno(), 0, access=mmap.ACCESS_READ) as data:
                for key, dtype, shape, source, target in entries:
                    meta = header[key]
                    start, end = meta["data_offsets"]
                    if (
                        meta["dtype"] != dtype
                        or tuple(meta["shape"]) != shape
                        or not 0 <= start < end <= len(data) - data_start
                        or end - start != shape[0] * shape[1]
                    ):
                        raise ValueError(f"Unexpected layout for {key} in {filename}")
                    raw = np.ndarray(shape, dtype=np.uint8, buffer=data, offset=data_start + start)
                    np.copyto(target, raw[source])
                    del raw

    return (w13.cuda(), s13.cuda()), (w2.cuda(), s2.cuda())
