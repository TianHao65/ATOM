# SPDX-License-Identifier: MIT
"""Experimental gfx942 V4.1 Decode K512 launcher for AITER 7f36447ac7a2."""

import ctypes
import logging
from functools import cache

import torch
from aiter.jit.utils.chip_info import get_gfx

from atom.utils import envs

logger = logging.getLogger("atom")

# Candidate rows, ratio-2 rows, and ratio-1 rows in the 100K serving setup.
_TESTED_WIDTHS = (16384, 51200, 102400)
_MAX_UNSPLIT_WIDTH = 49 * 2048


@cache
def _load_library(path: str):
    library = ctypes.CDLL(path)
    library.topk_ballot_parts.argtypes = [ctypes.c_int, ctypes.c_int]
    library.topk_ballot_parts.restype = ctypes.c_int
    library.topk_ballot_workspace_bytes.argtypes = [ctypes.c_int, ctypes.c_int]
    library.topk_ballot_workspace_bytes.restype = ctypes.c_longlong
    library.topk_ballot_split_launch.argtypes = (
        [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_void_p] * 2
    )
    library.topk_ballot_split_launch.restype = ctypes.c_int
    logger.info("DeepSeek-V4.1 Ballot top-k library loaded from %s", path)
    return library


def try_ballot_topk(
    logits: torch.Tensor, lengths: torch.Tensor, indices: torch.Tensor
) -> bool:
    """Fill one stable top-k row in place if the serving shape is supported."""
    path = envs.ATOM_DSV41_BALLOT_TOPK_LIB
    if not (
        path
        and logits.is_cuda
        and logits.dtype == torch.float32
        and logits.ndim == 2
        and logits.size(0) == 1
        and lengths.device == indices.device == logits.device
        and lengths.dtype == indices.dtype == torch.int32
        and lengths.shape == (1,)
        and indices.shape == (1, 512)
        and lengths.is_contiguous()
        and indices.is_contiguous()
    ):
        return False
    width = logits.size(1)
    if (
        width not in _TESTED_WIDTHS
        or logits.stride() != (width, 1)
        or logits.data_ptr() % 16
    ):
        return False

    with torch.cuda.device(logits.device):
        if get_gfx() != "gfx942":
            return False
        library = _load_library(path)
        parts = library.topk_ballot_parts(1, width)
        if parts == 1 and width > _MAX_UNSPLIT_WIDTH:
            return False
        scratch = None
        if parts > 1:
            # Use the current stream's allocator. During graph capture this
            # allocation and zeroing belong to the graph that launches it,
            # so overlapping graphs or streams never share mutable scratch.
            scratch = torch.zeros(
                library.topk_ballot_workspace_bytes(1, parts),
                dtype=torch.uint8,
                device=logits.device,
            )
        stream = torch.cuda.current_stream(logits.device)
        status = library.topk_ballot_split_launch(
            logits.data_ptr(),
            lengths.data_ptr(),
            indices.data_ptr(),
            None,
            1,
            width,
            1,
            indices.size(1),
            parts,
            None if scratch is None else scratch.data_ptr(),
            stream.cuda_stream,
        )
        if status:
            raise RuntimeError(f"DeepSeek-V4.1 Ballot top-k launch failed ({status})")
    return True
