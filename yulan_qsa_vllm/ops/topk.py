# SPDX-License-Identifier: Apache-2.0
"""Length-dispatched SM80 top-k with exact FP32 thresholds and ID ties."""

from pathlib import Path

import torch
from torch.utils.cpp_extension import load


_topk_root = Path(__file__).resolve().parents[2]
_topk_build = _topk_root / ".cache" / "qsa_topk_sm80"
_topk_build.mkdir(parents=True, exist_ok=True)
load(
    name="yulan_qsa_topk_sm80",
    sources=[str(Path(__file__).with_name("csrc") / "topk.cu")],
    build_directory=str(_topk_build),
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_80,code=sm_80"],
    with_cuda=True,
    is_python_module=False,
    verbose=False,
)


def topk(scores, lengths, output):
    torch.ops.yulan_qsa_kernels.topk(scores, lengths, output)


def topk_expand(scores, lengths, query_positions, sequence_lengths, token_to_req,
                output, compress_ratio, *, main_block_table=None, main_page_size=1,
                valid_counts_out=None):
    """Select blocks and write expanded slots, causal tail and counts in place."""
    torch.ops.yulan_qsa_kernels.topk_expand(
        scores, lengths, query_positions, sequence_lengths, token_to_req,
        main_block_table if main_block_table is not None else output,
        output, valid_counts_out if valid_counts_out is not None else output,
        compress_ratio, main_page_size, main_block_table is not None,
        valid_counts_out is not None,
    )
