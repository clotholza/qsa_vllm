# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 tensor-core QSA index scoring and fused CUDA top-k for SM80.

Scoring kernels are adapted from the official vLLM v0.30.0 implementation:
https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py
The decode tile pads the four real index heads to CUDA's MMA width; prefill
shares paged keys across query tiles. Both use BF16 inputs/FP32 accumulation.
A radix selector chooses compressed blocks and boundary ties in one launch.
"""

from __future__ import annotations

import os
import torch
from vllm.triton_utils import tl, triton

from ..graph_dispatch import decode_logits_columns
from .topk import topk_expand as _topk_expand

_DECODE_BLOCK_N = 64
_INDEXER_WORKSPACE_BYTES = int(os.environ.get("QSA_INDEXER_WORKSPACE_MB", "64")) * 1024 * 1024

@triton.jit
def _qsa_mqa_paged_uniform_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
    BLOCK_N: tl.constexpr,
    PROGRAMS_PER_REQUEST: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    NUM_COLUMNS: tl.constexpr = PAGE_TABLE_WIDTH * PAGE_SIZE
    DECODE_QUERY_LEN_PADDED: tl.constexpr = triton.next_power_of_2(DECODE_QUERY_LEN)
    NUM_HEADS_PADDED: tl.constexpr = max(
        16 // DECODE_QUERY_LEN_PADDED, triton.next_power_of_2(NUM_HEADS)
    )
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    tile_start = tl.program_id(1)
    query_offsets = tl.arange(0, DECODE_QUERY_LEN_PADDED)
    valid_query_offsets = query_offsets < DECODE_QUERY_LEN
    rows = request * DECODE_QUERY_LEN + query_offsets
    visible = tl.load(
        visible_blocks_ptr + rows,
        mask=valid_query_offsets,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    if tile_start * BLOCK_N >= max_visible:
        return
    tile_end = tl.minimum(tl.cdiv(max_visible, BLOCK_N), tl.cdiv(NUM_COLUMNS, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    n = tl.arange(0, DECODE_QUERY_LEN_PADDED * NUM_HEADS_PADDED)
    query_offset = n // NUM_HEADS_PADDED
    head = n % NUM_HEADS_PADDED
    valid_query = (query_offset < DECODE_QUERY_LEN) & (head < NUM_HEADS)
    query = tl.load(
        q_ptr
        + (request * DECODE_QUERY_LEN + query_offset)[None, :] * stride_q_row
        + head[None, :] * stride_q_head
        + dims[:, None],
        mask=valid_query[None, :] & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(tile_start, tile_end, PROGRAMS_PER_REQUEST, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, PAGE_TABLE_WIDTH - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        keys = tl.load(
            k_cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.where(valid_query[None, :], tl.maximum(scores, 0.0), 0.0)
        scores = tl.reshape(
            scores,
            (BLOCK_N, DECODE_QUERY_LEN_PADDED, NUM_HEADS_PADDED),
        )
        score = tl.sum(scores, axis=2) * (HEAD_DIM**-0.5)
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=valid_query_offsets[None, :]
            & (columns[:, None] < NUM_COLUMNS)
            & (columns[:, None] < visible[None, :]),
        )


@triton.jit(do_not_specialize=["num_rows", "query_offset"])
def _qsa_mqa_paged_prefill_kernel(
    q_ptr,
    k_cache_ptr,
    page_table_ptr,
    query_start_loc_ptr,
    visible_blocks_ptr,
    logits_ptr,
    stride_q_row,
    stride_q_head,
    stride_cache_block,
    stride_cache_token,
    stride_table_req,
    stride_logits_row,
    num_rows,
    query_offset,
    page_table_width,
    PAGE_SIZE: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TILE_R: tl.constexpr,
    BLOCK_N: tl.constexpr,
    K_TILES: tl.constexpr,
    STAGES: tl.constexpr,
) -> None:
    num_columns = page_table_width * PAGE_SIZE
    NUM_HEADS_PADDED: tl.constexpr = triton.next_power_of_2(NUM_HEADS)
    # tl.dot requires a reduction dimension of at least 16.
    BLOCK_D: tl.constexpr = max(16, triton.next_power_of_2(HEAD_DIM))
    request = tl.program_id(0)
    query_base = tl.load(query_start_loc_ptr)
    query_end = query_offset + num_rows
    request_start = tl.maximum(
        tl.load(query_start_loc_ptr + request) - query_base, query_offset
    )
    request_end = tl.minimum(
        tl.load(query_start_loc_ptr + request + 1) - query_base, query_end
    )
    absolute_row_start = request_start + tl.program_id(1) * TILE_R
    if absolute_row_start >= request_end:
        return

    lanes = tl.arange(0, TILE_R)
    absolute_rows = absolute_row_start + lanes
    rows = absolute_rows - query_offset
    valid_rows = absolute_rows < request_end
    visible = tl.load(
        visible_blocks_ptr + absolute_rows,
        mask=valid_rows,
        other=0,
    )
    max_visible = tl.max(visible, axis=0)
    k_tile_start = tl.program_id(2) * K_TILES
    if k_tile_start * BLOCK_N >= max_visible:
        return
    k_tile_end = tl.minimum(k_tile_start + K_TILES, tl.cdiv(max_visible, BLOCK_N))
    k_tile_end = tl.minimum(k_tile_end, tl.cdiv(num_columns, BLOCK_N))

    dims = tl.arange(0, BLOCK_D)
    m = tl.arange(0, TILE_R * NUM_HEADS_PADDED)
    q_row_offsets = m // NUM_HEADS_PADDED
    q_rows = absolute_row_start + q_row_offsets
    heads = m % NUM_HEADS_PADDED
    query = tl.load(
        q_ptr
        + q_rows[None, :] * stride_q_row
        + heads[None, :] * stride_q_head
        + dims[:, None],
        mask=(heads[None, :] < NUM_HEADS)
        & (absolute_row_start + q_row_offsets[None, :] < request_end)
        & (dims[:, None] < HEAD_DIM),
        other=0.0,
    )
    column_offsets = tl.arange(0, BLOCK_N)
    for tile in tl.range(k_tile_start, k_tile_end, num_stages=STAGES):
        columns = tile * BLOCK_N + column_offsets
        live = columns < max_visible
        logical_page = tl.minimum(columns // PAGE_SIZE, page_table_width - 1)
        page_offset = columns % PAGE_SIZE
        physical_page = tl.load(
            page_table_ptr + request * stride_table_req + logical_page,
            mask=live,
            other=0,
        )
        keys = tl.load(
            k_cache_ptr
            + physical_page[:, None].to(tl.int64) * stride_cache_block
            + page_offset[:, None] * stride_cache_token
            + dims[None, :],
            mask=live[:, None] & (dims[None, :] < HEAD_DIM),
            other=0.0,
            eviction_policy="evict_first",
        )
        # BF16 tensor-core products accumulate into FP32 on SM80.
        scores = tl.dot(keys, query, out_dtype=tl.float32)
        scores = tl.reshape(scores, (BLOCK_N, TILE_R, NUM_HEADS_PADDED))
        score = tl.sum(tl.maximum(scores, 0.0), axis=2) * (HEAD_DIM**-0.5)
        store_mask = (
            valid_rows[None, :]
            & (columns[:, None] < visible[None, :])
            & (columns[:, None] < num_columns)
        )
        tl.store(
            logits_ptr + rows[None, :] * stride_logits_row + columns[:, None],
            score,
            mask=store_mask,
        )



def qsa_select_paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    decode_query_len: int,
    out: torch.Tensor,
    max_seq_len: int,
    *, query_positions, sequence_lengths, token_to_req,
    main_block_table=None, main_page_size=1, valid_counts_out=None,
) -> None:
    """Score decode queries and write expanded selection in the CUDA top-k."""
    num_requests = q.shape[0] // decode_query_len
    columns = decode_logits_columns(
        max_seq_len, compress_ratio, page_table.shape[1] * k_cache.shape[1],
    )
    requests_per_chunk = max(
        1, _INDEXER_WORKSPACE_BYTES // (columns * 4 * decode_query_len)
    )
    workspace_rows = min(num_requests, requests_per_chunk) * decode_query_len
    logits_workspace = torch.empty(
        (workspace_rows, columns), dtype=torch.float32, device=q.device,
    )
    for first_request in range(0, num_requests, requests_per_chunk):
        last_request = min(first_request + requests_per_chunk, num_requests)
        first_row = first_request * decode_query_len
        last_row = last_request * decode_query_len
        queries = q[first_row:last_row]
        table = page_table[first_request:last_request]
        visible = visible_blocks[first_row:last_row]
        logits = logits_workspace[:last_row - first_row]
        programs_per_request = max(16, triton.cdiv(128, table.shape[0]))
        _qsa_mqa_paged_uniform_kernel[
            (table.shape[0], programs_per_request)
        ](
            queries, k_cache, table, visible, logits,
            *queries.stride()[:-1], *k_cache.stride()[:2],
            *table.stride()[:-1], *logits.stride()[:-1],
            PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=table.shape[1],
            NUM_HEADS=q.shape[1], HEAD_DIM=q.shape[2],
            DECODE_QUERY_LEN=decode_query_len, BLOCK_N=_DECODE_BLOCK_N,
            PROGRAMS_PER_REQUEST=programs_per_request, STAGES=1,
            num_warps=4, num_stages=1,
        )
        _topk_expand(
            logits, visible, query_positions[first_row:last_row], sequence_lengths,
            token_to_req[first_row:last_row], out[first_row:last_row], compress_ratio,
            main_block_table=main_block_table, main_page_size=main_page_size,
            valid_counts_out=(valid_counts_out[first_row:last_row]
                              if valid_counts_out is not None else None),
        )


def _prefill_logits(
    q, k_cache, page_table, query_start_loc, visible_blocks, logits_workspace,
    max_query_len, logits_width, query_offset, num_queries,
):
    logits = logits_workspace[:num_queries * logits_width].view(num_queries, logits_width)
    # Four index heads x 32 query rows reuse each BF16 key tile on Ampere.
    tile_r, block_n, k_tiles, stages = 32, 64, 8, 1
    _qsa_mqa_paged_prefill_kernel[
        (page_table.shape[0],
         triton.cdiv(min(num_queries, max_query_len), tile_r),
         triton.cdiv(logits_width, block_n * k_tiles))
    ](
        q, k_cache, page_table, query_start_loc, visible_blocks, logits,
        *q.stride()[:-1], *k_cache.stride()[:2],
        *page_table.stride()[:-1], *logits.stride()[:-1],
        num_queries, query_offset, page_table.shape[1],
        PAGE_SIZE=k_cache.shape[1], NUM_HEADS=q.shape[1], HEAD_DIM=q.shape[2],
        TILE_R=tile_r, BLOCK_N=block_n, K_TILES=k_tiles, STAGES=stages,
        num_warps=4, num_stages=1,
    )
    return logits


def qsa_select_paged_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    visible_blocks: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    max_query_len: int,
    out: torch.Tensor,
    max_seq_len: int,
    *, query_positions, sequence_lengths, token_to_req,
    main_block_table=None, main_page_size=1, valid_counts_out=None,
) -> None:
    """Packed prefill scoring and fused final token-selection writeout."""
    rows = q.shape[0]
    columns = page_table.shape[1] * k_cache.shape[1]
    logits_width = triton.cdiv(triton.cdiv(max_seq_len, compress_ratio), 64) * 64
    logits_width = min(max(64, logits_width), columns)
    rows_per_chunk = max(1, _INDEXER_WORKSPACE_BYTES // (logits_width * 4))
    logits_workspace = torch.empty(
        min(rows, rows_per_chunk) * logits_width,
        dtype=torch.float32, device=q.device,
    )
    for query_start in range(0, rows, rows_per_chunk):
        query_end = min(query_start + rows_per_chunk, rows)
        logits = _prefill_logits(
            q, k_cache, page_table, query_start_loc, visible_blocks,
            logits_workspace, max_query_len, logits_width,
            query_offset=query_start, num_queries=query_end - query_start,
        )
        _topk_expand(
            logits, visible_blocks[query_start:query_end],
            query_positions[query_start:query_end], sequence_lengths,
            token_to_req[query_start:query_end], out[query_start:query_end], compress_ratio,
            main_block_table=main_block_table, main_page_size=main_page_size,
            valid_counts_out=(valid_counts_out[query_start:query_end]
                              if valid_counts_out is not None else None),
        )


def select_paged_tokens(
    q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
    token_topk, compress_ratio, out=None, *, query_start_loc, max_query_len,
    max_seq_len, num_decodes=0, num_decode_tokens=0, decode_query_len=1,
    visible_blocks=None, main_block_table=None, main_page_size=1,
    valid_counts_out=None,
):
    """Select blocks and write expanded decode/prefill tokens in the same kernel."""

    rows = q.shape[0]
    if out is None:
        out = torch.empty(
            (rows, token_topk + compress_ratio - 1),
            dtype=torch.int32, device=q.device,
        )
    if not rows:
        return out
    if visible_blocks is None:
        safe_requests = token_to_req.clamp_min(0).long()
        visible_blocks = torch.minimum(
            (query_positions + 1) // compress_ratio,
            sequence_lengths.index_select(0, safe_requests) // compress_ratio,
        ).to(torch.int32)
        visible_blocks = torch.where(token_to_req >= 0, visible_blocks, 0)
    if num_decode_tokens:
        qsa_select_paged_decode(
            q[:num_decode_tokens], k_cache, page_table[:num_decodes],
            visible_blocks[:num_decode_tokens], token_topk, compress_ratio,
            decode_query_len, out[:num_decode_tokens], max_seq_len,
            query_positions=query_positions[:num_decode_tokens],
            sequence_lengths=sequence_lengths, token_to_req=token_to_req[:num_decode_tokens],
            main_block_table=main_block_table, main_page_size=main_page_size,
            valid_counts_out=(valid_counts_out[:num_decode_tokens]
                              if valid_counts_out is not None else None),
        )
    if num_decode_tokens < rows:
        qsa_select_paged_prefill(
            q[num_decode_tokens:], k_cache, page_table[num_decodes:],
            query_start_loc[num_decodes:], visible_blocks[num_decode_tokens:],
            token_topk, compress_ratio, max_query_len,
            out[num_decode_tokens:], max_seq_len,
            query_positions=query_positions[num_decode_tokens:],
            sequence_lengths=sequence_lengths, token_to_req=token_to_req[num_decode_tokens:],
            main_block_table=main_block_table, main_page_size=main_page_size,
            valid_counts_out=(valid_counts_out[num_decode_tokens:]
                              if valid_counts_out is not None else None),
        )
    return out
