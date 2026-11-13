# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16 paged QSA kernels for CUDA SM80.

Adapted from vLLM v0.30.0:
https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/models/qwen4_exp/amd/ops/qsa.py
The paged split-K GQA, causal-tail expansion, cache stores and raw-key group
compression retain the upstream algorithms. Prefill pads the Tensor Core M
dimension to 16. SM80 decode uses physical sparse indices, per-row valid
counts and a separate launch configuration. Main and index caches remain BF16. No hardware preflight or architecture-ratio check is added.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

@triton.jit
def _expand_qsa_indices_kernel(
    block_indices_ptr,
    query_positions_ptr,
    sequence_lengths_ptr,
    token_to_req_ptr,
    output_ptr,
    main_block_table_ptr,
    valid_counts_ptr,
    stride_main_table_req,
    stride_blocks_row,
    stride_blocks_column,
    stride_output_row,
    stride_output_column,
    rows,
    num_requests,
    BLOCK_TOPK: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_TOPK: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    COLUMN_BLOCK: tl.constexpr,
    PHYSICAL_OUTPUT: tl.constexpr,
    MAIN_PAGE_SIZE: tl.constexpr,
    MAIN_TABLE_WIDTH: tl.constexpr,
    STORE_VALID_COUNTS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    columns = tl.program_id(1) * COLUMN_BLOCK + tl.arange(0, COLUMN_BLOCK)
    query_position = tl.load(query_positions_ptr + row)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    sequence_length = tl.load(
        sequence_lengths_ptr + safe_request,
        mask=(request >= 0) & (request < num_requests),
        other=0,
    )
    complete_blocks = tl.minimum(
        tl.minimum(
            (query_position + 1) // COMPRESS_RATIO,
            sequence_length // COMPRESS_RATIO,
        ),
        BLOCK_TOPK,
    )
    expanded_count = complete_blocks * COMPRESS_RATIO
    tail_start = ((query_position + 1) // COMPRESS_RATIO) * COMPRESS_RATIO
    tail_count = (query_position + 1) - tail_start

    is_expanded = columns < expanded_count
    block_rank = columns // COMPRESS_RATIO
    offset = columns % COMPRESS_RATIO
    safe_rank = tl.minimum(block_rank, BLOCK_TOPK - 1)
    block = tl.load(
        block_indices_ptr + row * stride_blocks_row + safe_rank * stride_blocks_column,
        mask=(row < rows) & is_expanded,
        other=-1,
    )
    expanded = block * COMPRESS_RATIO + offset
    tail_offset = columns - expanded_count
    is_tail = (
        (columns >= expanded_count)
        & (tail_offset < tail_count)
        & (tail_offset < COMPRESS_RATIO - 1)
    )
    token = tl.where(is_expanded, expanded, tail_start + tail_offset)
    valid = (
        (row < rows)
        & (columns < OUTPUT_WIDTH)
        & (is_expanded | is_tail)
        & (token >= 0)
        & (token < sequence_length)
    )
    if STORE_VALID_COUNTS:
        if tl.program_id(1) == 0:
            selected_count = tl.minimum(expanded_count + tail_count, sequence_length)
            selected_count = tl.where((request >= 0) & (request < num_requests), selected_count, 0)
            tl.store(valid_counts_ptr + row, selected_count)
    if PHYSICAL_OUTPUT:
        logical_page = tl.maximum(token, 0) // MAIN_PAGE_SIZE
        physical_page = tl.load(
            main_block_table_ptr + safe_request * stride_main_table_req + logical_page,
            mask=valid & (logical_page < MAIN_TABLE_WIDTH), other=-1,
        )
        token = physical_page * MAIN_PAGE_SIZE + token % MAIN_PAGE_SIZE
        valid &= physical_page >= 0
    tl.store(
        output_ptr + row * stride_output_row + columns * stride_output_column,
        tl.where(valid, token, -1),
        mask=(row < rows) & (columns < OUTPUT_WIDTH),
    )


@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    valid_counts_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    stride_output_gate_dim,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HAS_OUTPUT_GATE: tl.constexpr,
    INDICES_ARE_PHYSICAL: tl.constexpr,
    HAS_VALID_COUNTS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    if HAS_VALID_COUNTS:
        valid_count = tl.load(valid_counts_ptr + row)
        row_tiles = tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N)
    else:
        row_tiles = NUM_TILES
    split_tile_start = split_id * row_tiles // NUM_SPLITS
    split_tile_end = (split_id + 1) * row_tiles // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        if INDICES_ARE_PHYSICAL:
            physical_page = logical_page
            valid = (logical_token >= 0) & (physical_page < num_cache_blocks)
        else:
            valid = (
                (request >= 0)
                & (request < num_requests)
                & (logical_token >= 0)
                & (logical_page < PAGE_TABLE_WIDTH)
            )
            physical_page = tl.load(
                block_table_ptr
                + safe_request * stride_table_req
                + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        if HAS_OUTPUT_GATE:
            # Match the standalone gate's BF16 attention-output rounding.
            normalized_output = normalized_output.to(output_ptr.dtype.element_ty)
            output_gate = tl.load(
                output_gate_ptr
                + row * stride_output_gate_row
                + (first_head + head_offsets[:, None]) * stride_output_gate_head
                + dim_offsets[None, :] * stride_output_gate_dim,
                mask=output_mask,
                other=0.0,
            ).to(tl.float32)
            normalized_output = normalized_output.to(tl.float32) * tl.sigmoid(output_gate)
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_id * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    output_gate_ptr,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    stride_output_gate_dim,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    HAS_OUTPUT_GATE: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + ((split_offsets[:, None] * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    if HAS_OUTPUT_GATE:
        # Match the standalone gate's BF16 attention-output rounding.
        merged = merged.to(output_ptr.dtype.element_ty)
        output_gate = tl.load(
            output_gate_ptr
            + row * stride_output_gate_row
            + head * stride_output_gate_head
            + dim_offsets * stride_output_gate_dim,
        ).to(tl.float32)
        merged = merged.to(tl.float32) * tl.sigmoid(output_gate)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,  # this step's raw key rows, straight from activations
    raw_positions_ptr,  # this step's per-token positions
    compressor_state_cache_ptr,  # per-request ring of previous raw keys
    rope_cache_ptr,  # packed RoPE position tail of the ring
    compressor_state_table_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_raw_dim,
    stride_raw_positions_row,
    stride_raw_positions_dim,
    stride_compressor_state_block,
    stride_compressor_state_token,
    stride_compressor_state_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_compressor_state_table_req,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_compressor_state_blocks,
    num_requests,
    COMPRESSOR_STATE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # A group can span the compressor-state ring (older members) and this
    # step's raw rows (members at positions >= chunk_start_position).
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims * stride_raw_dim,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64)
            * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims * stride_compressor_state_dim,
            mask=valid_row
            & ~use_raw
            & valid_compressor_state_block
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        raw_position_values = tl.load(
            raw_positions_ptr
            + raw_first_row * stride_raw_positions_row
            + position_dims * stride_raw_positions_dim,
            mask=valid_row
            & first_from_raw
            & (raw_first_row >= query_row_start)
            & (raw_first_row < query_row_end)
            & (raw_first_row < num_rows)
            & (position_dims < 3),
            other=0,
        )
        compressor_state_position_values = tl.load(
            rope_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_rope_block
            + (first_position % COMPRESSOR_STATE_SIZE) * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_row
            & ~first_from_raw
            & valid_compressor_state_block
            & (position_dims < 3),
            other=0,
        )
        position_values = tl.where(
            first_from_raw,
            raw_position_values,
            compressor_state_position_values,
        )
    else:
        position_values = tl.where(valid_row, first_position, 0)
    tl.store(
        first_positions_ptr
        + row * stride_positions_row
        + position_dims * stride_positions_dim,
        position_values,
        mask=(row < num_rows) & (position_dims < 3),
    )



def expand_qsa_block_indices_cuda(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
    *,
    main_block_table: torch.Tensor | None = None,
    main_page_size: int = 1,
    valid_counts_out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand chosen complete blocks plus the visible incomplete causal tail."""
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32, device=block_indices.device,
        )
    if not block_indices.shape[0]:
        return out
    column_block = 256
    _expand_qsa_indices_kernel[
        (block_indices.shape[0], triton.cdiv(output_width, column_block))
    ](
        block_indices, query_positions, sequence_lengths, token_to_req, out,
        main_block_table if main_block_table is not None else block_indices,
        valid_counts_out if valid_counts_out is not None else out,
        main_block_table.stride(0) if main_block_table is not None else 0,
        *block_indices.stride(), *out.stride(),
        block_indices.shape[0], sequence_lengths.shape[0],
        BLOCK_TOPK=token_topk // compress_ratio,
        COMPRESS_RATIO=compress_ratio, TOKEN_TOPK=token_topk,
        OUTPUT_WIDTH=output_width, COLUMN_BLOCK=column_block,
        PHYSICAL_OUTPUT=main_block_table is not None, MAIN_PAGE_SIZE=main_page_size,
        MAIN_TABLE_WIDTH=main_block_table.shape[1] if main_block_table is not None else 1,
        STORE_VALID_COUNTS=valid_counts_out is not None, num_warps=4,
    )
    return out


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    output_gate: torch.Tensor | None = None,
    use_decode_kernel: bool = False,
    indices_are_physical: bool = False,
    valid_counts: torch.Tensor | None = None,
    decode_config: tuple[int, int, int] | None = None,
) -> torch.Tensor:
    """BF16 paged GQA with physical-index SM80 decode and Tensor Core prefill."""
    if out is None:
        out = torch.empty_like(q)
    if not q.shape[0]:
        return out
    has_output_gate = output_gate is not None
    gate_pointer = output_gate if has_output_gate else out
    gate_strides = output_gate.stride() if has_output_gate else (0, 0, 0)
    group_size = q.shape[1] // k_cache.shape[2]
    base_programs = q.shape[0] * k_cache.shape[2]
    block_m = max(16, triton.next_power_of_2(group_size))
    if use_decode_kernel:
        if decode_config is not None:
            block_n, target_splits, partial_warps = decode_config
        elif base_programs <= 8:
            block_n, target_splits, partial_warps = 32, 64, 1
        elif base_programs <= 32:
            block_n, target_splits, partial_warps = 32, 16, 1
        elif base_programs <= 128:
            block_n, target_splits, partial_warps = 32, 4, 4
        else:
            block_n, target_splits, partial_warps = 64, 1, 4
    else:
        small_profile_limit = 8 if group_size <= 8 else 4
        if base_programs <= small_profile_limit:
            block_n, target_splits, partial_warps = 16, 32, 4
        elif base_programs < 32:
            block_n, target_splits, partial_warps = 32, 16, 4
        elif base_programs <= 256:
            block_n, target_splits, partial_warps = 64, 8, 4
        elif base_programs <= 512:
            block_n, target_splits, partial_warps = 64, 4, 4
        else:
            block_n, target_splits, partial_warps = 64, 1, 4
    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device,
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32, device=q.device,
        )
    _qsa_sparse_paged_gqa_splitk_kernel[
        (q.shape[0], k_cache.shape[2], num_splits)
    ](
        q, k_cache, v_cache, logical_indices, block_table, token_to_req,
        partial_output, partial_lse, out, gate_pointer,
        valid_counts if valid_counts is not None else logical_indices,
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        logical_indices.stride(0), block_table.stride(0),
        out.stride(0), out.stride(1),
        *gate_strides,
        q.shape[0], k_cache.shape[0], block_table.shape[0],
        TOPK=logical_indices.shape[1], PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1], GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits, NUM_TILES=num_tiles,
        BLOCK_M=block_m, BLOCK_N=block_n,
        HAS_OUTPUT_GATE=has_output_gate, INDICES_ARE_PHYSICAL=indices_are_physical,
        HAS_VALID_COUNTS=valid_counts is not None,
        num_warps=partial_warps, num_stages=1,
    )
    if num_splits != 1:
        _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
            partial_output, partial_lse, out, gate_pointer,
            out.stride(0), out.stride(1), *gate_strides, q.shape[0],
            HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1],
            NUM_SPLITS=num_splits, BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            HAS_OUTPUT_GATE=has_output_gate,
            num_warps=4, num_stages=1,
        )
    return out


def qsa_store_cache_rows(
    cache: torch.Tensor, slot_mapping: torch.Tensor, rows: torch.Tensor,
) -> None:
    """Write BF16 raw or compressed key rows into vLLM-managed cache pages."""
    if rows.ndim == 3:
        rows = rows[:, 0]
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache, slot_mapping, rows,
        cache.stride(0), cache.stride(1), cache.stride(3),
        rows.stride(0), rows.stride(1),
        rows.shape[0], cache.shape[0],
        PAGE_SIZE=cache.shape[1], WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]), num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,
    raw_positions: torch.Tensor,
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 mean-pool complete groups across the prior ring/current chunk."""
    rows = token_to_req.numel()
    pooled = torch.empty(
        (rows, 1, raw_keys.shape[2]), dtype=raw_keys.dtype, device=raw_keys.device,
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return pooled, first_positions
    load_rope_positions = rope_cache is not None
    if rope_cache is None:
        rope_cache = compressor_state_cache
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys, raw_positions, compressor_state_cache, rope_cache,
        compressor_state_block_table, token_to_req, query_start_loc,
        logical_positions, compressed_slots, pooled, first_positions,
        raw_keys.stride(0), raw_keys.stride(2),
        raw_positions.stride(0), raw_positions.stride(2),
        compressor_state_cache.stride(0), compressor_state_cache.stride(1),
        compressor_state_cache.stride(3),
        rope_cache.stride(0), rope_cache.stride(1), rope_cache.stride(3),
        compressor_state_block_table.stride(0),
        pooled.stride(0), pooled.stride(2),
        first_positions.stride(0), first_positions.stride(1),
        rows, compressor_state_cache.shape[0], query_start_loc.shape[0] - 1,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1],
        COMPRESS_RATIO=compress_ratio, HEAD_DIM=raw_keys.shape[2],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]), num_warps=4,
    )
    return pooled, first_positions


def qsa_select_paged_tokens(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
    *,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
    decode_query_len: int = 1,
    visible_blocks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select with fused CUDA top-k and SM80 BF16 prefill/decode scoring."""
    from .indexer import select_paged_tokens

    return select_paged_tokens(
        q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
        token_topk, compress_ratio, out,
        query_start_loc=query_start_loc, max_query_len=max_query_len,
        max_seq_len=max_seq_len, num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens, decode_query_len=decode_query_len,
        visible_blocks=visible_blocks,
    )
