# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-launch BF16 indexer preparation and NoPE main-KV writes for SM80.

The request/group work assignment follows vLLM's QSA pre-indexer. The first
compression-group CTA also owns the raw-state commit, so its historical ring
loads finish before that ring is overwritten. Main queries and output gates
remain views of the projection because the YuLan main attention uses NoPE.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _ordinary_norm_rope(x, positions, weights, cos_sin, EPS: tl.constexpr,
                        DIM: tl.constexpr, ROTARY: tl.constexpr):
    """Preserve the training indexer's BF16 rounding at each post-norm step."""
    dims = tl.arange(0, DIM)
    x = x.to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=1) / DIM + EPS)
    gamma = tl.load(weights + dims).to(tl.float32)
    normalized = (x * inv_rms[:, None]).to(tl.bfloat16)
    scaled = (normalized.to(tl.float32) * gamma[None, :]).to(tl.bfloat16)
    half: tl.constexpr = ROTARY // 2
    partner_dims = tl.where(dims < half, dims + half, dims - half)
    partner_dims = tl.broadcast_to(partner_dims[None, :], x.shape)
    partner = tl.gather(scaled, partner_dims, axis=1)
    rotated = tl.where(dims[None, :] < half, -partner, partner).to(tl.bfloat16)
    channel = dims % half
    positions = positions.to(tl.int64)
    cosine = tl.load(
        cos_sin + positions[:, None] * ROTARY + channel[None, :],
        mask=dims[None, :] < ROTARY,
        other=0,
    ).to(tl.float32)
    sine = tl.load(
        cos_sin + positions[:, None] * ROTARY + half + channel[None, :],
        mask=dims[None, :] < ROTARY,
        other=0,
    ).to(tl.float32)
    first = (scaled.to(tl.float32) * cosine).to(tl.bfloat16)
    second = (rotated.to(tl.float32) * sine).to(tl.bfloat16)
    combined = (first.to(tl.float32) + second.to(tl.float32)).to(tl.bfloat16)
    return tl.where(dims[None, :] < ROTARY, combined, scaled)


@triton.jit(do_not_specialize=["num_tokens", "num_k_work"])
def _qsa_prepare_kernel(
    index_qk, positions, q_weight, k_weight, cos_sin, index_q_out,
    state_cache, state_slots, state_table, query_start_loc, logical_positions,
    compressed_cache, compressed_slots, k_work,
    main_key, main_value, main_key_cache, main_value_cache, main_slots,
    stride_index_token, stride_pos, stride_state_block, stride_state_token,
    stride_state_table, stride_compressed_block, stride_compressed_token,
    stride_main_key_token, stride_main_key_head, stride_main_key_dim,
    stride_main_value_token, stride_main_value_head, stride_main_value_dim,
    stride_main_key_cache_block, stride_main_key_cache_token,
    stride_main_key_cache_head, stride_main_key_cache_dim,
    stride_main_value_cache_block, stride_main_value_cache_token,
    stride_main_value_cache_head, stride_main_value_cache_dim,
    num_tokens, num_k_work,
    INDEX_HEADS: tl.constexpr, INDEX_DIM: tl.constexpr,
    ROTARY: tl.constexpr, EPS: tl.constexpr, COMPRESS_RATIO: tl.constexpr,
    STATE_SIZE: tl.constexpr, COMP_PAGE_SIZE: tl.constexpr,
    MAIN_KV_HEADS: tl.constexpr, MAIN_DIM: tl.constexpr,
    MAIN_PAGE_SIZE: tl.constexpr, TILE_T_Q: tl.constexpr, TILE_H_Q: tl.constexpr,
):
    pid = tl.program_id(0)
    num_q_work = tl.cdiv(num_tokens, TILE_T_Q) * tl.cdiv(INDEX_HEADS, TILE_H_Q)
    num_index_work = num_k_work + num_q_work

    if pid >= num_index_work:
        # Main K/V are already BF16, biased and in the final NoPE coordinates.
        work = pid - num_index_work
        token = (work // MAIN_KV_HEADS).to(tl.int64)
        head = work % MAIN_KV_HEADS
        slot = tl.load(main_slots + token).to(tl.int64)
        dims = tl.arange(0, MAIN_DIM)
        if slot >= 0:
            key = tl.load(main_key + token * stride_main_key_token
                          + head * stride_main_key_head + dims * stride_main_key_dim)
            value = tl.load(main_value + token * stride_main_value_token
                            + head * stride_main_value_head + dims * stride_main_value_dim)
            block, offset = slot // MAIN_PAGE_SIZE, slot % MAIN_PAGE_SIZE
            tl.store(main_key_cache + block * stride_main_key_cache_block
                     + offset * stride_main_key_cache_token
                     + head * stride_main_key_cache_head + dims * stride_main_key_cache_dim,
                     key)
            tl.store(main_value_cache + block * stride_main_value_cache_block
                     + offset * stride_main_value_cache_token
                     + head * stride_main_value_cache_head + dims * stride_main_value_cache_dim,
                     value)
        return

    if pid >= num_k_work:
        work = pid - num_k_work
        head_tiles: tl.constexpr = tl.cdiv(INDEX_HEADS, TILE_H_Q)
        token_tile, head_tile = work // head_tiles, work % head_tiles
        row = tl.arange(0, TILE_T_Q * TILE_H_Q)
        tokens = token_tile * TILE_T_Q + row // TILE_H_Q
        heads = head_tile * TILE_H_Q + row % TILE_H_Q
        valid = (tokens < num_tokens) & (heads < INDEX_HEADS)
        dims = tl.arange(0, INDEX_DIM)
        query = tl.load(
            index_qk + tokens[:, None].to(tl.int64) * stride_index_token
            + heads[:, None] * INDEX_DIM + dims[None, :],
            mask=valid[:, None], other=0,
        )
        pos = tl.load(positions + tokens * stride_pos, tokens < num_tokens, 0)
        result = _ordinary_norm_rope(query, pos, q_weight, cos_sin, EPS, INDEX_DIM, ROTARY)
        tl.store(
            index_q_out + (tokens[:, None].to(tl.int64) * INDEX_HEADS
                           + heads[:, None]) * INDEX_DIM + dims[None, :],
            result, mask=valid[:, None],
        )
        return

    metadata = tl.load(k_work + pid * 2 + tl.arange(0, 2))
    request, work_in_request = tl.split(metadata)
    if request < 0:
        return
    query_start = tl.load(query_start_loc + request)
    query_end = tl.load(query_start_loc + request + 1)
    query_len = query_end - query_start
    chunk_end = tl.load(logical_positions + query_end - 1)
    chunk_start = chunk_end - query_len + 1
    num_groups = (chunk_end + 1) // COMPRESS_RATIO - chunk_start // COMPRESS_RATIO
    dims = tl.arange(0, INDEX_DIM)
    key_offset: tl.constexpr = INDEX_HEADS * INDEX_DIM

    if work_in_request < num_groups:
        first_boundary = ((chunk_start + COMPRESS_RATIO) // COMPRESS_RATIO) * COMPRESS_RATIO - 1
        end_position = first_boundary + work_in_request * COMPRESS_RATIO
        boundary_token = query_start + end_position - chunk_start
        compressed_slot = tl.load(compressed_slots + boundary_token).to(tl.int64)
        state_block = tl.load(state_table + request * stride_state_table).to(tl.int64)
        accumulator = tl.zeros((INDEX_DIM,), tl.float32)
        # Only work item zero can need previous-chunk raw keys. It alone commits
        # the suffix below; all other group CTAs read this step's projection.
        for member in tl.static_range(0, COMPRESS_RATIO):
            source_position = end_position - (COMPRESS_RATIO - 1) + member
            source_in_chunk = source_position >= chunk_start
            source_token = query_start + source_position - chunk_start
            current = index_qk + source_token.to(tl.int64) * stride_index_token + key_offset
            cached = state_cache + state_block * stride_state_block + (source_position % STATE_SIZE) * stride_state_token
            source = tl.where(source_in_chunk, current, cached)
            value = tl.load(source + dims).to(tl.float32)
            accumulator += value
        pooled = (accumulator / COMPRESS_RATIO).to(tl.bfloat16)
        first_position = end_position - (COMPRESS_RATIO - 1)
        compressed = _ordinary_norm_rope(
            tl.reshape(pooled, (1, INDEX_DIM)),
            first_position + tl.arange(0, 1), k_weight, cos_sin,
            EPS, INDEX_DIM, ROTARY,
        )
        if compressed_slot >= 0:
            tl.store(compressed_cache
                     + (compressed_slot // COMP_PAGE_SIZE) * stride_compressed_block
                     + (compressed_slot % COMP_PAGE_SIZE) * stride_compressed_token + dims,
                     tl.reshape(compressed, (INDEX_DIM,)))

    if work_in_request == 0:
        # Finish the first group's historical loads before reusing ring slots.
        tl.debug_barrier()
        suffix_length = tl.minimum(query_len, STATE_SIZE)
        for offset in tl.range(0, suffix_length):
            token = (query_end - suffix_length + offset).to(tl.int64)
            slot = tl.load(state_slots + token).to(tl.int64)
            if slot >= 0:
                value = tl.load(index_qk + token * stride_index_token + key_offset + dims)
                tl.store(state_cache + (slot // STATE_SIZE) * stride_state_block
                         + (slot % STATE_SIZE) * stride_state_token + dims, value)


def qsa_prepare(
    index_qk: torch.Tensor,
    positions: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    raw_metadata,
    compressed_metadata,
    state_cache: torch.Tensor,
    compressed_cache: torch.Tensor,
    main_key: torch.Tensor,
    main_value: torch.Tensor,
    main_key_cache: torch.Tensor,
    main_value_cache: torch.Tensor,
    main_slots: torch.Tensor,
    *,
    num_index_heads: int,
    index_head_dim: int,
    rotary_dim: int,
    compress_ratio: int,
) -> torch.Tensor:
    """Prepare the replicated indexer and write TP-local NoPE main K/V."""
    tokens = index_qk.shape[0]
    query = index_qk.new_empty(tokens, num_index_heads, index_head_dim)
    tile_t, tile_h = 2, 2
    k_work = compressed_metadata.k_work_metadata
    q_work = triton.cdiv(tokens, tile_t) * triton.cdiv(num_index_heads, tile_h)
    main_work = tokens * main_key.shape[1]
    _qsa_prepare_kernel[(k_work.shape[0] + q_work + main_work,)](
        index_qk, positions, q_norm_weight, k_norm_weight, cos_sin_cache, query,
        state_cache, raw_metadata.slot_mapping, raw_metadata.block_table,
        raw_metadata.query_start_loc, raw_metadata.logical_positions,
        compressed_cache, compressed_metadata.slot_mapping, k_work,
        main_key, main_value, main_key_cache, main_value_cache, main_slots,
        index_qk.stride(0), positions.stride(0), state_cache.stride(0),
        state_cache.stride(1), raw_metadata.block_table.stride(0),
        compressed_cache.stride(0), compressed_cache.stride(1),
        *main_key.stride(), *main_value.stride(),
        *main_key_cache.stride(), *main_value_cache.stride(),
        tokens, k_work.shape[0],
        INDEX_HEADS=num_index_heads, INDEX_DIM=index_head_dim,
        ROTARY=rotary_dim, EPS=eps, COMPRESS_RATIO=compress_ratio,
        STATE_SIZE=state_cache.shape[1], COMP_PAGE_SIZE=compressed_cache.shape[1],
        MAIN_KV_HEADS=main_key.shape[1], MAIN_DIM=main_key.shape[2],
        MAIN_PAGE_SIZE=main_key_cache.shape[1], TILE_T_Q=tile_t, TILE_H_Q=tile_h,
        num_warps=1, num_stages=1, enable_fp_fusion=False,
    )
    return query


__all__ = ["qsa_prepare"]
