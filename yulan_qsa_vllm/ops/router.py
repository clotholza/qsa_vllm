# SPDX-License-Identifier: Apache-2.0
"""FP32 router logits and fused softmax/top-k/square-root routing."""

import torch
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


# Populated from the manual SM80 graph benchmark, keyed by decode batch size.
# Larger batches continue to use BF16 GEMM with an FP32 output.
_SM80_GEMV_PROFILES = {1: (1, 4), 2: (1, 4), 4: (2, 4), 8: (2, 4)}


@triton.jit
def _router_gemv_kernel(
    hidden_ptr, weight_ptr, logits_ptr,
    stride_hidden_row, stride_hidden_dim, stride_weight_row, stride_weight_dim,
    EXPERTS: tl.constexpr, HIDDEN: tl.constexpr,
    BLOCK_E: tl.constexpr, BLOCK_D: tl.constexpr,
):
    row = tl.program_id(1)
    experts = tl.program_id(0) * BLOCK_E + tl.arange(0, BLOCK_E)
    dims = tl.arange(0, BLOCK_D)
    hidden = tl.load(
        hidden_ptr + row * stride_hidden_row + dims * stride_hidden_dim,
        mask=dims < HIDDEN, other=0.0,
    ).to(tl.float32)
    weights = tl.load(
        weight_ptr + experts[:, None] * stride_weight_row
        + dims[None, :] * stride_weight_dim,
        mask=(experts[:, None] < EXPERTS) & (dims[None, :] < HIDDEN), other=0.0,
    ).to(tl.float32)
    logits = tl.sum(weights * hidden[None, :], axis=1)
    tl.store(logits_ptr + row * EXPERTS + experts, logits, experts < EXPERTS)


def router_gemv(hidden_states, weight, expert_tile, num_warps):
    logits = torch.empty(
        (hidden_states.shape[0], weight.shape[0]),
        dtype=torch.float32, device=hidden_states.device,
    )
    _router_gemv_kernel[(triton.cdiv(weight.shape[0], expert_tile), hidden_states.shape[0])](
        hidden_states, weight, logits,
        *hidden_states.stride(), *weight.stride(),
        EXPERTS=weight.shape[0], HIDDEN=weight.shape[1], BLOCK_E=expert_tile,
        BLOCK_D=triton.next_power_of_2(weight.shape[1]),
        num_warps=num_warps, num_stages=1,
    )
    return logits


def _router_logits_impl(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    profile = _SM80_GEMV_PROFILES.get(triton.next_power_of_2(hidden_states.shape[0]))
    if profile is not None:
        return router_gemv(hidden_states, weight, *profile)
    return torch.mm(hidden_states, weight.T, out_dtype=torch.float32)


def _router_logits_fake(hidden_states: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.empty(
        (hidden_states.shape[0], weight.shape[0]),
        device=hidden_states.device, dtype=torch.float32,
    )


direct_register_custom_op(
    op_name="yulan_router_logits",
    op_func=_router_logits_impl,
    fake_impl=_router_logits_fake,
    mutates_args=[],
)


def router_logits(hidden_states, weight):
    """Keep small-M dispatch opaque to the backbone's dynamic compilation."""
    return torch.ops.vllm.yulan_router_logits(hidden_states, weight)


@triton.jit
def _argmax_pair(value_left, id_left, value_right, id_right):
    take_left = (id_right < 0) | (
        (id_left >= 0)
        & ((value_left > value_right)
           | ((value_left == value_right) & (id_left < id_right)))
    )
    return tl.where(take_left, value_left, value_right), tl.where(take_left, id_left, id_right)


@triton.jit
def _route_kernel(LOGITS, WEIGHTS, INDICES, stride_logits,
                  ROWS: tl.constexpr, EXPERTS: tl.constexpr,
                  TOPK: tl.constexpr, RENORMALIZE: tl.constexpr,
                  SQRT_GATE: tl.constexpr, BLOCK_E: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    experts = tl.arange(0, BLOCK_E)
    values = tl.load(LOGITS + row * stride_logits + experts,
                     mask=experts < EXPERTS, other=-float('inf')).to(tl.float32)
    probabilities = tl.exp(values - tl.max(values, 0))
    probabilities /= tl.sum(probabilities, 0)
    ranks = tl.arange(0, BLOCK_K)
    selected_weights = tl.full((BLOCK_K,), 0.0, tl.float32)
    selected_indices = tl.full((BLOCK_K,), 0, tl.int32)
    candidate_ids = tl.where(experts < EXPERTS, experts, -1)
    for rank in tl.static_range(TOPK):
        maximum, index = tl.reduce((probabilities, candidate_ids), 0, _argmax_pair)
        selected_weights = tl.where(ranks == rank, maximum, selected_weights)
        selected_indices = tl.where(ranks == rank, index, selected_indices)
        candidate_ids = tl.where(experts == index, -1, candidate_ids)
    if RENORMALIZE:
        selected_weights /= tl.sum(selected_weights, 0)
    if SQRT_GATE:
        selected_weights = tl.sqrt(selected_weights)
    tl.store(WEIGHTS + row * TOPK + ranks, selected_weights, ranks < TOPK)
    tl.store(INDICES + row * TOPK + ranks, selected_indices, ranks < TOPK)


def route_experts(logits, topk, renormalize, sqrt_gate):
    rows, experts = logits.shape
    weights = torch.empty((rows, topk), dtype=torch.float32, device=logits.device)
    indices = torch.empty((rows, topk), dtype=torch.int32, device=logits.device)
    _route_kernel[(rows,)](
        logits, weights, indices, logits.stride(0), ROWS=rows, EXPERTS=experts,
        TOPK=topk, RENORMALIZE=renormalize, SQRT_GATE=sqrt_gate,
        BLOCK_E=triton.next_power_of_2(experts), BLOCK_K=triton.next_power_of_2(topk),
        num_warps=4,
    )
    return weights, indices
