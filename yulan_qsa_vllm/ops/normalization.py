# SPDX-License-Identifier: Apache-2.0
"""Ordinary-gamma indexer RMSNorm and partial RoPE for SM80.

The casts follow the YuLan training indexer: normalize in FP32, cast to BF16,
multiply the learned gamma in BF16, then apply BF16 cosine/sine products.
"""

from functools import cache

import torch
from torch import nn

from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.triton_utils import tl, triton


@cache
def rotary_cache(max_position, rotary_dim, theta, device):
    channels = torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
    inverse = theta ** (-channels / rotary_dim)
    positions = torch.arange(max_position, dtype=torch.float32, device=device)
    phases = torch.outer(positions, inverse)
    return torch.cat((phases.cos(), phases.sin()), dim=-1).to(torch.bfloat16)


class IndexerRMSNorm(nn.Module):
    def __init__(self, width, epsilon):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width, dtype=torch.bfloat16))
        self.weight.weight_loader = default_weight_loader
        self.variance_epsilon = epsilon


@triton.jit
def _norm_rope(
    X, W, POS, COS_SIN, OUT,
    stride_xt: tl.constexpr, stride_xh: tl.constexpr, stride_xd: tl.constexpr,
    stride_pt: tl.constexpr,
    HEADS: tl.constexpr, DIM: tl.constexpr, ROTARY: tl.constexpr,
    EPS: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    token = row // HEADS
    head = row % HEADS
    offsets = tl.arange(0, BLOCK)
    address = X + token * stride_xt + head * stride_xh
    values = tl.load(address + offsets * stride_xd, offsets < DIM, 0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(values * values, 0) / DIM + EPS)
    gamma = tl.load(W + offsets, offsets < DIM, 0).to(tl.float32)
    normalized = (values * inv_rms).to(tl.bfloat16)
    scaled = (normalized.to(tl.float32) * gamma).to(tl.bfloat16)

    half = ROTARY // 2
    partner_offsets = tl.where(offsets < half, offsets + half, offsets - half)
    partners = tl.load(
        address + partner_offsets * stride_xd, offsets < ROTARY, 0
    ).to(tl.float32)
    partner_gamma = tl.load(W + partner_offsets, offsets < ROTARY, 0).to(tl.float32)
    partner_norm = (partners * inv_rms).to(tl.bfloat16)
    rotated = (partner_norm.to(tl.float32) * partner_gamma).to(tl.bfloat16)
    rotated = tl.where(offsets < half, -rotated, rotated)
    position = tl.load(POS + token * stride_pt).to(tl.int64)
    channel = offsets % half
    cosine = tl.load(COS_SIN + position * ROTARY + channel, offsets < ROTARY, 0)
    sine = tl.load(COS_SIN + position * ROTARY + half + channel, offsets < ROTARY, 0)
    first = (scaled.to(tl.float32) * cosine.to(tl.float32)).to(tl.bfloat16)
    second = (rotated.to(tl.float32) * sine.to(tl.float32)).to(tl.bfloat16)
    result = (first.to(tl.float32) + second.to(tl.float32)).to(tl.bfloat16)
    result = tl.where(offsets < ROTARY, result, scaled)
    tl.store(OUT + row * DIM + offsets, result, offsets < DIM)


def indexer_norm_rope(values, norm, positions, cos_sin, rotary_dim):
    output = torch.empty(values.shape, device=values.device, dtype=torch.bfloat16)
    tokens, heads, dimension = values.shape
    _norm_rope[(tokens * heads,)](
        values, norm.weight, positions, cos_sin, output,
        *values.stride(), positions.stride(0),
        HEADS=heads, DIM=dimension, ROTARY=rotary_dim,
        EPS=norm.variance_epsilon, BLOCK=triton.next_power_of_2(dimension),
        num_warps=4, num_stages=1, enable_fp_fusion=False,
    )
    return output
