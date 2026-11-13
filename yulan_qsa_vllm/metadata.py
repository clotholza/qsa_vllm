# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned metadata for YuLan paged QSA.

The official side-cache builder owns logical positions, compression-boundary
slots, token-to-request mapping and cross-chunk compression work. The main
attention needs only its page table and write slots; it does not run dense
FlashAttention planning for queries selected by the QSA indexer.
"""

from dataclasses import dataclass, replace
from typing import ClassVar

import torch

from vllm.config import VllmConfig
from vllm.models.qwen4_exp.common.qsa_cache import (
    QSAForwardMetadata,
    QSAMetadataBuilder,
)
from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadata,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec


@dataclass
class YuLanQSAMetadata(AttentionMetadata):
    num_actual_tokens: int
    max_query_len: int
    block_table: torch.Tensor
    slot_mapping: torch.Tensor


class YuLanQSAMetadataBuilder(AttentionMetadataBuilder[YuLanQSAMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH
    supports_update_block_table = True
    supports_draft_decode_metadata_update = True

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> YuLanQSAMetadata:
        return YuLanQSAMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
        )

    def update_block_table(
        self,
        metadata: YuLanQSAMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> YuLanQSAMetadata:
        return replace(metadata, block_table=blk_table, slot_mapping=slot_mapping)

    def update_draft_decode_metadata(self, metadata: YuLanQSAMetadata) -> None:
        pass


__all__ = [
    "QSAForwardMetadata",
    "QSAMetadataBuilder",
    "YuLanQSAMetadata",
    "YuLanQSAMetadataBuilder",
]
