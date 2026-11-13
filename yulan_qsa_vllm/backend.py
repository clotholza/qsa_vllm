# SPDX-License-Identifier: Apache-2.0
"""BF16 paged-GQA cache integration for vLLM 0.30 and SM80 QSA."""

from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.attention import set_default_quant_scales
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.torch_utils import canonicalize_singleton_dim_strides
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImplBase,
    AttentionType,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheLayout, KVCacheSpec

from .metadata import YuLanQSAMetadata, YuLanQSAMetadataBuilder


class YuLanQSAImpl(AttentionImplBase[YuLanQSAMetadata]):
    """Sparse paged attention against the owner's TP-local K/V heads."""

    is_sparse: ClassVar[bool] = True
    supports_dense_mha_prefill: ClassVar[bool] = False
    supports_dcp: ClassVar[bool] = False
    supports_pcp: ClassVar[bool] = False

    def __init__(self, num_heads: int, num_kv_heads: int, head_size: int) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = head_size ** -0.5

    def forward(
        self,
        layer,
        query: torch.Tensor,
        attn_metadata: YuLanQSAMetadata,
        output: torch.Tensor,
        *,
        token_to_req: torch.Tensor,
        output_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from .ops.qsa import qsa_sparse_paged_attention

        num_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = layer.main_kv_views()
        output.zero_()
        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            layer.topk_indices_buffer[:num_tokens],
            attn_metadata.block_table,
            token_to_req[:num_tokens],
            output[:num_tokens],
            output_gate=output_gate[:num_tokens] if output_gate is not None else None,
            use_decode_kernel=attn_metadata.max_query_len == 1,
            indices_are_physical=attn_metadata.max_query_len == 1,
            valid_counts=(layer.topk_counts_buffer[:num_tokens]
                          if attn_metadata.max_query_len == 1 else None),
        )
        return output


class YuLanQSABackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto", "bfloat16"]
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name() -> str:
        return "YULAN_QSA_SM80"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[YuLanQSAImpl]:
        return YuLanQSAImpl

    @staticmethod
    def get_builder_cls() -> type[YuLanQSAMetadataBuilder]:
        return YuLanQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return (KVCacheLayout.BLNHC, KVCacheLayout.BLHNC)


class YuLanQSACacheOwner(AttentionLayerBase):
    """Mixin registering an attention module as its main-cache owner.

    The attention module initializes ``nn.Module`` and its TP-local head counts
    before calling ``_init_main_cache``. vLLM binds its managed logical
    ``[blocks, heads, states, K+V]`` view directly to ``kv_cache``.
    """

    supports_dcp = False

    def _init_main_cache(
        self,
        vllm_config: VllmConfig,
        *,
        num_kv_heads: int,
        head_size: int,
        prefix: str,
    ) -> None:
        self.layer_name = f"{prefix}.attn"
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = "auto"
        self.kv_cache_torch_dtype = torch.bfloat16
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        self.attn_backend = YuLanQSABackend
        self.impl = YuLanQSAImpl(self.num_heads, num_kv_heads, head_size)
        set_default_quant_scales(self, register_buffer=True)
        vllm_config.compilation_config.static_forward_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return YuLanQSABackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size,
            dtype=torch.bfloat16,
        )

    def main_kv_views(self) -> tuple[torch.Tensor, torch.Tensor]:
        key_cache, value_cache = self.kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        return (
            canonicalize_singleton_dim_strides(key_cache),
            canonicalize_singleton_dim_strides(value_cache),
        )

    def update_main_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        key_cache, value_cache = self.main_kv_views()
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            self._k_scale,
            self._v_scale,
        )


__all__ = ["YuLanQSABackend", "YuLanQSACacheOwner", "YuLanQSAImpl"]
