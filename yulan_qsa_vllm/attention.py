# SPDX-License-Identifier: Apache-2.0
"""YuLan NoPE GQA/QSA with managed BF16 caches and SM80 Triton kernels."""

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear, RowParallelLinear,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper
from vllm.utils.torch_utils import (
    LayerNameType, _encode_layer_name, _resolve_layer_name,
    direct_register_custom_op,
)

from .backend import YuLanQSACacheOwner
from .cache import QSACompressedKeyCache, QSAKeyStateCache
from .ops.indexer import select_paged_tokens as qsa_select_paged_tokens
from .ops.normalization import IndexerRMSNorm, rotary_cache
from .ops.prepare import qsa_prepare


class YuLanQSAQKVIndexerLinear(MergedColumnParallelLinear):
    """One BF16 GEMM for TP-local Q/gate/K/V and replicated indexer Q/K."""

    def __init__(self, config, prefix):
        tp_size = get_tensor_model_parallel_world_size()
        query_factor = 2 if config.attention_output_gate else 1
        self.index_size = (config.indexer_n_heads + 1) * config.indexer_head_dim
        super().__init__(
            input_size=config.hidden_size,
            output_sizes=[
                config.num_attention_heads * config.head_dim * query_factor,
                config.num_key_value_heads * config.head_dim,
                config.num_key_value_heads * config.head_dim,
                self.index_size * tp_size,
            ],
            bias=config.attention_bias, params_dtype=torch.bfloat16,
            quant_config=None, prefix=prefix,
        )
        # The indexer has no bias; its slice shares the main projection GEMM.
        if self.bias is not None:
            self.bias.data[-self.index_size:].zero_()

    def _load_shard(self, loader, param, loaded_weight, loaded_shard_id):
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == 3:
            self.tp_rank = 0
            if param_tp_rank is not None:
                param.tp_rank = 0
        try:
            loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None):
        self._load_shard(super().weight_loader, param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None):
        self._load_shard(super().weight_loader_v2, param, loaded_weight, loaded_shard_id)

    def load_weights(self, weights):
        def remap_shards():
            for name, weight in weights:
                shard_id = getattr(weight, "shard_id", None)
                if isinstance(shard_id, str):
                    weight = weight.detach()
                    weight.shard_id = {"q": 0, "k": 1, "v": 2}[shard_id]
                yield name, weight
        return super().load_weights(remap_shards())


class YuLanQSAIndexer(nn.Module):
    """Replicated indexer weights and vLLM-managed raw/compressed BF16 state."""

    def __init__(self, vllm_config, config, prefix):
        super().__init__()
        self.num_heads = config.indexer_n_heads
        self.head_dim = config.indexer_head_dim
        self.compress_ratio = config.indexer_compress_ratio
        self.token_topk = config.indexer_budget
        self.rotary_dim = config.indexer_rope_dim
        self.q_layernorm = IndexerRMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_layernorm = IndexerRMSNorm(self.head_dim, config.rms_norm_eps)
        self.register_buffer(
            "cos_sin_cache",
            rotary_cache(
                config.max_position_embeddings, self.rotary_dim,
                float(config.indexer_rope_theta), self.q_layernorm.weight.device,
            ),
            persistent=False,
        )
        self.raw_key_cache = QSAKeyStateCache(
            head_size=self.head_dim, dtype=torch.bfloat16,
            cache_rope_positions=False, prefix=f"{prefix}.raw_key_cache",
            cache_config=vllm_config.cache_config, compress_ratio=self.compress_ratio,
            vllm_config=vllm_config,
        )
        self.compressed_key_cache = QSACompressedKeyCache(
            head_size=self.head_dim, dtype=torch.bfloat16,
            prefix=f"{prefix}.compressed_key_cache",
            cache_config=vllm_config.cache_config, compress_ratio=self.compress_ratio,
            vllm_config=vllm_config,
        )

    @property
    def output_width(self):
        return self.token_topk + self.compress_ratio - 1

    def forward(self, projected, positions, raw_metadata, compressed_metadata, out,
                main_key, main_value, main_key_cache, main_value_cache, main_slots,
                *, main_block_table=None, valid_counts_out=None):
        query = qsa_prepare(
            projected, positions, self.q_layernorm.weight, self.k_layernorm.weight,
            self.cos_sin_cache, self.q_layernorm.variance_epsilon,
            raw_metadata, compressed_metadata,
            self.raw_key_cache.key_cache, self.compressed_key_cache.kv_cache,
            main_key, main_value, main_key_cache, main_value_cache, main_slots,
            num_index_heads=self.num_heads, index_head_dim=self.head_dim,
            rotary_dim=self.rotary_dim, compress_ratio=self.compress_ratio,
        )
        return qsa_select_paged_tokens(
            query, self.compressed_key_cache.kv_cache,
            compressed_metadata.block_table, compressed_metadata.token_to_req,
            compressed_metadata.logical_positions, compressed_metadata.seq_lens,
            self.token_topk, self.compress_ratio, out,
            query_start_loc=compressed_metadata.query_start_loc,
            max_query_len=compressed_metadata.max_query_len,
            max_seq_len=compressed_metadata.max_seq_len,
            num_decodes=compressed_metadata.num_decodes,
            num_decode_tokens=compressed_metadata.num_decode_tokens,
            decode_query_len=compressed_metadata.decode_query_len,
            visible_blocks=compressed_metadata.visible_blocks,
            main_block_table=main_block_table, main_page_size=main_key_cache.shape[1],
            valid_counts_out=valid_counts_out,
        )


class YuLanQSAAttention(nn.Module, YuLanQSACacheOwner):
    def __init__(self, vllm_config: VllmConfig, config, prefix):
        nn.Module.__init__(self)
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = config.num_attention_heads // tp_size
        self.num_kv_heads = config.num_key_value_heads // tp_size
        self.head_dim = config.head_dim
        self.attention_output_gate = config.attention_output_gate
        query_factor = 2 if self.attention_output_gate else 1
        self.q_size = self.num_heads * self.head_dim * query_factor
        self.kv_size = self.num_kv_heads * self.head_dim
        # HF q_proj is head-interleaved [Q,gate]. The packed q shard therefore
        # contains twice the ordinary number of head-width rows when gated.
        self.qkv_proj = YuLanQSAQKVIndexerLinear(config, f"{prefix}.qkv_proj")
        self.o_proj = RowParallelLinear(
            config.num_attention_heads * self.head_dim, config.hidden_size,
            bias=False, input_is_parallel=True, params_dtype=torch.bfloat16,
            quant_config=None, prefix=f"{prefix}.o_proj",
        )
        self.indexer = YuLanQSAIndexer(vllm_config, config, f"{prefix}.indexer")
        self._init_main_cache(
            vllm_config, num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim, prefix=prefix,
        )
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                vllm_config.scheduler_config.max_num_batched_tokens,
                self.indexer.output_width, dtype=torch.int32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "topk_counts_buffer",
            torch.empty(vllm_config.scheduler_config.max_num_batched_tokens,
                        dtype=torch.int32),
            persistent=False,
        )

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(
            weights, mapper=WeightsMapper(orig_to_new_stacked={
                "indexer.index_qk_proj.": ("qkv_proj.", 3),
            }),
        )

    def _run_qsa(self, projected, positions, query, key, value, gate, output):
        metadata = get_forward_context().attn_metadata
        # vLLM's memory-profile forward runs without cache metadata.
        if isinstance(metadata, list):
            metadata = metadata[0]
        if metadata is None:
            output.zero_()
            return
        main = metadata[self.layer_name]
        raw = metadata[self.indexer.raw_key_cache.prefix]
        compressed = metadata[self.indexer.compressed_key_cache.prefix]
        count = main.num_actual_tokens
        is_decode = main.max_query_len == 1
        key_cache, value_cache = self.main_kv_views()
        self.indexer(
            projected[:count], positions[:count], raw, compressed,
            self.topk_indices_buffer[:count],
            key[:count], value[:count], key_cache, value_cache,
            main.slot_mapping[:count],
            main_block_table=main.block_table if is_decode else None,
            valid_counts_out=self.topk_counts_buffer[:count] if is_decode else None,
        )
        self.impl.forward(self, query, main, output, token_to_req=raw.token_to_req,
                          output_gate=gate)

    def forward(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value, projected = qkv.split(
            (self.q_size, self.kv_size, self.kv_size, self.qkv_proj.index_size), dim=-1
        )
        tokens = hidden_states.shape[0]
        if self.attention_output_gate:
            query, gate = query.reshape(tokens, self.num_heads, 2 * self.head_dim).chunk(2, -1)
        else:
            query = query.reshape(tokens, self.num_heads, self.head_dim)
            gate = None
        key = key.reshape(tokens, self.num_kv_heads, self.head_dim)
        value = value.reshape(tokens, self.num_kv_heads, self.head_dim)
        core_output = torch.empty_like(query, memory_format=torch.contiguous_format)
        torch.ops.vllm.yulan_qsa_with_output(
            projected, positions, query, key, value, gate, core_output,
            _encode_layer_name(self.layer_name),
        )
        output, _ = self.o_proj(core_output.reshape(tokens, -1))
        return output


def yulan_qsa_with_output(
    projected: torch.Tensor, positions: torch.Tensor,
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    gate: torch.Tensor | None,
    output: torch.Tensor, layer_name: LayerNameType,
) -> None:
    layer = get_forward_context().no_compile_layers[_resolve_layer_name(layer_name)]
    layer._run_qsa(projected, positions, query, key, value, gate, output)


def _qsa_fake(
    projected: torch.Tensor, positions: torch.Tensor,
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
    gate: torch.Tensor | None,
    output: torch.Tensor, layer_name: LayerNameType,
) -> None:
    pass


direct_register_custom_op(
    op_name="yulan_qsa_with_output", op_func=yulan_qsa_with_output,
    mutates_args=["output"], fake_impl=_qsa_fake,
)
