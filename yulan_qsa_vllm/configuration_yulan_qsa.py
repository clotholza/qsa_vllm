# SPDX-License-Identifier: Apache-2.0
"""HF configuration for the YuLan GDN/QSA checkpoint format."""

from transformers import PretrainedConfig


class YuLanQSAConfig(PretrainedConfig):
    model_type = "yulan_qsa"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=99072,
        padded_vocab_size=None,
        hidden_size=2048,
        intermediate_size=4800,
        num_hidden_layers=56,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
        layer_types=None,
        linear_attention_freq=None,
        moe_layer_freq=None,
        num_experts=192,
        num_experts_per_tok=4,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        shared_expert_gate=True,
        moe_shared_expert_gate=None,
        norm_topk_prob=True,
        moe_router_dtype="fp32",
        moe_router_sqrt_gate=True,
        linear_num_key_heads=8,
        linear_num_value_heads=32,
        linear_key_head_dim=64,
        linear_value_head_dim=64,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
        gated_norm=True,
        gated_norm_lora_rank=16,
        hidden_act="silu",
        initializer_range=0.02,
        attention_bias=True,
        attention_output_gate=True,
        attn_output_gate=None,
        qk_layernorm=False,
        position_embedding_type="none",
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_compress_ratio=4,
        indexer_budget=2048,
        indexer_rope_dim=64,
        indexer_rope_theta=10000000.0,
        qsa_query_chunk_size=32,
        max_position_embeddings=524288,
        training_sequence_length=524288,
        multi_latent_attention=False,
        mtp_num_layers=0,
        use_cache=True,
        pad_token_id=None,
        bos_token_id=None,
        eos_token_id=None,
        tie_word_embeddings=False,
        **kwargs,
    ):
        if layer_types is None:
            frequency = 8 if linear_attention_freq is None else linear_attention_freq
            if isinstance(frequency, int):
                layer_types = [
                    "full_attention" if (i + 1) % frequency == 0 else "linear_attention"
                    for i in range(num_hidden_layers)
                ]
            else:
                layer_types = ["linear_attention" if kind else "full_attention"
                               for kind in frequency]
        self.vocab_size = vocab_size
        self.padded_vocab_size = vocab_size if padded_vocab_size is None else padded_vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_types = list(layer_types)
        self.linear_attention_freq = [int(kind == "linear_attention") for kind in self.layer_types]
        self.moe_layer_freq = list(moe_layer_freq) if moe_layer_freq is not None else [int(i >= 2) for i in range(num_hidden_layers)]
        self.mlp_only_layers = [i for i, kind in enumerate(self.moe_layer_freq) if not kind]
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.shared_expert_gate = shared_expert_gate if moe_shared_expert_gate is None else moe_shared_expert_gate
        self.moe_shared_expert_gate = self.shared_expert_gate
        self.norm_topk_prob = norm_topk_prob
        self.moe_router_dtype = moe_router_dtype
        self.moe_router_sqrt_gate = moe_router_sqrt_gate
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.rms_norm_eps = rms_norm_eps
        self.gated_norm = gated_norm
        self.gated_norm_lora_rank = gated_norm_lora_rank
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.attention_bias = attention_bias
        self.attention_output_gate = attention_output_gate
        self.attn_output_gate = attn_output_gate
        self.qk_layernorm = qk_layernorm
        self.position_embedding_type = position_embedding_type
        self.indexer_n_heads = indexer_n_heads
        self.indexer_kv_heads = indexer_kv_heads
        self.indexer_head_dim = indexer_head_dim
        self.indexer_compress_ratio = indexer_compress_ratio
        self.indexer_budget = indexer_budget
        self.indexer_rope_dim = indexer_rope_dim
        self.indexer_rope_theta = float(indexer_rope_theta)
        self.qsa_query_chunk_size = qsa_query_chunk_size
        self.max_position_embeddings = max_position_embeddings
        self.training_sequence_length = training_sequence_length
        self.multi_latent_attention = multi_latent_attention
        self.mtp_num_layers = mtp_num_layers
        self.use_cache = use_cache
        kwargs.setdefault("architectures", ["YuLanQSAForCausalLM"])
        kwargs.setdefault("auto_map", {
            "AutoConfig": "configuration_yulan_qsa.YuLanQSAConfig",
            "AutoModel": "modeling_yulan_qsa.YuLanQSAModel",
            "AutoModelForCausalLM": "modeling_yulan_qsa.YuLanQSAForCausalLM",
        })
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


YuLanQSAConfig.register_for_auto_class("AutoConfig")
