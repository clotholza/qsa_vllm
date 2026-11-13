# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""YuLan GatedNorm GDN/QSA model for the vLLM 0.30.0 execution interfaces.

GDN, tensor-parallel projections, fused experts, vocabulary projections and
hybrid-state management use the official vLLM implementations. QSA attention
is provided by the local paged BF16 backend. Layer placement comes from the
exported checkpoint's explicit layer lists.
"""

from collections.abc import Iterable
from functools import partial
from itertools import islice

import torch
from torch import nn
from torch.nn import functional as F

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group, get_pp_group
from vllm.model_executor.layers.fused_moe import FusedMoEFactory
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsPP,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)
from vllm.sequence import IntermediateTensors

from .attention import YuLanQSAAttention
from .ops.router import route_experts, router_logits


class YuLanGatedNorm(nn.Module):
    """Ordinary-gamma RMSNorm followed by its replicated low-rank gate.

    The RMSNorm residual form keeps vLLM's fused residual-add/norm operation.
    Projection names match the exported checkpoint, including gate_up.bias.
    """

    def __init__(self, config, prefix: str):
        super().__init__()
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gate_down = ReplicatedLinear(
            config.hidden_size,
            config.gated_norm_lora_rank,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate_down",
        )
        self.gate_up = ReplicatedLinear(
            config.gated_norm_lora_rank,
            config.hidden_size,
            bias=True,
            quant_config=None,
            prefix=f"{prefix}.gate_up",
        )

    def forward(self, hidden_states, residual=None):
        if residual is None:
            normalized = self.norm(hidden_states)
        else:
            normalized, residual = self.norm(hidden_states, residual)
        down, _ = self.gate_down(normalized)
        up, _ = self.gate_up(F.silu(down))
        gated = normalized * up.sigmoid()
        if residual is None:
            return gated
        return gated, residual


def _residual_norm(config, prefix: str):
    if config.gated_norm:
        return YuLanGatedNorm(config, prefix)
    return RMSNorm(config.hidden_size, eps=config.rms_norm_eps)


def _route_experts(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    *,
    sqrt_gate: bool,
):
    """Compute YuLan routing and its square-root epilogue in one launch."""
    return route_experts(gating_output, topk, renormalize, sqrt_gate)


class YuLanSparseMoeBlock(nn.Module):
    """TP-sharded official fused experts with an FP32 router computation.

    Router parameters remain BF16 with the other main weights. As in training,
    only the router GEMM and routing probabilities use FP32. The shared expert
    runs through vLLM's shared-expert runner, retaining one combined TP reduce.
    """

    def __init__(self, vllm_config: VllmConfig, config, prefix: str):
        super().__init__()
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )
        self.shared_expert_gate = (
            ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=f"{prefix}.shared_expert_gate",
            )
            if config.shared_expert_gate and config.shared_expert_intermediate_size
            else None
        )
        self.shared_expert = (
            Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=None,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                prefix=f"{prefix}.shared_expert",
            )
            if config.shared_expert_intermediate_size
            else None
        )
        parallel_config = vllm_config.parallel_config
        self.experts = FusedMoEFactory(
            num_experts=config.num_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=config.norm_topk_prob,
            quant_config=None,
            prefix=f"{prefix}.experts",
            custom_routing_function=partial(
                _route_experts, sqrt_gate=config.moe_router_sqrt_gate
            ),
            enable_eplb=parallel_config.enable_eplb,
            num_redundant_experts=parallel_config.eplb_config.num_redundant_experts,
            shared_experts=self.shared_expert,
            reduce_results=True,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = router_logits(hidden_states, self.gate.weight)
        return self.experts(hidden_states=hidden_states, router_logits=logits)


class YuLanQSADecoderLayer(nn.Module):
    def __init__(self, vllm_config: VllmConfig, config, prefix: str):
        super().__init__()
        self.layer_idx = extract_layer_index(prefix)
        self.layer_type = config.layer_types[self.layer_idx]
        self.input_layernorm = _residual_norm(config, f"{prefix}.input_layernorm")
        self.post_attention_layernorm = _residual_norm(
            config, f"{prefix}.post_attention_layernorm"
        )
        if self.layer_type == "linear_attention":
            self.linear_attn = QwenGatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=True,
            )
            # These two decay parameters are exported in FP32 by the bridge.
            self.linear_attn.dt_bias.data = self.linear_attn.dt_bias.data.float()
        else:
            self.self_attn = YuLanQSAAttention(
                vllm_config=vllm_config,
                config=config,
                prefix=f"{prefix}.self_attn",
            )
        if config.moe_layer_freq[self.layer_idx]:
            self.mlp = YuLanSparseMoeBlock(vllm_config, config, f"{prefix}.mlp")
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=None,
                prefix=f"{prefix}.mlp",
            )

    def forward(self, hidden_states, residual, positions):
        if residual is None:
            residual = hidden_states
            normalized = self.input_layernorm(hidden_states)
        else:
            normalized, residual = self.input_layernorm(hidden_states, residual)
        if self.layer_type == "linear_attention":
            hidden_states = self.linear_attn(hidden_states=normalized)
        else:
            hidden_states = self.self_attn(positions, normalized)
        normalized, residual = self.post_attention_layernorm(hidden_states, residual)
        return self.mlp(normalized), residual


@support_torch_compile
class YuLanQSAModel(nn.Module):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_stacked={
            ".q_proj": (".qkv_proj", "q"),
            ".k_proj": (".qkv_proj", "k"),
            ".v_proj": (".qkv_proj", "v"),
            ".mlp.gate_proj": (".mlp.gate_up_proj", 0),
            ".mlp.up_proj": (".mlp.gate_up_proj", 1),
            ".shared_expert.gate_proj": (".shared_expert.gate_up_proj", 0),
            ".shared_expert.up_proj": (".shared_expert.gate_up_proj", 1),
        }
    )

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_text_config
        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            self.config.num_hidden_layers,
            lambda prefix: YuLanQSADecoderLayer(vllm_config, self.config, prefix),
            prefix=f"{prefix}.layers",
        )
        self.norm = (
            _residual_norm(self.config, f"{prefix}.norm")
            if get_pp_group().is_last_rank
            else PPMissingLayer()
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            hidden_states = (
                inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
            )
            residual = None
        else:
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(hidden_states, residual, positions)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights, mapper=self.hf_to_vllm_mapper)


class YuLanQSAForCausalLM(nn.Module, HasInnerState, SupportsPP, MixtureOfExperts, IsHybrid):
    """Independent architecture registered for the exported YuLan checkpoint."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkvz"],
        "in_proj_ba": ["in_proj_ba"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_text_config
        self.model_config = vllm_config.model_config
        self.scheduler_config = vllm_config.scheduler_config
        self.quant_config = None
        self.model = YuLanQSAModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(self.config.vocab_size)
        self.make_empty_intermediate_tensors = self.model.make_empty_intermediate_tensors
        self.moe_layers = [
            layer.mlp.experts
            for layer in self.model.layers
            if isinstance(layer, YuLanQSADecoderLayer)
            and isinstance(layer.mlp, YuLanSparseMoeBlock)
        ]
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = self.config.num_experts
        self.num_redundant_experts = (
            vllm_config.parallel_config.eplb_config.num_redundant_experts
        )
        self.num_physical_experts = self.num_logical_experts + self.num_redundant_experts
        self.num_local_physical_experts = (
            self.num_physical_experts // get_ep_group().world_size
        )
        self.num_routed_experts = self.config.num_experts
        self.expert_weights = []

    def update_physical_experts_metadata(
        self, num_physical_experts: int, num_local_physical_experts: int
    ) -> None:
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for experts in self.moe_layers:
            experts.update_expert_map()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config: VllmConfig):
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config: VllmConfig):
        config = vllm_config.model_config.hf_text_config
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            vllm_config.parallel_config.tensor_parallel_size,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
            config.linear_key_head_dim,
            config.linear_value_head_dim,
            config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls):
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        return AutoWeightsLoader(self).load_weights(weights)
