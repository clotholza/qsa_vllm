# SPDX-License-Identifier: Apache-2.0
"""Register YuLan QSA support in vLLM."""


def register():
    from pathlib import Path

    from transformers import AutoConfig
    from vllm import ModelRegistry
    from vllm.config.compilation import CompilationConfig
    from vllm.tokenizers.registry import _MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS
    from vllm.tool_parsers import ToolParserManager
    from vllm.transformers_utils.chat_templates.registry import (
        register_chat_template_fallback_path,
    )

    from .capacity import install_capacity_observer
    from .configuration_yulan_qsa import YuLanQSAConfig
    from .graph_dispatch import install_short_decode_dispatch

    install_capacity_observer()
    install_short_decode_dispatch()
    AutoConfig.register("yulan_qsa", YuLanQSAConfig, exist_ok=True)
    # Transformers 5's LlamaTokenizer drops the serialized normalizer, changing
    # token IDs after chat delimiters. Load YuLan's tokenizer.json directly.
    _MODEL_TYPES_WITH_INCORRECT_TOKENIZER_CLASS.add(YuLanQSAConfig.model_type)
    ToolParserManager.register_lazy_module(
        "yulan", "yulan_qsa_vllm.tool_parser", "YulanToolParser"
    )
    register_chat_template_fallback_path(
        "yulan_qsa", Path(__file__).with_name("tool_chat_template.jinja")
    )
    # Keep dynamic prefill work outside piecewise graphs, like native QSA/GDN.
    # Uniform decode is still captured by the full-decode CUDA graph path.
    CompilationConfig._attention_ops.append("vllm::yulan_qsa_with_output")
    ModelRegistry.register_model(
        "YuLanQSAForCausalLM", "yulan_qsa_vllm.model:YuLanQSAForCausalLM"
    )
