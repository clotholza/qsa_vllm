# SPDX-License-Identifier: Apache-2.0
"""Register the independent YuLan QSA architecture in every vLLM worker."""


def register():
    from transformers import AutoConfig
    from vllm import ModelRegistry
    from vllm.config.compilation import CompilationConfig

    from .capacity import install_capacity_observer
    from .configuration_yulan_qsa import YuLanQSAConfig
    from .graph_dispatch import install_short_decode_dispatch

    install_capacity_observer()
    install_short_decode_dispatch()
    AutoConfig.register("yulan_qsa", YuLanQSAConfig, exist_ok=True)
    # Keep dynamic prefill work outside piecewise graphs, like native QSA/GDN.
    # Uniform decode is still captured by the full-decode CUDA graph path.
    CompilationConfig._attention_ops.append("vllm::yulan_qsa_with_output")
    ModelRegistry.register_model(
        "YuLanQSAForCausalLM", "yulan_qsa_vllm.model:YuLanQSAForCausalLM"
    )
