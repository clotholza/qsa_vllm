# Source attribution

This package contains Apache-2.0 code adapted from the vLLM project, tag v0.30.0:

- `ops/indexer.py`: `vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py`.
- `ops/qsa.py`: `vllm/models/qwen4_exp/amd/ops/qsa.py`.
- `metadata.py` and `backend.py`: the QSA cache/attention owner interfaces, with
  ordinary BF16 GQA and the YuLan NoPE projection path.
- `model.py`: the Qwen3Next model, official GDN/MLP/MoE/TP interfaces.

Local changes use BF16 weights and paged caches, SM80 MMA tile sizes, separate
prefill/decode scoring with a bounded workspace, ordinary-gamma indexer norms,
YuLan GatedNorm and sqrt router gates, and the independently registered
`YuLanQSAForCausalLM` checkpoint schema. The raw-cache derived-view lifecycle
follows the corresponding newer upstream cache-owner implementation.

Copyright contributors to the vLLM project. The upstream SPDX notices are
retained in derived kernel and model files. See LICENSE for Apache-2.0 terms.
