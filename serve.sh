#!/usr/bin/env bash
set -euo pipefail
qsa_runtime_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${QSA_RUNTIME_ENV:-${qsa_runtime_root}/env_paths.sh}"

export PYTHONPATH="${qsa_runtime_root}:${qsa_runtime_root}/plugin_site${PYTHONPATH:+:${PYTHONPATH}}"
export VLLM_PLUGINS=yulan_qsa
export CUDA_VISIBLE_DEVICES="${QSA_CUDA_VISIBLE_DEVICES}"
export TORCH_CUDA_ARCH_LIST=8.0
export VLLM_GDN_DECODE_KERNEL=triton
export QSA_INDEXER_WORKSPACE_MB
export VLLM_TUNED_CONFIG_FOLDER="${VLLM_TUNED_CONFIG_FOLDER:-${qsa_runtime_root}/configs/moe_sm80}"
export TOKENIZERS_PARALLELISM=false

qsa_optional_args=()
qsa_mamba_cache_mode=none
if [[ "${QSA_TP_COMM}" == shm ]]; then
    export NCCL_P2P_DISABLE=1
    export NCCL_IB_DISABLE=1
    export NCCL_CUMEM_ENABLE=0
    export NCCL_CUMEM_HOST_ENABLE=0
    qsa_optional_args+=(--disable-custom-all-reduce)
fi
if [[ "${QSA_ENABLE_PREFIX_CACHING}" == true ]]; then
    qsa_optional_args+=(--enable-prefix-caching)
    qsa_mamba_cache_mode=align
else
    qsa_optional_args+=(--no-enable-prefix-caching)
fi
if [[ "${QSA_ASYNC_SCHEDULING}" == true ]]; then
    qsa_optional_args+=(--async-scheduling)
else
    qsa_optional_args+=(--no-async-scheduling)
fi
if [[ "${QSA_SCHEDULER_RESERVE_FULL_ISL}" == true ]]; then
    qsa_optional_args+=(--scheduler-reserve-full-isl)
else
    qsa_optional_args+=(--no-scheduler-reserve-full-isl)
fi
if [[ -n "${QSA_KV_CACHE_MEMORY_BYTES}" ]]; then
    qsa_optional_args+=(--kv-cache-memory-bytes "${QSA_KV_CACHE_MEMORY_BYTES}")
fi
if [[ -n "${QSA_CHAT_TEMPLATE}" ]]; then
    qsa_optional_args+=(--chat-template "${QSA_CHAT_TEMPLATE}")
fi
if [[ "${QSA_TRUST_REQUEST_CHAT_TEMPLATE}" == true ]]; then
    qsa_optional_args+=(--trust-request-chat-template)
fi

exec "${QSA_PYTHON_BIN:-python3}" -m vllm.entrypoints.cli.main serve "${QSA_MODEL_DIR}" \
    --served-model-name "${QSA_SERVED_MODEL_NAME}" \
    --tokenizer "${QSA_TOKENIZER_DIR}" \
    --host "${QSA_HOST}" --port "${QSA_PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${QSA_TENSOR_PARALLEL_SIZE}" \
    --pipeline-parallel-size 1 --distributed-executor-backend mp \
    --generation-config "${QSA_GENERATION_CONFIG}" \
    --stream-interval "${QSA_STREAM_INTERVAL}" \
    --dtype bfloat16 --kv-cache-dtype bfloat16 \
    --attention-config '{"indexer_kv_dtype":"bf16"}' \
    --moe-backend triton \
    --additional-config '{"gdn_prefill_backend":"triton"}' \
    --mamba-ssm-cache-dtype float32 --mamba-cache-mode "${qsa_mamba_cache_mode}" \
    --enable-chunked-prefill \
    --max-model-len "${QSA_MAX_MODEL_LEN}" \
    --max-num-seqs "${QSA_MAX_NUM_SEQS}" \
    --max-num-batched-tokens "${QSA_MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${QSA_GPU_MEMORY_UTILIZATION}" \
    --scheduling-policy "${QSA_SCHEDULING_POLICY}" \
    --long-prefill-token-threshold "${QSA_LONG_PREFILL_TOKEN_THRESHOLD}" \
    --watermark "${QSA_KV_WATERMARK}" \
    --compilation-config "${QSA_COMPILATION_CONFIG}" \
    "${qsa_optional_args[@]}" "$@"
