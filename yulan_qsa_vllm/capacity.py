# SPDX-License-Identifier: Apache-2.0
"""Report capacity from the vLLM 0.30 scheduler's allocated hybrid KV pool."""

from copy import copy
from functools import wraps
import json
import os
from pathlib import Path
import tempfile


# Match the lengths expected by the evaluation admission proxy.
REQUEST_LENGTHS = {
    "math_code": 36864,
    **{f"{k}k": k * 1024 + 512 for k in (4, 8, 16, 32, 64, 128)},
    "256k": 262400,
    "512k": 524288,
}


def _write_capacity_report(destination, vllm_config, kv_cache_config):
    from vllm.v1.core.kv_cache_utils import get_max_concurrency_for_kv_cache_config

    # Keep the running configuration intact while varying the request length.
    trial = copy(vllm_config)
    trial.model_config = copy(vllm_config.model_config)
    capacities = {}
    for label, length in REQUEST_LENGTHS.items():
        trial.model_config.max_model_len = length
        capacities[label] = get_max_concurrency_for_kv_cache_config(
            trial, kv_cache_config
        )
    record = {
        "pid": os.getpid(),
        "num_blocks": kv_cache_config.num_blocks,
        "groups": len(kv_cache_config.kv_cache_groups),
        "service_max_model_len": vllm_config.model_config.max_model_len,
        "request_lengths": REQUEST_LENGTHS,
        "concurrency": capacities,
        "method": "vLLM get_max_concurrency_for_kv_cache_config; actual hybrid groups",
    }
    data = json.dumps(record, indent=2, allow_nan=False) + "\n"
    # Readiness consumers must always see a complete report.
    with tempfile.NamedTemporaryFile(
        mode="w", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.close()
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def install_capacity_observer():
    """Hook only the opt-in report, after allocation and model warmup."""
    if not os.environ.get("QSA_CAPACITY_REPORT"):
        return

    from vllm.v1.engine.core import EngineCore

    original = EngineCore._initialize_kv_caches
    if getattr(original, "_qsa_capacity_observer", False):
        return

    @wraps(original)
    def observed(core, vllm_config):
        report_path = os.environ.get("QSA_CAPACITY_REPORT")
        if report_path:
            destination = Path(report_path)
            if not destination.is_absolute():
                raise ValueError("QSA_CAPACITY_REPORT must be an absolute path")
            if not destination.parent.is_dir():
                raise FileNotFoundError(destination.parent)
        kv_cache_config = original(core, vllm_config)
        if report_path:
            _write_capacity_report(destination, vllm_config, kv_cache_config)
        return kv_cache_config

    observed._qsa_capacity_observer = True
    EngineCore._initialize_kv_caches = observed
