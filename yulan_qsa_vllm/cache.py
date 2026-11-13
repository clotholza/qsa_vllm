# SPDX-License-Identifier: Apache-2.0
"""vLLM 0.30 paged owners for the BF16 QSA indexer state.

The raw owner keeps the open compression group in a request-local circular
buffer. The compressed owner stores one normalized key per four tokens using
the cache manager's ordinary page lifecycle. Both are replicated across TP;
the main K/V heads are independently tensor-parallel sharded.
"""

import torch

from vllm.models.qwen4_exp.common.qsa_cache import QSACompressedKeyCache
from vllm.models.qwen4_exp.common.qsa_cache import (
    QSAKeyStateCache as _OfficialKeyStateCache,
)
from vllm.models.qwen4_exp.common.qsa_cache import _QSAStateCache


class QSAKeyStateCache(_OfficialKeyStateCache):
    """Keep only the managed cache as a reference to bound storage.

    This follows the newer official owner: views are derived on access, so
    clearing ``kv_cache`` after profiling also releases the raw cache storage.
    Constructor, metadata backend, page specification and page binding remain
    the pinned vLLM implementation.
    """

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        _QSAStateCache.bind_kv_cache(self, kv_cache)

    @property
    def key_cache(self) -> torch.Tensor:
        return self.kv_cache[..., : self.key_head_size]

    @property
    def rope_position_cache(self) -> torch.Tensor | None:
        if not self.cache_rope_positions:
            return None
        return self.kv_cache[..., self.rope_position_offset :].view(torch.int64)


__all__ = ["QSAKeyStateCache", "QSACompressedKeyCache"]
