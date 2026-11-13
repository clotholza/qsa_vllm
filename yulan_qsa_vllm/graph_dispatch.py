# SPDX-License-Identifier: Apache-2.0
"""Context-length CUDA graph variants for short QSA decode selection."""

from contextlib import contextmanager
from dataclasses import asdict, dataclass

from vllm.config.compilation import CUDAGraphMode
from vllm.v1.worker.gpu.cudagraph_utils import (
    BatchExecutionDescriptor,
    CudaGraphManager,
    ModelCudaGraphManager,
)
from vllm.v1.worker.gpu.model_runner import GPUModelRunner


SHORT_DECODE_COLUMNS = 32768
_runtime_columns = 0
_capture_columns = None


@dataclass(frozen=True)
class QSABatchExecutionDescriptor(BatchExecutionDescriptor):
    qsa_columns: int = 0


def decode_column_bucket(max_seq_len: int, compress_ratio: int = 4) -> int:
    """Zero denotes the full page-table capacity; short rows use one selector."""
    return SHORT_DECODE_COLUMNS if max_seq_len // compress_ratio <= SHORT_DECODE_COLUMNS else 0


def decode_logits_columns(max_seq_len: int, compress_ratio: int, capacity: int) -> int:
    columns = (_capture_columns if _capture_columns is not None
               else decode_column_bucket(max_seq_len, compress_ratio))
    return min(columns, capacity) if columns else capacity


@contextmanager
def _capture_column_scope(columns):
    global _capture_columns
    previous = _capture_columns
    _capture_columns = columns
    try:
        yield
    finally:
        _capture_columns = previous


def install_short_decode_dispatch():
    original_init = ModelCudaGraphManager._init_candidates
    original_dispatch = ModelCudaGraphManager.dispatch
    original_capture = CudaGraphManager.capture
    original_gather = GPUModelRunner.gather_batch_req_state

    def init_candidates(manager):
        original_init(manager)
        if CUDAGraphMode.FULL in manager._capture_descs:
            manager._qsa_full_descs = {
                (desc, columns): QSABatchExecutionDescriptor(**asdict(desc), qsa_columns=columns)
                for desc in manager._capture_descs[CUDAGraphMode.FULL]
                for columns in (0, SHORT_DECODE_COLUMNS)
            }
            manager._capture_descs[CUDAGraphMode.FULL] = list(manager._qsa_full_descs.values())

    def dispatch(manager, *args, **kwargs):
        desc = original_dispatch(manager, *args, **kwargs)
        if desc.cg_mode == CUDAGraphMode.FULL:
            desc = manager._qsa_full_descs[(desc, _runtime_columns)]
        return desc

    def capture(manager, create_forward_fn, progress_bar_desc="Capturing CUDA graphs"):
        if not isinstance(manager, ModelCudaGraphManager):
            return original_capture(manager, create_forward_fn, progress_bar_desc)

        def create_qsa_forward(desc, warmup):
            columns = desc.qsa_columns if desc.cg_mode == CUDAGraphMode.FULL else None
            with _capture_column_scope(columns):
                forward = create_forward_fn(desc, warmup)

            def qsa_forward(mode):
                with _capture_column_scope(columns):
                    return forward(mode)

            return qsa_forward

        return original_capture(manager, create_qsa_forward, progress_bar_desc)

    def gather(runner, scheduler_output, dummy_run):
        global _runtime_columns
        batch, uniform = original_gather(runner, scheduler_output, dummy_run)
        _runtime_columns = 0
        if batch is not None:
            lengths = (runner.req_states.num_computed_tokens_np[batch.idx_mapping_np]
                       + batch.num_scheduled_tokens)
            _runtime_columns = decode_column_bucket(int(lengths.max()))
        return batch, uniform

    ModelCudaGraphManager._init_candidates = init_candidates
    ModelCudaGraphManager.dispatch = dispatch
    CudaGraphManager.capture = capture
    GPUModelRunner.gather_batch_req_state = gather
