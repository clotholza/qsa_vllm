// SPDX-License-Identifier: Apache-2.0
// SM80 QSA selection: short/medium histogram paths and partitioned decode.
// The histogram/vector-load structure follows vLLM's persistent top-k:
// https://github.com/vllm-project/vllm/blob/v0.30.0/csrc/libtorch_stable/persistent_topk.cuh
// Exact FP32 refinement and the final scan select boundary ties by logical ID.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <stdint.h>

namespace {
constexpr int Threads = 1024;
constexpr int Buffered = 2048;
constexpr int ShortColumns = 8192;
constexpr int MediumColumns = 32768;
constexpr int PartitionColumns = 8192;
struct Range { uint32_t lo, hi; };
struct MergeRange {
  __device__ Range operator()(Range a, Range b) const {
    return {min(a.lo, b.lo), max(a.hi, b.hi)};
  }
};
struct Counts { int strict, tied; };
struct AddCounts {
  __device__ Counts operator()(Counts a, Counts b) const {
    return {a.strict + b.strict, a.tied + b.tied};
  }
};
using Reduce = cub::BlockReduce<Range, Threads>;
using Scan = cub::BlockScan<Counts, Threads>;
using HistogramScan = cub::BlockScan<int, Threads>;
union CollectiveStorage {
  Reduce::TempStorage reduce;
  Scan::TempStorage scan;
  HistogramScan::TempStorage histogram_scan;
};
struct Shared {
  int histogram[2048];
  int remaining, candidate_count, threshold_bin, counter;
  Range key_range;
  uint32_t lower, upper;
  CollectiveStorage collective;
};

// ReLU-summed indexer scores are nonnegative. FP16 is only a monotone coarse
// bin; every selected threshold is subsequently refined using full FP32 bits.
template <int Bins>
__device__ __forceinline__ int coarse(float x) {
  const uint32_t half_bits = __half_as_ushort(__float2half_rn(x));
  return Bins == 2048 ? half_bits >> 5 : half_bits >> 8;
}

__device__ __forceinline__ void histogram_add(int* histogram, bool active, int bin) {
  const unsigned lanes = __ballot_sync(0xffffffffu, active);
  if (active) {
    const unsigned peers = __match_any_sync(lanes, bin);
    if ((threadIdx.x & 31) == __ffs(peers) - 1)
      atomicAdd(histogram + bin, __popc(peers));
  }
}

template <bool Indirect>
__device__ __forceinline__ int logical_id(const int32_t* ids, int start, int offset) {
  if constexpr (Indirect) return ids[offset];
  return start + offset;
}

template <bool Indirect>
__device__ __forceinline__ float value_at(const float* scores, const int32_t* ids,
                                         int start, int offset) {
  return scores[logical_id<Indirect>(ids, start, offset)];
}

template <bool Indirect>
__device__ __forceinline__ float4 load_four(const float* scores, const int32_t* ids,
                                           int start, int offset) {
  if constexpr (Indirect) {
    const int4 selected = *reinterpret_cast<const int4*>(ids + offset);
    return {scores[selected.x], scores[selected.y], scores[selected.z], scores[selected.w]};
  } else {
    return *reinterpret_cast<const float4*>(scores + start + offset);
  }
}

struct CompressedWriter {
  int32_t* output;
  __device__ __forceinline__ void operator()(int rank, int id) const {
    output[rank] = id;
  }
};

struct ExpansionMetadata {
  const int64_t* positions;
  const int32_t* sequence_lengths;
  const int32_t* token_to_req;
  const int32_t* table;
  int32_t* counts;
  int64_t table_stride, output_column_stride;
  int num_requests, table_width, page_size, ratio, output_width;
};

template <bool Physical, bool StoreCounts>
struct ExpandedWriter {
  int32_t* output;
  int32_t* count;
  const int32_t* table;
  int64_t output_column_stride;
  int sequence_length, complete_blocks, tail_start, tail_count;
  int page_size, table_width, ratio, output_width;

  __device__ __forceinline__ int map_token(int token) const {
    if (token < 0 || token >= sequence_length) return -1;
    if constexpr (Physical) {
      const int logical_page = token / page_size;
      if (logical_page >= table_width) return -1;
      const int physical_page = table[logical_page];
      return physical_page >= 0 ? physical_page * page_size + token % page_size : -1;
    }
    return token;
  }

  __device__ __forceinline__ void operator()(int rank, int block) const {
    if (rank >= complete_blocks) return;
    const int first_token = block * ratio;
    if constexpr (Physical) {
      // Compression groups are page-aligned when page_size is a multiple of
      // the ratio (272/4 in the serving cache), so one page lookup serves all
      // members. The other layout retains the per-token page mapping.
      if (page_size % ratio == 0) {
        const int logical_page = max(first_token, 0) / page_size;
        const int physical_page = first_token >= 0 && first_token < sequence_length
            && logical_page < table_width ? table[logical_page] : -1;
        const int first_slot = physical_page * page_size + first_token % page_size;
        for (int offset = 0; offset < ratio; ++offset) {
          const int token = first_token + offset;
          output[(rank * ratio + offset) * output_column_stride] =
              physical_page >= 0 && token >= 0 && token < sequence_length
              ? first_slot + offset : -1;
        }
        return;
      }
    }
    for (int offset = 0; offset < ratio; ++offset)
      output[(rank * ratio + offset) * output_column_stride] = map_token(first_token + offset);
  }

  __device__ __forceinline__ void finish() const {
    const int expanded_count = complete_blocks * ratio;
    for (int column = expanded_count + threadIdx.x; column < output_width; column += Threads) {
      const int offset = column - expanded_count;
      output[column * output_column_stride] = offset < tail_count && offset < ratio - 1
          ? map_token(tail_start + offset) : -1;
    }
    if constexpr (StoreCounts) {
      if (threadIdx.x == 0) *count = min(expanded_count + tail_count, sequence_length);
    }
  }
};

template <bool Physical, bool StoreCounts>
__device__ __forceinline__ ExpandedWriter<Physical, StoreCounts> make_expanded_writer(
    const ExpansionMetadata& meta, int row, int topk, int32_t* output) {
  const int request = meta.token_to_req[row];
  const bool mapped = request >= 0 && request < meta.num_requests;
  const int sequence_length = mapped ? meta.sequence_lengths[request] : 0;
  const int position = int(meta.positions[row]);
  const int complete_blocks = min(min((position + 1) / meta.ratio,
                                      sequence_length / meta.ratio), topk);
  const int tail_start = ((position + 1) / meta.ratio) * meta.ratio;
  return {output, StoreCounts ? meta.counts + row : nullptr,
          Physical ? meta.table + int64_t(mapped ? request : 0) * meta.table_stride : nullptr,
          meta.output_column_stride, sequence_length, complete_blocks, tail_start,
          position + 1 - tail_start, meta.page_size, meta.table_width, meta.ratio,
          meta.output_width};
}

template <bool Indirect, int Bins, typename Writer>
__device__ void select_row(const float* scores, const int32_t* ids, int start,
                           int length, Writer output, int topk,
                           Shared& shared, int* candidates) {
  const int tid = threadIdx.x;
  if (length <= topk) {
    for (int i = tid; i < topk; i += Threads)
      output(i, i < length ? logical_id<Indirect>(ids, start, i) : -1);
    return;
  }
  for (int bin = tid; bin < Bins; bin += Threads) shared.histogram[bin] = 0;
  __syncthreads();
  for (int tile = 0; tile < length; tile += Threads * 4) {
    const int base = tile + tid * 4;
    float values[4];
    if (base + 3 < length) {
      const float4 v = load_four<Indirect>(scores, ids, start, base);
      values[0] = v.x; values[1] = v.y; values[2] = v.z; values[3] = v.w;
    } else {
      #pragma unroll
      for (int item = 0; item < 4; ++item)
        values[item] = base + item < length
            ? value_at<Indirect>(scores, ids, start, base + item) : 0.0f;
    }
    #pragma unroll
    for (int item = 0; item < 4; ++item)
      histogram_add(shared.histogram, base + item < length, coarse<Bins>(values[item]));
  }
  __syncthreads();
  if constexpr (Bins == 2048) {
    // Parallel threshold location for the wider short-row histogram.
    const int h0 = shared.histogram[2 * tid];
    const int pair = h0 + shared.histogram[2 * tid + 1];
    int prefix, total;
    HistogramScan(shared.collective.histogram_scan).ExclusiveSum(pair, prefix, total);
    const int suffix = total - prefix;
    const int right_suffix = suffix - h0;
    const int next_suffix = suffix - pair;
    if (suffix >= topk && right_suffix < topk) {
      shared.threshold_bin = 2 * tid;
      shared.remaining = topk - right_suffix;
      shared.candidate_count = h0;
    }
    if (right_suffix >= topk && next_suffix < topk) {
      shared.threshold_bin = 2 * tid + 1;
      shared.remaining = topk - next_suffix;
      shared.candidate_count = pair - h0;
    }
  } else if (tid == 0) {
    int need = topk;
    int bin = 255;
    for (; bin > 0 && need > shared.histogram[bin]; --bin)
      need -= shared.histogram[bin];
    shared.threshold_bin = bin;
    shared.remaining = need;
    shared.candidate_count = shared.histogram[bin];
  }
  if (tid == 0) {
    shared.counter = 0;
    shared.lower = 0;
    shared.upper = 0xffffffffu;
  }
  __syncthreads();
  const int selected_coarse = shared.threshold_bin;
  Range local = {0xffffffffu, 0u};
  for (int tile = 0; tile < length; tile += Threads * 4) {
    const int base = tile + tid * 4;
    float values[4];
    if (base + 3 < length) {
      const float4 v = load_four<Indirect>(scores, ids, start, base);
      values[0] = v.x; values[1] = v.y; values[2] = v.z; values[3] = v.w;
    } else {
      #pragma unroll
      for (int item = 0; item < 4; ++item)
        values[item] = base + item < length
            ? value_at<Indirect>(scores, ids, start, base + item) : 0.0f;
    }
    #pragma unroll
    for (int item = 0; item < 4; ++item) {
      if (base + item < length && coarse<Bins>(values[item]) == selected_coarse) {
        const uint32_t key = __float_as_uint(values[item]);
        local = MergeRange{}(local, {key, key});
        if (shared.candidate_count <= Buffered)
          candidates[atomicAdd(&shared.counter, 1)] = base + item;
      }
    }
  }
  Range all = Reduce(shared.collective.reduce).Reduce(local, MergeRange{});
  if (tid == 0) shared.key_range = all;
  __syncthreads();

  bool buffered = shared.candidate_count <= Buffered;
  int bank = 0;
  while (shared.key_range.lo != shared.key_range.hi) {
    const int high_bit = 31 - __clz(shared.key_range.lo ^ shared.key_range.hi);
    const int shift = max(0, high_bit - 7);
    if (tid < 256) shared.histogram[tid] = 0;
    __syncthreads();
    const int input_count = buffered ? shared.candidate_count : length;
    for (int base = 0; base < input_count; base += Threads) {
      const int offset = base + tid;
      const int source_offset = offset < input_count
          ? (buffered ? candidates[bank * Buffered + offset] : offset) : 0;
      const float value = offset < input_count
          ? value_at<Indirect>(scores, ids, start, source_offset) : 0.0f;
      const uint32_t key = __float_as_uint(value);
      const bool active = offset < input_count &&
          (buffered || (coarse<Bins>(value) == selected_coarse
                        && key >= shared.lower && key <= shared.upper));
      histogram_add(shared.histogram, active, (key >> shift) & 255);
    }
    __syncthreads();
    if (tid == 0) {
      int need = shared.remaining;
      int bin = 255;
      for (; bin > 0 && need > shared.histogram[bin]; --bin)
        need -= shared.histogram[bin];
      shared.threshold_bin = bin;
      shared.remaining = need;
      shared.candidate_count = shared.histogram[bin];
      const uint32_t prefix_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
      shared.lower = (shared.key_range.lo & prefix_mask) | (uint32_t(bin) << shift);
      shared.upper = shared.lower | ((1u << shift) - 1u);
      shared.counter = 0;
    }
    __syncthreads();
    local = {0xffffffffu, 0u};
    const int next_bank = bank ^ 1;
    for (int offset = tid; offset < input_count; offset += Threads) {
      const int source_offset = buffered ? candidates[bank * Buffered + offset] : offset;
      const float value = value_at<Indirect>(scores, ids, start, source_offset);
      const uint32_t key = __float_as_uint(value);
      if (key >= shared.lower && key <= shared.upper
          && (buffered || coarse<Bins>(value) == selected_coarse)) {
        local = MergeRange{}(local, {key, key});
        if (shared.candidate_count <= Buffered)
          candidates[next_bank * Buffered + atomicAdd(&shared.counter, 1)] = source_offset;
      }
    }
    all = Reduce(shared.collective.reduce).Reduce(local, MergeRange{});
    if (tid == 0) shared.key_range = all;
    __syncthreads();
    buffered = shared.candidate_count <= Buffered;
    bank = next_bank;
  }

  // Input IDs are ascending in both the raw row and the concatenated local-K
  // sets. The two-count scan therefore writes the exact selected set in global
  // logical-ID order, including the smallest IDs at the FP32 boundary score.
  const uint32_t threshold = shared.key_range.lo;
  Counts prior = {0, 0};
  for (int base = 0; base < length; base += Threads) {
    const int offset = base + tid;
    const uint32_t key = offset < length
        ? __float_as_uint(value_at<Indirect>(scores, ids, start, offset)) : 0u;
    Counts flags = {int(offset < length && key > threshold),
                    int(offset < length && key == threshold)};
    Counts prefix, total;
    Scan(shared.collective.scan).ExclusiveScan(flags, prefix, Counts{0, 0}, AddCounts{}, total);
    const int strict_before = prior.strict + prefix.strict;
    const int tie_before = prior.tied + prefix.tied;
    if (flags.strict || (flags.tied && tie_before < shared.remaining))
      output(strict_before + min(tie_before, shared.remaining),
             logical_id<Indirect>(ids, start, offset));
    prior = AddCounts{}(prior, total);
    __syncthreads();
    if (prior.strict + min(prior.tied, shared.remaining) == topk) break;
  }
}

template <bool Indirect, typename Writer>
__device__ void dispatch_row(const float* scores, const int32_t* ids, int start,
                             int length, Writer output, int topk,
                             Shared& shared, int* candidates) {
  if (length <= ShortColumns)
    select_row<Indirect, 2048>(scores, ids, start, length, output, topk, shared, candidates);
  else
    select_row<Indirect, 256>(scores, ids, start, length, output, topk, shared, candidates);
}

__global__ __launch_bounds__(Threads) void qsa_topk_direct(
    const float* scores, const int32_t* lengths, int32_t* output,
    int64_t score_stride, int64_t output_stride, int topk) {
  __shared__ Shared shared;
  extern __shared__ int candidates[];
  const int row = blockIdx.x;
  dispatch_row<false>(scores + int64_t(row) * score_stride, nullptr, 0,
                       lengths[row], CompressedWriter{output + int64_t(row) * output_stride},
                       topk, shared, candidates);
}

__global__ __launch_bounds__(Threads) void qsa_topk_partitions(
    const float* scores, const int32_t* lengths, int32_t* output, int32_t* partial,
    int64_t score_stride, int64_t output_stride, int parts, int topk) {
  __shared__ Shared shared;
  extern __shared__ int candidates[];
  const int row = blockIdx.x;
  const int part = blockIdx.y;
  const int length = lengths[row];
  const float* row_scores = scores + int64_t(row) * score_stride;
  if (length <= MediumColumns) {
    if (part == 0)
      dispatch_row<false>(row_scores, nullptr, 0, length,
          CompressedWriter{output + int64_t(row) * output_stride}, topk, shared, candidates);
    return;
  }
  const int start = part * PartitionColumns;
  if (start >= length) return;
  const int count = min(PartitionColumns, length - start);
  dispatch_row<false>(row_scores, nullptr, start, count,
      CompressedWriter{partial + (int64_t(row) * parts + part) * topk}, topk, shared, candidates);
}

__global__ __launch_bounds__(Threads) void qsa_topk_merge(
    const float* scores, const int32_t* lengths, int32_t* output, const int32_t* partial,
    int64_t score_stride, int64_t output_stride, int parts, int topk) {
  __shared__ Shared shared;
  extern __shared__ int candidates[];
  const int row = blockIdx.x;
  const int length = lengths[row];
  if (length <= MediumColumns) return;
  const int count = (length / PartitionColumns) * topk
      + min(length % PartitionColumns, topk);
  dispatch_row<true>(scores + int64_t(row) * score_stride,
      partial + int64_t(row) * parts * topk, 0, count,
      CompressedWriter{output + int64_t(row) * output_stride}, topk, shared, candidates);
}

template <bool Physical, bool StoreCounts>
__global__ __launch_bounds__(Threads) void qsa_topk_expand_direct(
    const float* scores, const int32_t* lengths, int32_t* output,
    int64_t score_stride, int64_t output_stride, int topk, ExpansionMetadata meta) {
  __shared__ Shared shared;
  extern __shared__ int candidates[];
  const int row = blockIdx.x;
  const auto writer = make_expanded_writer<Physical, StoreCounts>(
      meta, row, topk, output + int64_t(row) * output_stride);
  dispatch_row<false>(scores + int64_t(row) * score_stride, nullptr, 0,
      lengths[row], writer, topk, shared, candidates);
  writer.finish();
}

template <bool Physical, bool StoreCounts>
__global__ __launch_bounds__(Threads) void qsa_topk_expand_partitions(
    const float* scores, const int32_t* lengths, int32_t* output, int32_t* partial,
    int64_t score_stride, int64_t output_stride, int parts, int topk, ExpansionMetadata meta) {
  __shared__ Shared shared;
  extern __shared__ int candidates[];
  const int row = blockIdx.x;
  const int part = blockIdx.y;
  const int length = lengths[row];
  const float* row_scores = scores + int64_t(row) * score_stride;
  if (length <= MediumColumns) {
    if (part == 0) {
      const auto writer = make_expanded_writer<Physical, StoreCounts>(
          meta, row, topk, output + int64_t(row) * output_stride);
      dispatch_row<false>(row_scores, nullptr, 0, length, writer, topk, shared, candidates);
      writer.finish();
    }
    return;
  }
  const int start = part * PartitionColumns;
  if (start >= length) return;
  const int count = min(PartitionColumns, length - start);
  dispatch_row<false>(row_scores, nullptr, start, count,
      CompressedWriter{partial + (int64_t(row) * parts + part) * topk},
      topk, shared, candidates);
}

template <bool Physical, bool StoreCounts>
__global__ __launch_bounds__(Threads) void qsa_topk_expand_merge(
    const float* scores, const int32_t* lengths, int32_t* output, const int32_t* partial,
    int64_t score_stride, int64_t output_stride, int parts, int topk, ExpansionMetadata meta) {
  __shared__ Shared shared;
  extern __shared__ int candidates[];
  const int row = blockIdx.x;
  const int length = lengths[row];
  if (length <= MediumColumns) return;
  const int count = (length / PartitionColumns) * topk
      + min(length % PartitionColumns, topk);
  const auto writer = make_expanded_writer<Physical, StoreCounts>(
      meta, row, topk, output + int64_t(row) * output_stride);
  dispatch_row<true>(scores + int64_t(row) * score_stride,
      partial + int64_t(row) * parts * topk, 0, count,
      writer, topk, shared, candidates);
  writer.finish();
}

template <bool Physical, bool StoreCounts>
void launch_topk_expand(const at::Tensor& scores, const at::Tensor& lengths,
                       const at::Tensor& positions, const at::Tensor& sequence_lengths,
                       const at::Tensor& token_to_req, const at::Tensor& table,
                       at::Tensor output, at::Tensor counts, int ratio, int page_size,
                       cudaStream_t stream) {
  constexpr size_t ScratchBytes = 2 * Buffered * sizeof(int);
  const int topk = output.size(1) / ratio;
  const ExpansionMetadata meta = {
      positions.data_ptr<int64_t>(), sequence_lengths.data_ptr<int32_t>(),
      token_to_req.data_ptr<int32_t>(), table.data_ptr<int32_t>(), counts.data_ptr<int32_t>(),
      Physical ? table.stride(0) : 0, output.stride(1),
      int(sequence_lengths.numel()), Physical ? int(table.size(1)) : 0,
      page_size, ratio, int(output.size(1))};
  if (scores.size(0) <= 32 && scores.size(1) > MediumColumns) {
    const int parts = (scores.size(1) + PartitionColumns - 1) / PartitionColumns;
    auto partial = at::empty({scores.size(0), parts, topk}, output.options());
    qsa_topk_expand_partitions<Physical, StoreCounts>
        <<<dim3(scores.size(0), parts), Threads, ScratchBytes, stream>>>(
        scores.data_ptr<float>(), lengths.data_ptr<int32_t>(), output.data_ptr<int32_t>(),
        partial.data_ptr<int32_t>(), scores.stride(0), output.stride(0), parts, topk, meta);
    qsa_topk_expand_merge<Physical, StoreCounts>
        <<<scores.size(0), Threads, ScratchBytes, stream>>>(
        scores.data_ptr<float>(), lengths.data_ptr<int32_t>(), output.data_ptr<int32_t>(),
        partial.data_ptr<int32_t>(), scores.stride(0), output.stride(0), parts, topk, meta);
  } else {
    qsa_topk_expand_direct<Physical, StoreCounts>
        <<<scores.size(0), Threads, ScratchBytes, stream>>>(
        scores.data_ptr<float>(), lengths.data_ptr<int32_t>(), output.data_ptr<int32_t>(),
        scores.stride(0), output.stride(0), topk, meta);
  }
}

void qsa_topk_expand(const at::Tensor& scores, const at::Tensor& lengths,
                     const at::Tensor& positions, const at::Tensor& sequence_lengths,
                     const at::Tensor& token_to_req, const at::Tensor& table,
                     at::Tensor output, at::Tensor counts, int64_t ratio, int64_t page_size,
                     bool physical, bool store_counts) {
  const c10::cuda::CUDAGuard device_guard(scores.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(scores.get_device());
  if (physical) {
    if (store_counts)
      launch_topk_expand<true, true>(scores, lengths, positions, sequence_lengths,
          token_to_req, table, output, counts, ratio, page_size, stream.stream());
    else
      launch_topk_expand<true, false>(scores, lengths, positions, sequence_lengths,
          token_to_req, table, output, counts, ratio, page_size, stream.stream());
  } else {
    if (store_counts)
      launch_topk_expand<false, true>(scores, lengths, positions, sequence_lengths,
          token_to_req, table, output, counts, ratio, page_size, stream.stream());
    else
      launch_topk_expand<false, false>(scores, lengths, positions, sequence_lengths,
          token_to_req, table, output, counts, ratio, page_size, stream.stream());
  }
}

void qsa_topk(const at::Tensor& scores, const at::Tensor& lengths, at::Tensor output) {
  const c10::cuda::CUDAGuard device_guard(scores.device());
  const auto stream = c10::cuda::getCurrentCUDAStream(scores.get_device());
  constexpr size_t ScratchBytes = 2 * Buffered * sizeof(int);
  if (scores.size(0) <= 32 && scores.size(1) > MediumColumns) {
    const int parts = (scores.size(1) + PartitionColumns - 1) / PartitionColumns;
    auto partial = at::empty({scores.size(0), parts, output.size(1)}, output.options());
    qsa_topk_partitions<<<dim3(scores.size(0), parts), Threads, ScratchBytes, stream.stream()>>>(
        scores.data_ptr<float>(), lengths.data_ptr<int32_t>(), output.data_ptr<int32_t>(),
        partial.data_ptr<int32_t>(), scores.stride(0), output.stride(0), parts, output.size(1));
    qsa_topk_merge<<<scores.size(0), Threads, ScratchBytes, stream.stream()>>>(
        scores.data_ptr<float>(), lengths.data_ptr<int32_t>(), output.data_ptr<int32_t>(),
        partial.data_ptr<int32_t>(), scores.stride(0), output.stride(0), parts, output.size(1));
  } else {
    qsa_topk_direct<<<scores.size(0), Threads, ScratchBytes, stream.stream()>>>(
        scores.data_ptr<float>(), lengths.data_ptr<int32_t>(), output.data_ptr<int32_t>(),
        scores.stride(0), output.stride(0), output.size(1));
  }
}
}  // namespace

TORCH_LIBRARY_FRAGMENT(yulan_qsa_kernels, m) {
  m.def("topk(Tensor scores, Tensor lengths, Tensor(a!) output) -> ()");
  m.def("topk_expand(Tensor scores, Tensor lengths, Tensor positions, Tensor sequence_lengths, "
        "Tensor token_to_req, Tensor table, Tensor(a!) output, Tensor(b!) counts, "
        "int ratio, int page_size, bool physical, bool store_counts) -> ()");
}
TORCH_LIBRARY_IMPL(yulan_qsa_kernels, CUDA, m) {
  m.impl("topk", &qsa_topk);
  m.impl("topk_expand", &qsa_topk_expand);
}
