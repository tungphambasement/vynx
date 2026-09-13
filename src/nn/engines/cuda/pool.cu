#include "device/cuda_device.hpp"
#include "nn/engines/engine_handle.hpp"
#ifdef TUNX_USE_CUDA
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <ctime>

#include "cuda/helpers.cuh"
#include "nn/engines/cuda_engine.hpp"
#include "type/type.hpp"

namespace tunx {

#define BLOCK_SIZE 256
#define WARP_SIZE 32

template <typename IO_T, typename COMPUTE_T>
__global__ void avgpool_nchw_fwd_kernel(const IO_T* input, IO_T* output, size_t batch_size,
                                        size_t channels, size_t input_h, size_t input_w,
                                        size_t output_h, size_t output_w, size_t pool_h,
                                        size_t pool_w, size_t stride_h, size_t stride_w,
                                        size_t pad_h, size_t pad_w, COMPUTE_T pool_size_inv) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = batch_size * channels * output_h * output_w;

  if (idx >= total_outputs) return;

  int n = idx / (channels * output_h * output_w);
  int remaining = idx % (channels * output_h * output_w);
  int c = remaining / (output_h * output_w);
  remaining = remaining % (output_h * output_w);
  int out_h = remaining / output_w;
  int out_w = remaining % output_w;

  long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
  long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);

  long h_start_valid = max(0L, h_start);
  long w_start_valid = max(0L, w_start);
  long h_end_valid = min(static_cast<long>(input_h), h_start + static_cast<long>(pool_h));
  long w_end_valid = min(static_cast<long>(input_w), w_start + static_cast<long>(pool_w));

  size_t input_offset = (n * channels + c) * input_h * input_w;
  COMPUTE_T sum = COMPUTE_T(0);

  for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
    for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
      sum += static_cast<COMPUTE_T>(input[input_offset + ih * input_w + iw]);
    }
  }

  output[idx] = static_cast<IO_T>(sum * pool_size_inv);
}

template <typename IO_T, typename COMPUTE_T>
__global__ void avgpool_nchw_dgrad_kernel(const IO_T* gradient, IO_T* grad_input, size_t batch_size,
                                          size_t channels, size_t input_h, size_t input_w,
                                          size_t output_h, size_t output_w, size_t pool_h,
                                          size_t pool_w, size_t stride_h, size_t stride_w,
                                          size_t pad_h, size_t pad_w, COMPUTE_T pool_size_inv) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = batch_size * channels * output_h * output_w;

  if (idx >= total_outputs) return;

  int n = idx / (channels * output_h * output_w);
  int remaining = idx % (channels * output_h * output_w);
  int c = remaining / (output_h * output_w);
  remaining = remaining % (output_h * output_w);
  int out_h = remaining / output_w;
  int out_w = remaining % output_w;

  const COMPUTE_T grad_val = static_cast<COMPUTE_T>(gradient[idx]) * pool_size_inv;

  long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
  long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);

  long h_start_valid = max(0L, h_start);
  long w_start_valid = max(0L, w_start);
  long h_end_valid = min(static_cast<long>(input_h), h_start + static_cast<long>(pool_h));
  long w_end_valid = min(static_cast<long>(input_w), w_start + static_cast<long>(pool_w));

  size_t input_offset = (n * channels + c) * input_h * input_w;

  for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
    for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
      gpu_atomic_add(&grad_input[input_offset + ih * input_w + iw], static_cast<IO_T>(grad_val));
    }
  }
}

template <typename IO_T, typename COMPUTE_T>
void avgpool_nchw_fwd(const IO_T* input, IO_T* output, size_t batch_size, size_t channels,
                      size_t input_h, size_t input_w, size_t output_h, size_t output_w,
                      size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w, size_t pad_h,
                      size_t pad_w, cudaStream_t stream) {
  int total_outputs = batch_size * channels * output_h * output_w;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  COMPUTE_T pool_size_inv = COMPUTE_T(1) / static_cast<COMPUTE_T>(pool_h * pool_w);

  avgpool_nchw_fwd_kernel<IO_T, COMPUTE_T><<<num_blocks, threads_per_block, 0, stream>>>(
      input, output, batch_size, channels, input_h, input_w, output_h, output_w, pool_h, pool_w,
      stride_h, stride_w, pad_h, pad_w, pool_size_inv);
}

template <typename IO_T, typename COMPUTE_T>
void avgpool_nchw_bwd(const IO_T* gradient, IO_T* grad_input, size_t batch_size, size_t channels,
                      size_t input_h, size_t input_w, size_t output_h, size_t output_w,
                      size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w, size_t pad_h,
                      size_t pad_w, cudaStream_t stream) {
  int total_outputs = batch_size * channels * output_h * output_w;
  int threads_per_block = 256;
  int num_blocks = (total_outputs + threads_per_block - 1) / threads_per_block;

  COMPUTE_T pool_size_inv = COMPUTE_T(1) / static_cast<COMPUTE_T>(pool_h * pool_w);

  avgpool_nchw_dgrad_kernel<IO_T, COMPUTE_T><<<num_blocks, threads_per_block, 0, stream>>>(
      gradient, grad_input, batch_size, channels, input_h, input_w, output_h, output_w, pool_h,
      pool_w, stride_h, stride_w, pad_h, pad_w, pool_size_inv);
}

template <typename IO_T, typename COMPUTE_T>
__global__ void avgpool_fwd_kernel(const IO_T* input, IO_T* output, size_t batch_size,
                                   size_t height, size_t width, size_t channels, size_t pool_h,
                                   size_t pool_w, size_t stride_h, size_t stride_w, size_t pad_h,
                                   size_t pad_w, size_t output_h, size_t output_w) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs) return;

  size_t c = idx % channels;
  size_t ow = (idx / channels) % output_w;
  size_t oh = (idx / (channels * output_w)) % output_h;
  size_t b = idx / (channels * output_w * output_h);

  int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
  int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
  int h_end = min(h_start + static_cast<int>(pool_h), static_cast<int>(height));
  int w_end = min(w_start + static_cast<int>(pool_w), static_cast<int>(width));
  h_start = max(h_start, 0);
  w_start = max(w_start, 0);

  COMPUTE_T sum = COMPUTE_T(0);
  int count = 0;
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      size_t input_idx = ((b * height + h) * width + w) * channels + c;
      sum += static_cast<COMPUTE_T>(input[input_idx]);
      ++count;
    }
  }

  output[idx] = static_cast<IO_T>(count > 0 ? sum / static_cast<COMPUTE_T>(count) : COMPUTE_T(0));
}

template <typename IO_T, typename COMPUTE_T>
__global__ void avgpool_dgrad_kernel(const IO_T* grad_output, IO_T* grad_input, size_t batch_size,
                                     size_t input_h, size_t input_w, size_t channels, size_t pool_h,
                                     size_t pool_w, size_t stride_h, size_t stride_w, size_t pad_h,
                                     size_t pad_w, size_t output_h, size_t output_w) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs) return;

  size_t c = idx % channels;
  size_t ow = (idx / channels) % output_w;
  size_t oh = (idx / (channels * output_w)) % output_h;
  size_t b = idx / (channels * output_w * output_h);

  COMPUTE_T grad = static_cast<COMPUTE_T>(grad_output[idx]);

  int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
  int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
  int h_end = min(h_start + static_cast<int>(pool_h), static_cast<int>(input_h));
  int w_end = min(w_start + static_cast<int>(pool_w), static_cast<int>(input_w));
  h_start = max(h_start, 0);
  w_start = max(w_start, 0);

  int count = (h_end - h_start) * (w_end - w_start);
  if (count == 0) return;

  COMPUTE_T grad_per_element = grad / static_cast<COMPUTE_T>(count);
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      size_t input_idx = ((b * input_h + h) * input_w + w) * channels + c;
      cuda::gpu_atomic_add(&grad_input[input_idx], static_cast<IO_T>(grad_per_element));
    }
  }
}

template <typename IO_T>
__global__ void maxpool_nchw_fwd_kernel(const IO_T* input, IO_T* output, size_t batch_size,
                                        size_t channels, size_t input_h, size_t input_w,
                                        size_t output_h, size_t output_w, size_t pool_h,
                                        size_t pool_w, size_t stride_h, size_t stride_w,
                                        size_t pad_h, size_t pad_w, size_t* mask_indices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = batch_size * channels * output_h * output_w;

  if (idx >= total_outputs) return;

  int n = idx / (channels * output_h * output_w);
  int remaining = idx % (channels * output_h * output_w);
  int c = remaining / (output_h * output_w);
  remaining = remaining % (output_h * output_w);
  int out_h = remaining / output_w;
  int out_w = remaining % output_w;

  size_t input_offset = (n * channels + c) * input_h * input_w;

  long h_start = static_cast<long>(out_h * stride_h) - static_cast<long>(pad_h);
  long w_start = static_cast<long>(out_w * stride_w) - static_cast<long>(pad_w);
  long h_end = h_start + pool_h;
  long w_end = w_start + pool_w;

  long h_start_valid = max(0L, h_start);
  long w_start_valid = max(0L, w_start);
  long h_end_valid = min(static_cast<long>(input_h), h_end);
  long w_end_valid = min(static_cast<long>(input_w), w_end);

  IO_T max_val = cuda::lowest_value<IO_T>();
  size_t max_idx = 0;

  for (long ih = h_start_valid; ih < h_end_valid; ++ih) {
    for (long iw = w_start_valid; iw < w_end_valid; ++iw) {
      size_t cur_input_idx = input_offset + ih * input_w + iw;
      IO_T val = input[cur_input_idx];

      if (val > max_val || (ih == h_start_valid && iw == w_start_valid)) {
        max_val = val;
        max_idx = cur_input_idx;
      }
    }
  }

  output[idx] = max_val;
  if (mask_indices) {
    mask_indices[idx] = max_idx;
  }
}

template <typename IO_T>
__global__ void maxpool_nchw_dgrad_kernel(const IO_T* gradient, IO_T* grad_input, size_t batch_size,
                                          size_t channels, size_t output_h, size_t output_w,
                                          const size_t* mask_indices) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = batch_size * channels * output_h * output_w;

  if (idx >= total_outputs) return;

  const IO_T grad_val = gradient[idx];
  size_t input_idx = mask_indices[idx];

  gpu_atomic_add(&grad_input[input_idx], grad_val);
}

template <typename IO_T>
__global__ void maxpool_fwd_kernel(const IO_T* input, IO_T* output, int* mask_indices,
                                   size_t batch_size, size_t height, size_t width, size_t channels,
                                   size_t pool_h, size_t pool_w, size_t stride_h, size_t stride_w,
                                   size_t pad_h, size_t pad_w, size_t output_h, size_t output_w) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs) return;

  size_t c = idx % channels;
  size_t ow = (idx / channels) % output_w;
  size_t oh = (idx / (channels * output_w)) % output_h;
  size_t b = idx / (channels * output_w * output_h);

  int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
  int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
  int h_end = min(h_start + static_cast<int>(pool_h), static_cast<int>(height));
  int w_end = min(w_start + static_cast<int>(pool_w), static_cast<int>(width));
  h_start = max(h_start, 0);
  w_start = max(w_start, 0);

  IO_T max_val = cuda::lowest_value<IO_T>();
  int max_idx = -1;
  for (int h = h_start; h < h_end; ++h) {
    for (int w = w_start; w < w_end; ++w) {
      size_t input_idx = ((b * height + h) * width + w) * channels + c;
      IO_T val = input[input_idx];
      if (val > max_val) {
        max_val = val;

        int h_start_unclamped = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
        int w_start_unclamped = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);
        int rel_h = h - h_start_unclamped;
        int rel_w = w - w_start_unclamped;
        max_idx = rel_h * static_cast<int>(pool_w) + rel_w;
      }
    }
  }

  output[idx] = max_val;
  mask_indices[idx] = max_idx;
}

template <typename IO_T>
__global__ void maxpool_dgrad_kernel(const IO_T* grad_output, IO_T* grad_input,
                                     const int* mask_indices, size_t batch_size, size_t channels,
                                     size_t output_h, size_t output_w, size_t input_h,
                                     size_t input_w, size_t pool_w, size_t stride_h,
                                     size_t stride_w, size_t pad_h, size_t pad_w) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_outputs = batch_size * output_h * output_w * channels;

  if (idx >= total_outputs) return;

  int rel_idx = mask_indices[idx];
  if (rel_idx >= 0) {
    size_t c = idx % channels;
    size_t ow = (idx / channels) % output_w;
    size_t oh = (idx / (channels * output_w)) % output_h;
    size_t b = idx / (channels * output_w * output_h);

    int h_start = static_cast<int>(oh * stride_h) - static_cast<int>(pad_h);
    int w_start = static_cast<int>(ow * stride_w) - static_cast<int>(pad_w);

    int rel_h = rel_idx / static_cast<int>(pool_w);
    int rel_w = rel_idx % static_cast<int>(pool_w);

    int h = h_start + rel_h;
    int w = w_start + rel_w;

    if (h >= 0 && h < static_cast<int>(input_h) && w >= 0 && w < static_cast<int>(input_w)) {
      size_t in_idx = ((b * input_h + h) * input_w + w) * channels + c;
      cuda::gpu_atomic_add(&grad_input[in_idx], grad_output[idx]);
    }
  }
}

WorkspaceReq CUDAEngine::query_avgpool_graph(engine_handle backend_handle,
                                             const AvgPool2DStats& stats, DTypeDesc type_desc) {
  return {0, 0, 0};
}

WorkspaceReq CUDAEngine::query_maxpool2d_graph(engine_handle backend_handle,
                                               const MaxPool2DStats& stats, DTypeDesc type_desc) {
  return {0, 0, 0};
}

void CUDAEngine::avgpool_fwd(engine_handle backend_handle, const AvgPool2DStats& stats,
                             const void* input, void* output, void* workspace,
                             DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE2(type_desc.io_dtype, type_desc.compute_dtype, IO_T, COMPUTE_T, {
    size_t total_outputs = stats.batch_size * output_h * output_w * stats.channels;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    avgpool_fwd_kernel<IO_T, COMPUTE_T><<<blocks, threads, 0, stream>>>(
        static_cast<const IO_T*>(input), static_cast<IO_T*>(output), stats.batch_size, stats.height,
        stats.width, stats.channels, stats.pool_h, stats.pool_w, stats.stride_h, stats.stride_w,
        stats.pad_h, stats.pad_w, output_h, output_w);
  });
}

void CUDAEngine::avgpool_bwd(engine_handle backend_handle, const AvgPool2DStats& stats,
                             const void* grad_output, void* grad_input, void* workspace,
                             DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE2(type_desc.io_dtype, type_desc.compute_dtype, IO_T, COMPUTE_T, {
    size_t total_outputs = stats.batch_size * output_h * output_w * stats.channels;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    avgpool_dgrad_kernel<IO_T, COMPUTE_T><<<blocks, threads, 0, stream>>>(
        static_cast<const IO_T*>(grad_output), static_cast<IO_T*>(grad_input), stats.batch_size,
        stats.height, stats.width, stats.channels, stats.pool_h, stats.pool_w, stats.stride_h,
        stats.stride_w, stats.pad_h, stats.pad_w, output_h, output_w);
  });
}

void CUDAEngine::maxpool2d_fwd(engine_handle backend_handle, const MaxPool2DStats& stats,
                               const void* input, void* output, void* mask, void* workspace,
                               DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.io_dtype, IO_T, {
    size_t total_outputs = stats.batch_size * output_h * output_w * stats.channels;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    maxpool_fwd_kernel<IO_T><<<blocks, threads, 0, stream>>>(
        static_cast<const IO_T*>(input), static_cast<IO_T*>(output), static_cast<int*>(mask),
        stats.batch_size, stats.height, stats.width, stats.channels, stats.pool_h, stats.pool_w,
        stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w, output_h, output_w);
  });
}

void CUDAEngine::maxpool2d_infer(engine_handle backend_handle, const MaxPool2DStats& stats,
                                 const void* input, void* output, void* workspace,
                                 DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.io_dtype, IO_T, {
    size_t total_outputs = stats.batch_size * output_h * output_w * stats.channels;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    maxpool_fwd_kernel<IO_T><<<blocks, threads, 0, stream>>>(
        static_cast<const IO_T*>(input), static_cast<IO_T*>(output), static_cast<int*>(nullptr),
        stats.batch_size, stats.height, stats.width, stats.channels, stats.pool_h, stats.pool_w,
        stats.stride_h, stats.stride_w, stats.pad_h, stats.pad_w, output_h, output_w);
  });
}

void CUDAEngine::maxpool2d_bwd(engine_handle backend_handle, const MaxPool2DStats& stats,
                               const void* grad_output, void* grad_input, const void* mask,
                               void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  size_t output_h = (stats.height + 2 * stats.pad_h - stats.pool_h) / stats.stride_h + 1;
  size_t output_w = (stats.width + 2 * stats.pad_w - stats.pool_w) / stats.stride_w + 1;
  DISPATCH_DTYPE(type_desc.io_dtype, IO_T, {
    size_t total_outputs = stats.batch_size * output_h * output_w * stats.channels;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;
    maxpool_dgrad_kernel<IO_T><<<blocks, threads, 0, stream>>>(
        static_cast<const IO_T*>(grad_output), static_cast<IO_T*>(grad_input),
        static_cast<const int*>(mask), stats.batch_size, stats.channels, output_h, output_w,
        stats.height, stats.width, stats.pool_w, stats.stride_h, stats.stride_w, stats.pad_h,
        stats.pad_w);
  });
}

}  // namespace tunx

#endif
