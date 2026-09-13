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
#include "type/cuda/vectorized_types.cuh"
#include "type/type.hpp"

namespace tunx {

template <typename T>
__device__ __forceinline__ T compute_rsqrt(T x) {
  return static_cast<T>(rsqrt(static_cast<double>(x)));
}

template <>
__device__ __forceinline__ float compute_rsqrt<float>(float x) {
  return rsqrt(x);
}

template <>
__device__ __forceinline__ double compute_rsqrt<double>(double x) {
  return rsqrt(x);
}

#define BLOCK_SIZE 256
#define WARP_SIZE 32

template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
  }
  return val;
}

template <typename T>
__inline__ __device__ T blockReduceSum(T val) {
  static __shared__ T shared[32];
  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceSum(val);

  if (lane == 0) shared[wid] = val;
  __syncthreads();

  val = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : T(0);
  if (wid == 0) val = warpReduceSum(val);

  return val;
}

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void norm_add_bias_kernel(IO_T* output, const PARAM_T* bias, size_t batch_size,
                                     size_t output_features) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_size = batch_size * output_features;

  if (idx >= total_size) return;

  int out_f = idx % output_features;
  output[idx] += static_cast<IO_T>(bias[out_f]);
}

template <typename T>
struct WelfordData {
  T mean;
  T m2;
  T count;

  __device__ WelfordData()
      : mean(0),
        m2(0),
        count(0) {}
  __device__ WelfordData(T m, T v, T c)
      : mean(m),
        m2(v),
        count(c) {}
};

template <typename T>
__device__ WelfordData<T> welford_merge(WelfordData<T> a, WelfordData<T> b) {
  if (b.count == T(0)) return a;
  if (a.count == T(0)) return b;

  T new_count = a.count + b.count;
  T delta = b.mean - a.mean;
  T new_mean = a.mean + (delta * b.count) / new_count;
  T new_m2 = a.m2 + b.m2 + (delta * delta * a.count * b.count) / new_count;

  return WelfordData<T>(new_mean, new_m2, new_count);
}

template <typename T>
__device__ WelfordData<T> warpReduceWelford(WelfordData<T> val) {
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    T other_mean = __shfl_down_sync(0xffffffff, val.mean, offset);
    T other_m2 = __shfl_down_sync(0xffffffff, val.m2, offset);
    T other_count = __shfl_down_sync(0xffffffff, val.count, offset);
    val = welford_merge(val, WelfordData<T>(other_mean, other_m2, other_count));
  }
  return val;
}

template <typename T>
__device__ WelfordData<T> blockReduceWelford(WelfordData<T> val) {
  static __shared__ T shared_mean[32];
  static __shared__ T shared_m2[32];
  static __shared__ T shared_count[32];

  int lane = threadIdx.x % WARP_SIZE;
  int wid = threadIdx.x / WARP_SIZE;

  val = warpReduceWelford(val);

  if (lane == 0) {
    shared_mean[wid] = val.mean;
    shared_m2[wid] = val.m2;
    shared_count[wid] = val.count;
  }
  __syncthreads();

  WelfordData<T> block_val;

  if (threadIdx.x < (blockDim.x / WARP_SIZE)) {
    block_val.mean = shared_mean[threadIdx.x];
    block_val.m2 = shared_m2[threadIdx.x];
    block_val.count = shared_count[threadIdx.x];
  }

  if (wid == 0) {
    block_val = warpReduceWelford(block_val);
  }

  return block_val;
}

template <typename IO_T, typename COMPUTE_T>
__global__ void batchnorm_stats_kernel(const IO_T* __restrict__ input,
                                       COMPUTE_T* __restrict__ mean_out,
                                       COMPUTE_T* __restrict__ inv_std_out,
                                       COMPUTE_T* __restrict__ running_mean,
                                       COMPUTE_T* __restrict__ running_var, size_t N, size_t C,
                                       size_t S, COMPUTE_T momentum, COMPUTE_T epsilon) {
  int c = blockIdx.x;
  if (c >= C) return;

  size_t channel_stride = C * S;
  size_t channel_offset = c * S;
  size_t count = N * S;

  WelfordData<COMPUTE_T> thread;

  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    size_t n = i / S;
    size_t s = i % S;
    size_t idx = n * channel_stride + channel_offset + s;

    COMPUTE_T val = static_cast<COMPUTE_T>(input[idx]);

    thread.count += COMPUTE_T(1);
    COMPUTE_T delta = val - thread.mean;
    thread.mean += delta / thread.count;
    COMPUTE_T delta2 = val - thread.mean;
    thread.m2 += delta * delta2;
  }

  WelfordData<COMPUTE_T> result = blockReduceWelford(thread);

  if (threadIdx.x == 0) {
    COMPUTE_T mu = result.mean;

    COMPUTE_T var = result.m2 / result.count;

    mean_out[c] = mu;

    COMPUTE_T inv_std = compute_rsqrt(var + epsilon);
    inv_std_out[c] = inv_std;

    COMPUTE_T unbiased_var =
        (result.count > COMPUTE_T(1)) ? (result.m2 / (result.count - COMPUTE_T(1))) : COMPUTE_T(0);

    running_mean[c] = (COMPUTE_T(1) - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (COMPUTE_T(1) - momentum) * running_var[c] + momentum * unbiased_var;
  }
}

template <typename IO_T, typename COMPUTE_T>
__global__ void batchnorm_stats_kernel_vec(const IO_T* __restrict__ input,
                                           COMPUTE_T* __restrict__ mean_out,
                                           COMPUTE_T* __restrict__ inv_std_out,
                                           COMPUTE_T* __restrict__ running_mean,
                                           COMPUTE_T* __restrict__ running_var, size_t N, size_t C,
                                           size_t S, COMPUTE_T momentum, COMPUTE_T epsilon) {
  using VecT = typename VectoredTraits<IO_T>::type;
  constexpr int vec_size = VectoredTraits<IO_T>::size;

  int c = blockIdx.x;
  if (c >= C) return;

  size_t channel_stride = C * S;
  size_t channel_offset = c * S;
  size_t count = N * S;
  size_t num_vectors = count / vec_size;

  WelfordData<COMPUTE_T> thread;

  for (size_t i = threadIdx.x; i < num_vectors; i += blockDim.x) {
    size_t scalar_idx_start = i * vec_size;
    size_t n = scalar_idx_start / S;
    size_t s = scalar_idx_start % S;
    size_t idx = n * channel_stride + channel_offset + s;

    VecT val_vec = *reinterpret_cast<const VecT*>(&input[idx]);
    const IO_T* val_arr = reinterpret_cast<const IO_T*>(&val_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      COMPUTE_T val = static_cast<COMPUTE_T>(val_arr[k]);
      thread.count += COMPUTE_T(1);
      COMPUTE_T delta = val - thread.mean;
      thread.mean += delta / thread.count;
      COMPUTE_T delta2 = val - thread.mean;
      thread.m2 += delta * delta2;
    }
  }

  WelfordData<COMPUTE_T> result = blockReduceWelford(thread);

  if (threadIdx.x == 0) {
    COMPUTE_T mu = result.mean;
    COMPUTE_T var = result.m2 / result.count;
    mean_out[c] = mu;
    COMPUTE_T inv_std = compute_rsqrt(var + epsilon);
    inv_std_out[c] = inv_std;
    COMPUTE_T unbiased_var =
        (result.count > COMPUTE_T(1)) ? (result.m2 / (result.count - COMPUTE_T(1))) : COMPUTE_T(0);
    running_mean[c] = (COMPUTE_T(1) - momentum) * running_mean[c] + momentum * mu;
    running_var[c] = (COMPUTE_T(1) - momentum) * running_var[c] + momentum * unbiased_var;
  }
}

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void batchnorm_fwd_kernel(const IO_T* __restrict__ input,
                                     const COMPUTE_T* __restrict__ mean,
                                     const COMPUTE_T* __restrict__ inv_std,
                                     const PARAM_T* __restrict__ gamma,
                                     const PARAM_T* __restrict__ beta, IO_T* __restrict__ output,
                                     COMPUTE_T* __restrict__ normalized_cache, size_t N, size_t C,
                                     size_t S, bool affine) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = N * C * S;

  if (idx < total_elements) {
    int c = (idx / S) % C;

    COMPUTE_T mu = mean[c];
    COMPUTE_T istd = inv_std[c];
    COMPUTE_T x = static_cast<COMPUTE_T>(input[idx]);

    COMPUTE_T norm = (x - mu) * istd;

    if (normalized_cache) normalized_cache[idx] = norm;

    COMPUTE_T res = norm;
    if (affine) {
      res = res * static_cast<COMPUTE_T>(gamma[c]) + static_cast<COMPUTE_T>(beta[c]);
    }
    output[idx] = static_cast<IO_T>(res);
  }
}

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void batchnorm_fwd_kernel_vec(
    const IO_T* __restrict__ input, const COMPUTE_T* __restrict__ mean,
    const COMPUTE_T* __restrict__ inv_std, const PARAM_T* __restrict__ gamma,
    const PARAM_T* __restrict__ beta, IO_T* __restrict__ output,
    COMPUTE_T* __restrict__ normalized_cache, size_t N, size_t C, size_t S, bool affine) {
  using VecT = typename VectoredTraits<IO_T>::type;
  constexpr int vec_size = VectoredTraits<IO_T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_vectors = (N * C * S) / vec_size;

  if (idx < total_vectors) {
    size_t scalar_idx = idx * vec_size;
    int c = (scalar_idx / S) % C;

    COMPUTE_T mu = mean[c];
    COMPUTE_T istd = inv_std[c];
    COMPUTE_T g = (affine && gamma) ? static_cast<COMPUTE_T>(gamma[c]) : COMPUTE_T(1);
    COMPUTE_T b = (affine && beta) ? static_cast<COMPUTE_T>(beta[c]) : COMPUTE_T(0);

    VecT x_vec = reinterpret_cast<const VecT*>(input)[idx];
    const IO_T* x_arr = reinterpret_cast<const IO_T*>(&x_vec);

    VecT out_vec;
    IO_T* out_arr = reinterpret_cast<IO_T*>(&out_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      COMPUTE_T x = static_cast<COMPUTE_T>(x_arr[k]);
      COMPUTE_T norm = (x - mu) * istd;
      if (normalized_cache) normalized_cache[scalar_idx + k] = norm;

      COMPUTE_T res = norm;
      if (affine) {
        res = res * g + b;
      }
      out_arr[k] = static_cast<IO_T>(res);
    }

    reinterpret_cast<VecT*>(output)[idx] = out_vec;
  }
}

template <typename IO_T, typename COMPUTE_T>
__global__ void batchnorm_wgrad_bgrad_reduce_kernel(
    const IO_T* __restrict__ grad_output, const IO_T* __restrict__ input,
    const COMPUTE_T* __restrict__ batch_mean, const COMPUTE_T* __restrict__ batch_invar,
    COMPUTE_T* __restrict__ d_gamma, COMPUTE_T* __restrict__ d_beta, size_t N, size_t C, size_t S) {
  int c = blockIdx.x;
  if (c >= C) return;

  size_t count = N * S;
  COMPUTE_T sum_dy = COMPUTE_T(0);
  float sum_dy_x_norm = 0.0f;

  size_t stride = C * S;
  size_t offset = c * S;

  for (size_t i = threadIdx.x; i < count; i += blockDim.x) {
    size_t n = i / S;
    size_t s = i % S;
    size_t idx = n * stride + offset + s;

    COMPUTE_T dy = static_cast<COMPUTE_T>(grad_output[idx]);
    float x_hat = (static_cast<float>(input[idx]) - static_cast<float>(batch_mean[c])) *
                  static_cast<float>(batch_invar[c]);

    sum_dy += dy;
    sum_dy_x_norm += static_cast<float>(dy) * x_hat;
  }

  sum_dy = blockReduceSum(sum_dy);
  sum_dy_x_norm = blockReduceSum(sum_dy_x_norm);

  if (threadIdx.x == 0) {
    d_gamma[c] = sum_dy_x_norm;
    d_beta[c] = sum_dy;
  }
}

template <typename IO_T, typename COMPUTE_T>
__global__ void batchnorm_wgrad_bgrad_reduce_kernel_vec(
    const IO_T* __restrict__ grad_output, const IO_T* __restrict__ input,
    const COMPUTE_T* __restrict__ batch_mean, const COMPUTE_T* __restrict__ batch_invar,
    COMPUTE_T* __restrict__ d_gamma, COMPUTE_T* __restrict__ d_beta, size_t N, size_t C, size_t S) {
  using VecT = typename VectoredTraits<IO_T>::type;
  constexpr int vec_size = VectoredTraits<IO_T>::size;

  int c = blockIdx.x;
  if (c >= C) return;

  size_t count = N * S;
  size_t num_vectors = count / vec_size;

  COMPUTE_T sum_dy = COMPUTE_T(0);
  float sum_dy_x_norm = 0.0f;

  size_t stride = C * S;
  size_t offset = c * S;

  for (size_t i = threadIdx.x; i < num_vectors; i += blockDim.x) {
    size_t scalar_idx_start = i * vec_size;
    size_t n = scalar_idx_start / S;
    size_t s = scalar_idx_start % S;
    size_t idx = n * stride + offset + s;

    VecT dy_vec = *reinterpret_cast<const VecT*>(&grad_output[idx]);
    const IO_T* dy_arr = reinterpret_cast<const IO_T*>(&dy_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      COMPUTE_T dy = static_cast<COMPUTE_T>(dy_arr[k]);
      float x_hat = (static_cast<float>(input[idx + k]) - static_cast<float>(batch_mean[c])) *
                    static_cast<float>(batch_invar[c]);
      sum_dy += dy;
      sum_dy_x_norm += static_cast<float>(dy) * x_hat;
    }
  }

  sum_dy = blockReduceSum(sum_dy);
  sum_dy_x_norm = blockReduceSum(sum_dy_x_norm);

  if (threadIdx.x == 0) {
    d_gamma[c] = sum_dy_x_norm;
    d_beta[c] = sum_dy;
  }
}

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void batchnorm_dgrad_kernel(
    const IO_T* __restrict__ grad_output, const IO_T* __restrict__ input,
    const COMPUTE_T* __restrict__ batch_mean, const COMPUTE_T* __restrict__ batch_invar,
    const PARAM_T* __restrict__ gamma, const COMPUTE_T* __restrict__ d_gamma,
    const COMPUTE_T* __restrict__ d_beta, IO_T* __restrict__ grad_input, size_t N, size_t C,
    size_t S, bool affine) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_elements = N * C * S;

  if (idx < total_elements) {
    int c = (idx / S) % C;

    COMPUTE_T g = (affine && gamma) ? static_cast<COMPUTE_T>(gamma[c]) : COMPUTE_T(1);
    COMPUTE_T istd = batch_invar[c];

    float sum_dy = static_cast<float>(d_beta[c]);
    float sum_dy_x_norm = static_cast<float>(d_gamma[c]);
    float M = static_cast<float>(N * S);

    COMPUTE_T dy = static_cast<COMPUTE_T>(grad_output[idx]);
    float x_hat = (static_cast<float>(input[idx]) - static_cast<float>(batch_mean[c])) *
                  static_cast<float>(batch_invar[c]);

    float term1 = static_cast<float>(g * istd) / M;
    float term2 = M * static_cast<float>(dy) - sum_dy - (x_hat * sum_dy_x_norm);

    grad_input[idx] = static_cast<IO_T>(term1 * term2);
  }
}

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void batchnorm_dgrad_kernel_vec(
    const IO_T* __restrict__ grad_output, const IO_T* __restrict__ input,
    const COMPUTE_T* __restrict__ batch_mean, const COMPUTE_T* __restrict__ batch_invar,
    const PARAM_T* __restrict__ gamma, const COMPUTE_T* __restrict__ d_gamma,
    const COMPUTE_T* __restrict__ d_beta, IO_T* __restrict__ grad_input, size_t N, size_t C,
    size_t S, bool affine) {
  using VecT = typename VectoredTraits<IO_T>::type;
  constexpr int vec_size = VectoredTraits<IO_T>::size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t total_vectors = (N * C * S) / vec_size;

  if (idx < total_vectors) {
    size_t scalar_idx = idx * vec_size;
    int c = (scalar_idx / S) % C;

    COMPUTE_T g = (affine && gamma) ? static_cast<COMPUTE_T>(gamma[c]) : COMPUTE_T(1);
    COMPUTE_T istd = batch_invar[c];
    float sum_dy = static_cast<float>(d_beta[c]);
    float sum_dy_x_norm = static_cast<float>(d_gamma[c]);
    float M = static_cast<float>(N * S);

    float term1 = static_cast<float>(g * istd) / M;

    VecT dy_vec = reinterpret_cast<const VecT*>(grad_output)[idx];
    const IO_T* dy_arr = reinterpret_cast<const IO_T*>(&dy_vec);

    VecT dx_vec;
    IO_T* dx_arr = reinterpret_cast<IO_T*>(&dx_vec);

#pragma unroll
    for (int k = 0; k < vec_size; ++k) {
      COMPUTE_T dy = static_cast<COMPUTE_T>(dy_arr[k]);
      float x_hat =
          (static_cast<float>(input[scalar_idx + k]) - static_cast<float>(batch_mean[c])) *
          static_cast<float>(batch_invar[c]);
      float term2 = M * static_cast<float>(dy) - sum_dy - (x_hat * sum_dy_x_norm);
      dx_arr[k] = static_cast<IO_T>(term1 * term2);
    }

    reinterpret_cast<VecT*>(grad_input)[idx] = dx_vec;
  }
}

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void batchnorm_nchw_inf_kernel(const IO_T* input, const COMPUTE_T* running_mean,
                                          const COMPUTE_T* running_var, const PARAM_T* gamma,
                                          const PARAM_T* beta, IO_T* output, size_t batch_size,
                                          size_t channels, size_t spatial_size, COMPUTE_T epsilon,
                                          bool affine, bool use_relu) {
  extern __shared__ char shared_mem[];
  COMPUTE_T* s_mean = reinterpret_cast<COMPUTE_T*>(shared_mem);
  COMPUTE_T* s_inv_std = s_mean + channels;
  PARAM_T* s_gamma = reinterpret_cast<PARAM_T*>(s_inv_std + channels);
  PARAM_T* s_beta = s_gamma + channels;

  for (int c = threadIdx.x; c < channels; c += blockDim.x) {
    s_mean[c] = running_mean[c];
    COMPUTE_T var_val = running_var[c];
    s_inv_std[c] = compute_rsqrt(var_val + epsilon);
    if (affine) {
      s_gamma[c] = gamma[c];
      s_beta[c] = beta[c];
    }
  }
  __syncthreads();

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = batch_size * channels * spatial_size;

  if (idx >= total_elements) return;

  int c = (idx / spatial_size) % channels;

  float input_val = static_cast<float>(input[idx]);
  float normalized_val =
      (input_val - static_cast<float>(s_mean[c])) * static_cast<float>(s_inv_std[c]);

  float out_val;
  if (affine) {
    out_val = static_cast<float>(s_gamma[c]) * normalized_val + static_cast<float>(s_beta[c]);
  } else {
    out_val = normalized_val;
  }

  if (use_relu) {
    out_val = out_val > 0.0f ? out_val : 0.0f;
  }

  output[idx] = static_cast<IO_T>(out_val);
}

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void layernorm_fwd_kernel(const IO_T* input, IO_T* output, const PARAM_T* gamma,
                                     const PARAM_T* beta, size_t channels, COMPUTE_T epsilon) {
  size_t n = static_cast<size_t>(blockIdx.x);

  const IO_T* x = input + n * channels;
  IO_T* y = output + n * channels;

  COMPUTE_T sum = COMPUTE_T(0);
  for (size_t c = 0; c < channels; ++c) {
    sum += static_cast<COMPUTE_T>(x[c]);
  }
  const COMPUTE_T mean = sum / static_cast<COMPUTE_T>(channels);

  COMPUTE_T sq_sum = COMPUTE_T(0);
  for (size_t c = 0; c < channels; ++c) {
    const COMPUTE_T diff = static_cast<COMPUTE_T>(x[c]) - mean;
    sq_sum += diff * diff;
  }
  const COMPUTE_T var = sq_sum / static_cast<COMPUTE_T>(channels);
  const COMPUTE_T inv_std =
      COMPUTE_T(1) / static_cast<COMPUTE_T>(sqrt(static_cast<double>(var + epsilon)));

  for (size_t c = 0; c < channels; ++c) {
    const COMPUTE_T norm = (static_cast<COMPUTE_T>(x[c]) - mean) * inv_std;
    const COMPUTE_T g = gamma ? static_cast<COMPUTE_T>(gamma[c]) : COMPUTE_T(1);
    const COMPUTE_T b = beta ? static_cast<COMPUTE_T>(beta[c]) : COMPUTE_T(0);
    y[c] = static_cast<IO_T>(g * norm + b);
  }
}

template <typename IO_T, typename PARAM_T, typename COMPUTE_T>
__global__ void layernorm_bwd_kernel(const IO_T* grad_output, const IO_T* input,
                                     const PARAM_T* gamma, IO_T* grad_input, PARAM_T* grad_gamma,
                                     PARAM_T* grad_beta, size_t channels, COMPUTE_T epsilon) {
  size_t n = static_cast<size_t>(blockIdx.x);

  const IO_T* x = input + n * channels;
  const IO_T* go = grad_output + n * channels;
  IO_T* gi = grad_input ? (grad_input + n * channels) : nullptr;

  COMPUTE_T sum = COMPUTE_T(0);
  for (size_t c = 0; c < channels; ++c) {
    sum += static_cast<COMPUTE_T>(x[c]);
  }
  const COMPUTE_T mean = sum / static_cast<COMPUTE_T>(channels);

  COMPUTE_T sq_sum = COMPUTE_T(0);
  for (size_t c = 0; c < channels; ++c) {
    const COMPUTE_T diff = static_cast<COMPUTE_T>(x[c]) - mean;
    sq_sum += diff * diff;
  }
  const COMPUTE_T var = sq_sum / static_cast<COMPUTE_T>(channels);
  const COMPUTE_T inv_std =
      COMPUTE_T(1) / static_cast<COMPUTE_T>(sqrt(static_cast<double>(var + epsilon)));

  COMPUTE_T sum_dl_dnorm = COMPUTE_T(0);
  COMPUTE_T sum_dl_dnorm_x_hat = COMPUTE_T(0);

  for (size_t c = 0; c < channels; ++c) {
    const COMPUTE_T g = gamma ? static_cast<COMPUTE_T>(gamma[c]) : COMPUTE_T(1);
    const COMPUTE_T dl_dnorm = static_cast<COMPUTE_T>(go[c]) * g;
    const COMPUTE_T x_hat = (static_cast<COMPUTE_T>(x[c]) - mean) * inv_std;
    sum_dl_dnorm += dl_dnorm;
    sum_dl_dnorm_x_hat += dl_dnorm * x_hat;

    if constexpr (is_floating<PARAM_T>::value) {
      if (grad_gamma) {
        cuda::gpu_atomic_add(&grad_gamma[c],
                             static_cast<PARAM_T>(static_cast<COMPUTE_T>(go[c]) * x_hat));
      }
      if (grad_beta) {
        cuda::gpu_atomic_add(&grad_beta[c], static_cast<PARAM_T>(go[c]));
      }
    }
  }

  if (gi) {
    const COMPUTE_T mean_dl_dnorm = sum_dl_dnorm / static_cast<COMPUTE_T>(channels);
    const COMPUTE_T mean_dl_dnorm_x_hat = sum_dl_dnorm_x_hat / static_cast<COMPUTE_T>(channels);

    for (size_t c = 0; c < channels; ++c) {
      const COMPUTE_T g = gamma ? static_cast<COMPUTE_T>(gamma[c]) : COMPUTE_T(1);
      const COMPUTE_T dl_dnorm = static_cast<COMPUTE_T>(go[c]) * g;
      const COMPUTE_T x_hat = (static_cast<COMPUTE_T>(x[c]) - mean) * inv_std;
      gi[c] = static_cast<IO_T>(inv_std * (dl_dnorm - mean_dl_dnorm - x_hat * mean_dl_dnorm_x_hat));
    }
  }
}

WorkspaceReq CUDAEngine::query_batchnorm_graph(engine_handle backend_handle,
                                               const BatchNormStats& stats, DTypeDesc type_desc) {
  size_t temp = 2 * stats.channels * get_dtype_size(type_desc.param_dtype);
  return {0, temp, 0};
}

WorkspaceReq CUDAEngine::query_layernorm_graph(engine_handle backend_handle,
                                               const LayerNormStats& stats, DTypeDesc type_desc) {
  size_t temp = 2 * stats.channels * get_dtype_size(type_desc.param_dtype);
  return {0, temp, 0};
}

void CUDAEngine::batchnorm_fwd(engine_handle backend_handle, const BatchNormStats& stats,
                               const void* input, const void* gamma, const void* beta, void* output,
                               void* prev_running_mean, void* prev_running_var,
                               void* next_running_mean, void* next_running_var, void* batch_mean,
                               void* batch_invar, void* relu_mask, void* workspace,
                               DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE3(
      type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, IO_T, PARAM_T, COMPUTE_T,
      {
        constexpr int vec_size = VectoredTraits<IO_T>::size;
        if ((stats.height * stats.width) % vec_size == 0) {
          batchnorm_stats_kernel_vec<IO_T, COMPUTE_T><<<stats.channels, 256, 0, stream>>>(
              static_cast<const IO_T*>(input), static_cast<COMPUTE_T*>(batch_mean),
              static_cast<COMPUTE_T*>(batch_invar), static_cast<COMPUTE_T*>(next_running_mean),
              static_cast<COMPUTE_T*>(next_running_var), stats.batch_size, stats.channels,
              (stats.height * stats.width), stats.momentum, stats.epsilon);
          size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
          size_t total_vectors = total_elements / vec_size;
          int num_blocks = (total_vectors + 256 - 1) / 256;
          batchnorm_fwd_kernel_vec<IO_T, PARAM_T, COMPUTE_T><<<num_blocks, 256, 0, stream>>>(
              static_cast<const IO_T*>(input), static_cast<COMPUTE_T*>(batch_mean),
              static_cast<COMPUTE_T*>(batch_invar), static_cast<const PARAM_T*>(gamma),
              static_cast<const PARAM_T*>(beta), static_cast<IO_T*>(output), nullptr,
              stats.batch_size, stats.channels, (stats.height * stats.width), true);
        } else {
          batchnorm_stats_kernel<IO_T, COMPUTE_T><<<stats.channels, 256, 0, stream>>>(
              static_cast<const IO_T*>(input), static_cast<COMPUTE_T*>(batch_mean),
              static_cast<COMPUTE_T*>(batch_invar), static_cast<COMPUTE_T*>(next_running_mean),
              static_cast<COMPUTE_T*>(next_running_var), stats.batch_size, stats.channels,
              (stats.height * stats.width), stats.momentum, stats.epsilon);
          size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
          int num_blocks = (total_elements + 256 - 1) / 256;
          batchnorm_fwd_kernel<IO_T, PARAM_T, COMPUTE_T><<<num_blocks, 256, 0, stream>>>(
              static_cast<const IO_T*>(input), static_cast<COMPUTE_T*>(batch_mean),
              static_cast<COMPUTE_T*>(batch_invar), static_cast<const PARAM_T*>(gamma),
              static_cast<const PARAM_T*>(beta), static_cast<IO_T*>(output), nullptr,
              stats.batch_size, stats.channels, (stats.height * stats.width), true);
        }
      });
}

void CUDAEngine::batchnorm_infer(engine_handle backend_handle, const BatchNormStats& stats,
                                 const void* input, const void* gamma, const void* beta,
                                 const void* saved_mean, const void* saved_var, void* output,
                                 void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE3(
      type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, IO_T, PARAM_T, COMPUTE_T,
      {
        size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
        int threads_per_block = 256;
        int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        size_t shared_mem_size = 4 * stats.channels * sizeof(float);
        batchnorm_nchw_inf_kernel<IO_T, PARAM_T, COMPUTE_T>
            <<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
                static_cast<const IO_T*>(input), static_cast<const COMPUTE_T*>(saved_mean),
                static_cast<const COMPUTE_T*>(saved_var), static_cast<const PARAM_T*>(gamma),
                static_cast<const PARAM_T*>(beta), static_cast<IO_T*>(output), stats.batch_size,
                stats.channels, (stats.height * stats.width), static_cast<COMPUTE_T>(stats.epsilon),
                true, stats.use_relu);
      });
}

void CUDAEngine::batchnorm_bwd(engine_handle backend_handle, const BatchNormStats& stats,
                               const void* grad_output, const void* input, const void* relu_mask,
                               const void* gamma, void* grad_input, void* grad_gamma,
                               void* grad_beta, const void* batch_mean, const void* batch_invar,
                               void* workspace, DTypeDesc type_desc) {
  size_t grad_gamma_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  size_t grad_beta_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  void* grad_gamma_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_gamma_temp_size;
  void* grad_beta_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_beta_temp_size;

  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE3(
      type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, IO_T, PARAM_T, COMPUTE_T,
      {
        constexpr int vec_size = VectoredTraits<IO_T>::size;
        if ((stats.height * stats.width) % vec_size == 0) {
          batchnorm_wgrad_bgrad_reduce_kernel_vec<IO_T, COMPUTE_T>
              <<<stats.channels, 256, 0, stream>>>(
                  static_cast<const IO_T*>(grad_output), static_cast<const IO_T*>(input),
                  static_cast<const COMPUTE_T*>(batch_mean),
                  static_cast<const COMPUTE_T*>(batch_invar),
                  static_cast<COMPUTE_T*>(grad_gamma_temp), static_cast<COMPUTE_T*>(grad_beta_temp),
                  stats.batch_size, stats.channels, (stats.height * stats.width));
          size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
          size_t total_vectors = total_elements / vec_size;
          int num_blocks = (total_vectors + 256 - 1) / 256;
          batchnorm_dgrad_kernel_vec<IO_T, PARAM_T, COMPUTE_T><<<num_blocks, 256, 0, stream>>>(
              static_cast<const IO_T*>(grad_output), static_cast<const IO_T*>(input),
              static_cast<const COMPUTE_T*>(batch_mean), static_cast<const COMPUTE_T*>(batch_invar),
              static_cast<const PARAM_T*>(gamma), static_cast<COMPUTE_T*>(grad_gamma_temp),
              static_cast<COMPUTE_T*>(grad_beta_temp), static_cast<IO_T*>(grad_input),
              stats.batch_size, stats.channels, (stats.height * stats.width), true);
        } else {
          batchnorm_wgrad_bgrad_reduce_kernel<IO_T, COMPUTE_T><<<stats.channels, 256, 0, stream>>>(
              static_cast<const IO_T*>(grad_output), static_cast<const IO_T*>(input),
              static_cast<const COMPUTE_T*>(batch_mean), static_cast<const COMPUTE_T*>(batch_invar),
              static_cast<COMPUTE_T*>(grad_gamma_temp), static_cast<COMPUTE_T*>(grad_beta_temp),
              stats.batch_size, stats.channels, (stats.height * stats.width));
          size_t total_elements = stats.batch_size * stats.channels * (stats.height * stats.width);
          int num_blocks = (total_elements + 256 - 1) / 256;
          batchnorm_dgrad_kernel<IO_T, PARAM_T, COMPUTE_T><<<num_blocks, 256, 0, stream>>>(
              static_cast<const IO_T*>(grad_output), static_cast<const IO_T*>(input),
              static_cast<const COMPUTE_T*>(batch_mean), static_cast<const COMPUTE_T*>(batch_invar),
              static_cast<const PARAM_T*>(gamma), static_cast<COMPUTE_T*>(grad_gamma_temp),
              static_cast<COMPUTE_T*>(grad_beta_temp), static_cast<IO_T*>(grad_input),
              stats.batch_size, stats.channels, (stats.height * stats.width), true);
        }

        if (true) {
          int total_size = 1 * stats.channels;
          int num_blocks = (total_size + 256 - 1) / 256;
          norm_add_bias_kernel<PARAM_T, COMPUTE_T, COMPUTE_T><<<num_blocks, 256, 0, stream>>>(
              static_cast<PARAM_T*>(grad_gamma), static_cast<const COMPUTE_T*>(grad_gamma_temp), 1,
              stats.channels);
          norm_add_bias_kernel<PARAM_T, COMPUTE_T, COMPUTE_T><<<num_blocks, 256, 0, stream>>>(
              static_cast<PARAM_T*>(grad_beta), static_cast<const COMPUTE_T*>(grad_beta_temp), 1,
              stats.channels);
        }
      });
}

void CUDAEngine::layernorm_fwd(engine_handle backend_handle, const LayerNormStats& stats,
                               const void* input, const void* gamma, const void* beta, void* output,
                               void* mean, void* inv_variance, void* workspace,
                               DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE3(type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, IO_T, PARAM_T,
                  COMPUTE_T, {
                    if (stats.batch_size == 0 || stats.channels == 0) return;
                    dim3 blocks(static_cast<unsigned int>(stats.batch_size));
                    dim3 threads(1);
                    layernorm_fwd_kernel<IO_T, PARAM_T, COMPUTE_T><<<blocks, threads, 0, stream>>>(
                        static_cast<const IO_T*>(input), static_cast<IO_T*>(output),
                        static_cast<const PARAM_T*>(gamma), static_cast<const PARAM_T*>(beta),
                        stats.channels, static_cast<COMPUTE_T>(stats.epsilon));
                  });
}

void CUDAEngine::layernorm_infer(engine_handle backend_handle, const LayerNormStats& stats,
                                 const void* input, const void* gamma, const void* beta,
                                 void* output, void* workspace, DTypeDesc type_desc) {
  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE3(type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, IO_T, PARAM_T,
                  COMPUTE_T, {
                    if (stats.batch_size == 0 || stats.channels == 0) return;
                    dim3 blocks(static_cast<unsigned int>(stats.batch_size));
                    dim3 threads(1);
                    layernorm_fwd_kernel<IO_T, PARAM_T, COMPUTE_T><<<blocks, threads, 0, stream>>>(
                        static_cast<const IO_T*>(input), static_cast<IO_T*>(output),
                        static_cast<const PARAM_T*>(gamma), static_cast<const PARAM_T*>(beta),
                        stats.channels, static_cast<COMPUTE_T>(stats.epsilon));
                  });
}

void CUDAEngine::layernorm_bwd(engine_handle backend_handle, const LayerNormStats& stats,
                               const void* grad_output, const void* input, const void* gamma,
                               const void* mean, const void* inv_variance, void* grad_input,
                               void* grad_gamma_prev, void* grad_beta_prev, void* workspace,
                               DTypeDesc type_desc) {
  size_t grad_gamma_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  size_t grad_beta_temp_size = stats.channels * get_dtype_size(type_desc.param_dtype);
  void* grad_gamma_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_gamma_temp_size;
  void* grad_beta_temp = workspace;
  workspace = static_cast<char*>(workspace) + grad_beta_temp_size;

  cudaStream_t stream = *backend_handle.stream_as<cuda_stream>();
  DISPATCH_DTYPE3(
      type_desc.io_dtype, type_desc.param_dtype, type_desc.compute_dtype, IO_T, PARAM_T, COMPUTE_T,
      {
        if (stats.batch_size == 0 || stats.channels == 0) return;
        dim3 blocks(static_cast<unsigned int>(stats.batch_size));
        dim3 threads(1);
        layernorm_bwd_kernel<IO_T, PARAM_T, COMPUTE_T><<<blocks, threads, 0, stream>>>(
            static_cast<const IO_T*>(grad_output), static_cast<const IO_T*>(input),
            static_cast<const PARAM_T*>(gamma), static_cast<IO_T*>(grad_input),
            static_cast<PARAM_T*>(grad_gamma_temp), static_cast<PARAM_T*>(grad_beta_temp),
            stats.channels, static_cast<COMPUTE_T>(stats.epsilon));

        if (true) {
          int total_size = 1 * stats.channels;
          int num_blocks = (total_size + 256 - 1) / 256;
          norm_add_bias_kernel<PARAM_T, COMPUTE_T, COMPUTE_T><<<num_blocks, 256, 0, stream>>>(
              static_cast<PARAM_T*>(grad_gamma_prev),
              static_cast<const COMPUTE_T*>(grad_gamma_temp), 1, stats.channels);
          norm_add_bias_kernel<PARAM_T, COMPUTE_T, COMPUTE_T><<<num_blocks, 256, 0, stream>>>(
              static_cast<PARAM_T*>(grad_beta_prev), static_cast<const COMPUTE_T*>(grad_beta_temp),
              1, stats.channels);
        }
      });
}

}  // namespace tunx

#endif
