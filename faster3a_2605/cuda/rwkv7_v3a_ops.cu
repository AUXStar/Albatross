#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <climits>
#include <vector>

using dtype = at::Half;

namespace {

constexpr int LN_THREADS = 256;
constexpr int LN_SMALL_THREADS = 1024;
constexpr int LN_SMALL512_THREADS = 512;
constexpr int LN_SMALL_C = 4096;

inline int64_t ceil_div(int64_t n, int64_t d) {
  return (n + d - 1) / d;
}

inline void check_cublas(cublasStatus_t status, const char* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with cublas status ", static_cast<int>(status));
}

inline void check_cublaslt(cublasStatus_t status, const char* what) {
  TORCH_CHECK(status == CUBLAS_STATUS_SUCCESS, what, " failed with cublasLt status ", static_cast<int>(status));
}

template <int Act>
__device__ __forceinline__ float apply_act(float x) {
  if constexpr (Act == 1) {
    return tanhf(x);
  } else {
    return 1.0f / (1.0f + expf(-x));
  }
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

template <int Threads>
__device__ __forceinline__ float block_sum_t(float x) {
  __shared__ float partial[Threads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  x = warp_sum(x);
  if (lane == 0) {
    partial[warp] = x;
  }
  __syncthreads();
  x = (threadIdx.x < (Threads / 32)) ? partial[lane] : 0.0f;
  if (warp == 0) {
    x = warp_sum(x);
  }
  if (threadIdx.x == 0) {
    partial[0] = x;
  }
  __syncthreads();
  return partial[0];
}

__global__ void identity_kernel(const dtype* __restrict__ x, dtype* __restrict__ y, int64_t n_vec4) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_vec4) {
    reinterpret_cast<int4*>(y)[i] = reinterpret_cast<const int4*>(x)[i];
  }
}

__global__ void add_f16_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ y,
    dtype* __restrict__ out,
    int64_t n_pairs) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n_pairs) {
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[i]);
    const float2 yv = __half22float2(reinterpret_cast<const __half2*>(y)[i]);
    reinterpret_cast<__half2*>(out)[i] = __floats2half2_rn(xv.x + yv.x, xv.y + yv.y);
  }
}

__global__ void advance_i32_kernel(int* __restrict__ x, int amount, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    x[i] += amount;
  }
}

template <int ChunkK, int Warps>
__global__ __launch_bounds__(128, 2) void linear_f16_m1_splitk_partial_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,
    float* __restrict__ partial) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair = (blockIdx.x * Warps + warp) * 32 + lane;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  const int k0 = blockIdx.y * ChunkK;
  const int k1 = min(k0 + ChunkK, K);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int k = k0; k < k1; ++k) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight + static_cast<int64_t>(k) * N + n));
    acc0 = fmaf(xv, wv.x, acc0);
    acc1 = fmaf(xv, wv.y, acc1);
  }
  reinterpret_cast<float2*>(partial + static_cast<int64_t>(blockIdx.y) * N)[pair] = make_float2(acc0, acc1);
}

__global__ void linear_f16_m1_splitk_reduce_kernel(
    int chunks,
    int N,
    const float* __restrict__ partial,
    dtype* __restrict__ y) {
  const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int c = 0; c < chunks; ++c) {
    const float2 v = reinterpret_cast<const float2*>(partial + static_cast<int64_t>(c) * N)[pair];
    acc0 += v.x;
    acc1 += v.y;
  }
  reinterpret_cast<__half2*>(y)[pair] = __floats2half2_rn(acc0, acc1);
}

template <int ChunkK, int Warps>
__global__ __launch_bounds__(128, 2) void linear_f16_rows_splitk_partial_kernel(
    int K,
    int N,
    int chunks,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,
    float* __restrict__ partial) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair = (blockIdx.x * Warps + warp) * 32 + lane;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  const int chunk = blockIdx.y;
  const int m = blockIdx.z;
  const int k0 = chunk * ChunkK;
  const int k1 = min(k0 + ChunkK, K);
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int k = k0; k < k1; ++k) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight + static_cast<int64_t>(k) * N + n));
    acc0 = fmaf(xv, wv.x, acc0);
    acc1 = fmaf(xv, wv.y, acc1);
  }
  reinterpret_cast<float2*>(partial + (static_cast<int64_t>(m) * chunks + chunk) * N)[pair] = make_float2(acc0, acc1);
}

__global__ void linear_f16_rows_splitk_reduce_kernel(
    int chunks,
    int N,
    const float* __restrict__ partial,
    dtype* __restrict__ y) {
  const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int m = blockIdx.y;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int c = 0; c < chunks; ++c) {
    const float2 v = reinterpret_cast<const float2*>(partial + (static_cast<int64_t>(m) * chunks + c) * N)[pair];
    acc0 += v.x;
    acc1 += v.y;
  }
  reinterpret_cast<__half2*>(y + static_cast<int64_t>(m) * N)[pair] = __floats2half2_rn(acc0, acc1);
}

template <int ChunkK, int Warps>
__global__ __launch_bounds__(128, 2) void linear_mix_f16_m1_splitk_partial_kernel(
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ shift,
    const dtype* __restrict__ mix,
    const dtype* __restrict__ weight,
    float* __restrict__ partial) {
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pair = (blockIdx.x * Warps + warp) * 32 + lane;
  const int n = pair << 1;
  if (n >= N) {
    return;
  }
  const int k0 = blockIdx.y * ChunkK;
  const int k1 = min(k0 + ChunkK, K);
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  for (int k = k0; k < k1; ++k) {
    const float xv0 = __half2float(*reinterpret_cast<const __half*>(x + k));
    const float sv0 = __half2float(*reinterpret_cast<const __half*>(shift + k));
    const float mv0 = __half2float(*reinterpret_cast<const __half*>(mix + k));
    const float xv = fmaf(sv0 - xv0, mv0, xv0);
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight + static_cast<int64_t>(k) * N + n));
    acc0 = fmaf(xv, wv.x, acc0);
    acc1 = fmaf(xv, wv.y, acc1);
  }
  reinterpret_cast<float2*>(partial + static_cast<int64_t>(blockIdx.y) * N)[pair] = make_float2(acc0, acc1);
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_wag_rank_in_mix_f16_kernel(
    int K,
    int Rw,
    int Ra,
    int Rg,
    int Rmax,
    const dtype* __restrict__ x,
    const dtype* __restrict__ shift,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    const dtype* __restrict__ w1_t,
    const dtype* __restrict__ a1_t,
    const dtype* __restrict__ g1_t,
    dtype* __restrict__ w1,
    dtype* __restrict__ a1,
    dtype* __restrict__ g1) {
  const int r = blockIdx.x;
  const int group = blockIdx.z;
  int R = Rw;
  const dtype* mix = x_w;
  const dtype* wt = w1_t;
  dtype* y = w1;
  if (group == 1) {
    R = Ra;
    mix = x_a;
    wt = a1_t;
    y = a1;
  } else if (group == 2) {
    R = Rg;
    mix = x_g;
    wt = g1_t;
    y = g1;
  }
  if (r >= R || r >= Rmax) {
    return;
  }
  float acc = 0.0f;
  const dtype* w_row = wt + static_cast<int64_t>(r) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 sv0 = __half22float2(*reinterpret_cast<const __half2*>(shift + k));
    const float2 mv0 = __half22float2(*reinterpret_cast<const __half2*>(mix + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + k));
    acc = fmaf(fmaf(sv0.x - xv0.x, mv0.x, xv0.x), wv.x, acc);
    acc = fmaf(fmaf(sv0.y - xv0.y, mv0.y, xv0.y), wv.y, acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + r) = __float2half_rn(acc);
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_wagv_rank_in_mix_f16_kernel(
    int K,
    int Rw,
    int Ra,
    int Rg,
    int Rv,
    int Rmax,
    const dtype* __restrict__ x,
    const dtype* __restrict__ shift,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ w1_t,
    const dtype* __restrict__ a1_t,
    const dtype* __restrict__ g1_t,
    const dtype* __restrict__ v1_t,
    dtype* __restrict__ w1,
    dtype* __restrict__ a1,
    dtype* __restrict__ g1,
    dtype* __restrict__ v1) {
  const int r = blockIdx.x;
  const int group = blockIdx.z;
  int R = Rw;
  const dtype* mix = x_w;
  const dtype* wt = w1_t;
  dtype* y = w1;
  if (group == 1) {
    R = Ra;
    mix = x_a;
    wt = a1_t;
    y = a1;
  } else if (group == 2) {
    R = Rg;
    mix = x_g;
    wt = g1_t;
    y = g1;
  } else if (group == 3) {
    R = Rv;
    mix = x_v;
    wt = v1_t;
    y = v1;
  }
  if (r >= R || r >= Rmax) {
    return;
  }
  float acc = 0.0f;
  const dtype* w_row = wt + static_cast<int64_t>(r) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv0 = __half22float2(*reinterpret_cast<const __half2*>(x + k));
    const float2 sv0 = __half22float2(*reinterpret_cast<const __half2*>(shift + k));
    const float2 mv0 = __half22float2(*reinterpret_cast<const __half2*>(mix + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + k));
    acc = fmaf(fmaf(sv0.x - xv0.x, mv0.x, xv0.x), wv.x, acc);
    acc = fmaf(fmaf(sv0.y - xv0.y, mv0.y, xv0.y), wv.y, acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + r) = __float2half_rn(acc);
  }
}

__global__ void copy_m1_to_shift_f16_kernel(const dtype* __restrict__ x, dtype* __restrict__ shift, int64_t pairs) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < pairs) {
    reinterpret_cast<__half2*>(shift)[i] = reinterpret_cast<const __half2*>(x)[i];
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n = blockIdx.x;
  const int m = blockIdx.y;
  if (m >= M || n >= N) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = weight_t + static_cast<int64_t>(n) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + (k2 << 1)));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + (k2 << 1)));
    acc = fmaf(xv.x, wv.x, acc);
    acc = fmaf(xv.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(acc);
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_ntile_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_t + static_cast<int64_t>(n) * K + k));
        acc[j] = fmaf(xv.x, wv.x, acc[j]);
        acc[j] = fmaf(xv.y, wv.y, acc[j]);
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + K - 1));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + K - 1)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_f16_ntile_scalar_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile, int Act>
__global__ __launch_bounds__(Threads, 2) void linear_t_act_f16_ntile_scalar_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const float xv = apply_act<Act>(__half2float(*reinterpret_cast<const __half*>(x_row + k)));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile, int Act>
__global__ __launch_bounds__(Threads, 2) void linear_t_act_f16_ntile_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    xv.x = apply_act<Act>(xv.x);
    xv.y = apply_act<Act>(xv.y);
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_t + static_cast<int64_t>(n) * K + k));
        acc[j] = fmaf(xv.x, wv.x, acc[j]);
        acc[j] = fmaf(xv.y, wv.y, acc[j]);
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
    const float xv = apply_act<Act>(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + K - 1)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * N + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_wag_rank_in_f16_kernel(
    int M,
    int K,
    int Rw,
    int Ra,
    int Rg,
    int Rmax,
    const dtype* __restrict__ xw,
    const dtype* __restrict__ xa,
    const dtype* __restrict__ xg,
    const dtype* __restrict__ w1_t,
    const dtype* __restrict__ a1_t,
    const dtype* __restrict__ g1_t,
    dtype* __restrict__ w1,
    dtype* __restrict__ a1,
    dtype* __restrict__ g1) {
  const int r = blockIdx.x;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int R = Rw;
  const dtype* x = xw;
  const dtype* wt = w1_t;
  dtype* y = w1;
  if (group == 1) {
    R = Ra;
    x = xa;
    wt = a1_t;
    y = a1;
  } else if (group == 2) {
    R = Rg;
    x = xg;
    wt = g1_t;
    y = g1;
  }
  if (m >= M || r >= R || r >= Rmax) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = wt + static_cast<int64_t>(r) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + k));
    acc = fmaf(xv.x, wv.x, acc);
    acc = fmaf(xv.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * R + r) = __float2half_rn(acc);
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 2) void linear_wagv_rank_in_f16_kernel(
    int M,
    int K,
    int Rw,
    int Ra,
    int Rg,
    int Rv,
    int Rmax,
    const dtype* __restrict__ xw,
    const dtype* __restrict__ xa,
    const dtype* __restrict__ xg,
    const dtype* __restrict__ xv,
    const dtype* __restrict__ w1_t,
    const dtype* __restrict__ a1_t,
    const dtype* __restrict__ g1_t,
    const dtype* __restrict__ v1_t,
    dtype* __restrict__ w1,
    dtype* __restrict__ a1,
    dtype* __restrict__ g1,
    dtype* __restrict__ v1) {
  const int r = blockIdx.x;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int R = Rw;
  const dtype* x = xw;
  const dtype* wt = w1_t;
  dtype* y = w1;
  if (group == 1) {
    R = Ra;
    x = xa;
    wt = a1_t;
    y = a1;
  } else if (group == 2) {
    R = Rg;
    x = xg;
    wt = g1_t;
    y = g1;
  } else if (group == 3) {
    R = Rv;
    x = xv;
    wt = v1_t;
    y = v1;
  }
  if (m >= M || r >= R || r >= Rmax) {
    return;
  }
  float acc = 0.0f;
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const dtype* w_row = wt + static_cast<int64_t>(r) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv2 = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
    const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(w_row + k));
    acc = fmaf(xv2.x, wv.x, acc);
    acc = fmaf(xv2.y, wv.y, acc);
  }
  if ((K & 1) && threadIdx.x == 0) {
    acc = fmaf(__half2float(*reinterpret_cast<const __half*>(x_row + K - 1)),
               __half2float(*reinterpret_cast<const __half*>(w_row + K - 1)),
               acc);
  }
  acc = block_sum_t<Threads>(acc);
  if (threadIdx.x == 0) {
    *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * R + r) = __float2half_rn(acc);
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_wag_rank_out_f16_kernel(
    int M,
    int C,
    int Kw,
    int Ka,
    int Kg,
    const dtype* __restrict__ w1,
    const dtype* __restrict__ a1,
    const dtype* __restrict__ g1,
    const dtype* __restrict__ w2_t,
    const dtype* __restrict__ a2_t,
    const dtype* __restrict__ g2_t,
    dtype* __restrict__ w,
    dtype* __restrict__ a,
    dtype* __restrict__ g) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int K = Kw;
  const dtype* x = w1;
  const dtype* wt = w2_t;
  dtype* y = w;
  if (group == 1) {
    K = Ka;
    x = a1;
    wt = a2_t;
    y = a;
  } else if (group == 2) {
    K = Kg;
    x = g1;
    wt = g2_t;
    y = g;
  }
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    if (group == 0) {
      xv = tanhf(xv);
    } else if (group == 2) {
      xv = 1.0f / (1.0f + expf(-xv));
    }
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < C) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(wt + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int u = 0; u < Threads / 32; ++u) {
        sum += partial[u][j];
      }
      const int n = n0 + j;
      if (n < C) {
        *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * C + n) = __float2half_rn(sum);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_wagv_rank_out_f16_kernel(
    int M,
    int C,
    int Kw,
    int Ka,
    int Kg,
    int Kv,
    const dtype* __restrict__ w1,
    const dtype* __restrict__ a1,
    const dtype* __restrict__ g1,
    const dtype* __restrict__ v1,
    const dtype* __restrict__ w2_t,
    const dtype* __restrict__ a2_t,
    const dtype* __restrict__ g2_t,
    const dtype* __restrict__ v2_t,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    dtype* __restrict__ w,
    dtype* __restrict__ a,
    dtype* __restrict__ g,
    dtype* __restrict__ v_out) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  const int group = blockIdx.z;
  int K = Kw;
  const dtype* x = w1;
  const dtype* wt = w2_t;
  dtype* y = w;
  if (group == 1) {
    K = Ka;
    x = a1;
    wt = a2_t;
    y = a;
  } else if (group == 2) {
    K = Kg;
    x = g1;
    wt = g2_t;
    y = g;
  } else if (group == 3) {
    K = Kv;
    x = v1;
    wt = v2_t;
    y = v_out;
  }
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
    if (group == 0) {
      xv = tanhf(xv);
    } else if (group == 2) {
      xv = 1.0f / (1.0f + expf(-xv));
    }
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < C) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(wt + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int u = 0; u < Threads / 32; ++u) {
        sum += partial[u][j];
      }
      const int n = n0 + j;
      if (n < C) {
        if (group == 3) {
          const int64_t idx = static_cast<int64_t>(m) * C + n;
          const float vv = __half2float(*reinterpret_cast<const __half*>(v + idx));
          const float vf = __half2float(*reinterpret_cast<const __half*>(v_first + idx));
          const float gate = 1.0f / (1.0f + expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) + sum)));
          *reinterpret_cast<__half*>(y + idx) = __float2half_rn(vv + (vf - vv) * gate);
        } else {
          *reinterpret_cast<__half*>(y + static_cast<int64_t>(m) * C + n) = __float2half_rn(sum);
        }
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_vres_f16_ntile_scalar_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  for (int k = threadIdx.x; k < K; k += Threads) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + k)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        const int64_t idx = static_cast<int64_t>(m) * N + n;
        const float vv = __half2float(*reinterpret_cast<const __half*>(v + idx));
        const float vf = __half2float(*reinterpret_cast<const __half*>(v_first + idx));
        const float gate = 1.0f / (1.0f + expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) + sum)));
        *reinterpret_cast<__half*>(y + idx) = __float2half_rn(vv + (vf - vv) * gate);
      }
    }
  }
}

template <int Threads, int OutTile>
__global__ __launch_bounds__(Threads, 2) void linear_t_vres_f16_ntile_kernel(
    int M,
    int K,
    int N,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight_t,
    const dtype* __restrict__ v,
    const dtype* __restrict__ v_first,
    const dtype* __restrict__ v0,
    dtype* __restrict__ y) {
  const int n0 = blockIdx.x * OutTile;
  const int m = blockIdx.y;
  if (m >= M) {
    return;
  }
  float acc[OutTile];
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = 0.0f;
  }
  const dtype* x_row = x + static_cast<int64_t>(m) * K;
  const int K2 = K >> 1;
  for (int k2 = threadIdx.x; k2 < K2; k2 += Threads) {
    const int k = k2 << 1;
    const float2 xv = __half22float2(*reinterpret_cast<const __half2*>(x_row + k));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        const float2 wv = __half22float2(*reinterpret_cast<const __half2*>(weight_t + static_cast<int64_t>(n) * K + k));
        acc[j] = fmaf(xv.x, wv.x, acc[j]);
        acc[j] = fmaf(xv.y, wv.y, acc[j]);
      }
    }
  }
  if ((K & 1) && threadIdx.x == 0) {
    const float xv = __half2float(*reinterpret_cast<const __half*>(x_row + K - 1));
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      const int n = n0 + j;
      if (n < N) {
        acc[j] = fmaf(xv, __half2float(*reinterpret_cast<const __half*>(weight_t + static_cast<int64_t>(n) * K + K - 1)), acc[j]);
      }
    }
  }
  __shared__ float partial[Threads / 32][OutTile];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
#pragma unroll
  for (int j = 0; j < OutTile; ++j) {
    acc[j] = warp_sum(acc[j]);
    if (lane == 0) {
      partial[warp][j] = acc[j];
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
#pragma unroll
    for (int j = 0; j < OutTile; ++j) {
      float sum = 0.0f;
#pragma unroll
      for (int w = 0; w < Threads / 32; ++w) {
        sum += partial[w][j];
      }
      const int n = n0 + j;
      if (n < N) {
        const int64_t idx = static_cast<int64_t>(m) * N + n;
        const float vv = __half2float(*reinterpret_cast<const __half*>(v + idx));
        const float vf = __half2float(*reinterpret_cast<const __half*>(v_first + idx));
        const float gate = 1.0f / (1.0f + expf(-(__half2float(*reinterpret_cast<const __half*>(v0 + n)) + sum)));
        *reinterpret_cast<__half*>(y + idx) = __float2half_rn(vv + (vf - vv) * gate);
      }
    }
  }
}

__global__ void layer_norm_f16_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * C;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
    sum += v;
  }
  sum = block_sum_t<LN_THREADS>(sum);
  const float inv_c = 1.0f / static_cast<float>(C);
  const float mean = sum * inv_c;
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<LN_THREADS>(sum_var);
  const float var = sum_var * inv_c;
  const float rstd = rsqrtf(var + eps);
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
    const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
    const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
    *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
  }
}

__global__ void add_layer_norm_f16_kernel(
    int C,
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ x_out,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * C;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    sum += v;
  }
  sum = block_sum_t<LN_THREADS>(sum);
  const float inv_c = 1.0f / static_cast<float>(C);
  const float mean = sum * inv_c;
  float sum_var = 0.0f;
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float d = v - mean;
    sum_var += d * d;
  }
  sum_var = block_sum_t<LN_THREADS>(sum_var);
  const float rstd = rsqrtf(sum_var * inv_c + eps);
  for (int c = threadIdx.x; c < C; c += blockDim.x) {
    const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                    __half2float(*reinterpret_cast<const __half*>(residual + base + c));
    const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
    const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
    *reinterpret_cast<__half*>(x_out + base + c) = __float2half_rn(v);
    *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__ __launch_bounds__(Threads, 1) void layer_norm_f16_small_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      sum += v.x + v.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float dx = v.x - mean;
      const float dy = v.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 v = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(y + base)[idx] = __floats2half2_rn(
          (v.x - mean) * rstd * w.x + b.x,
          (v.y - mean) * rstd * w.y + b.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c));
      const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
    }
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_f16_small_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ x_out,
    dtype* __restrict__ y,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + base)[idx]);
      sum += xv.x + rv.x + xv.y + rv.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + base)[idx]);
      const float dx = xv.x + rv.x - mean;
      const float dy = xv.y + rv.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + base)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + base)[idx]);
      const float sx = xv.x + rv.x;
      const float sy = xv.y + rv.y;
      const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(x_out + base)[idx] = __floats2half2_rn(sx, sy);
      reinterpret_cast<__half2*>(y + base)[idx] = __floats2half2_rn(
          (sx - mean) * rstd * w.x + b.x,
          (sy - mean) * rstd * w.y + b.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + base + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + base + c));
      const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float b = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(x_out + base + c) = __float2half_rn(v);
      *reinterpret_cast<__half*>(y + base + c) = __float2half_rn((v - mean) * rstd * w + b);
    }
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_cmix_mix_f16_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_k,
    dtype* __restrict__ x_out,
    dtype* __restrict__ mixed,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base = row * LN_SMALL_C;
  float sum = 0.0f;
  const int64_t base2 = base >> 1;
  constexpr int pairs = LN_SMALL_C >> 1;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    sum += xv.x + rv.x + xv.y + rv.y;
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const float d0 = x0 - mean;
    const float d1 = x1 - mean;
    sum_var += d0 * d0 + d1 * d1;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float2 mix = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(mixed)[base2 + p] =
        __floats2half2_rn(yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads>
__global__ __launch_bounds__(Threads, 1) void add_layer_norm_tmix_mix6_f16_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    dtype* __restrict__ shift_state,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    const dtype* __restrict__ x_r,
    const dtype* __restrict__ x_w,
    const dtype* __restrict__ x_k,
    const dtype* __restrict__ x_v,
    const dtype* __restrict__ x_a,
    const dtype* __restrict__ x_g,
    dtype* __restrict__ x_out,
    dtype* __restrict__ out_r,
    dtype* __restrict__ out_w,
    dtype* __restrict__ out_k,
    dtype* __restrict__ out_v,
    dtype* __restrict__ out_a,
    dtype* __restrict__ out_g,
    int64_t rows,
    float eps) {
  const int64_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }
  const int64_t base2 = row * (LN_SMALL_C >> 1);
  constexpr int pairs = LN_SMALL_C >> 1;
  float sum = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    sum += xv.x + rv.x + xv.y + rv.y;
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const float d0 = x0 - mean;
    const float d1 = x1 - mean;
    sum_var += d0 * d0 + d1 * d1;
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
#pragma unroll
  for (int k = 0; k < pairs / Threads; ++k) {
    const int p = threadIdx.x + k * Threads;
    const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x)[base2 + p]);
    const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual)[base2 + p]);
    const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[p]);
    const float2 b = __half22float2(reinterpret_cast<const __half2*>(bias)[p]);
    const float2 prev = __half22float2(reinterpret_cast<const __half2*>(shift_state)[base2 + p]);
    const float x0 = xv.x + rv.x;
    const float x1 = xv.y + rv.y;
    const __half2 y2 = __floats2half2_rn((x0 - mean) * rstd * w.x + b.x, (x1 - mean) * rstd * w.y + b.y);
    const float2 yv = __half22float2(y2);
    const float dx0 = prev.x - yv.x;
    const float dx1 = prev.y - yv.y;
    const float2 mr = __half22float2(reinterpret_cast<const __half2*>(x_r)[p]);
    const float2 mw = __half22float2(reinterpret_cast<const __half2*>(x_w)[p]);
    const float2 mk = __half22float2(reinterpret_cast<const __half2*>(x_k)[p]);
    const float2 mv = __half22float2(reinterpret_cast<const __half2*>(x_v)[p]);
    const float2 ma = __half22float2(reinterpret_cast<const __half2*>(x_a)[p]);
    const float2 mg = __half22float2(reinterpret_cast<const __half2*>(x_g)[p]);
    reinterpret_cast<__half2*>(x_out)[base2 + p] = __floats2half2_rn(x0, x1);
    reinterpret_cast<__half2*>(out_r)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
    reinterpret_cast<__half2*>(out_w)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
    reinterpret_cast<__half2*>(out_k)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
    reinterpret_cast<__half2*>(out_v)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
    reinterpret_cast<__half2*>(out_a)[base2 + p] = __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
    reinterpret_cast<__half2*>(out_g)[base2 + p] = __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
    reinterpret_cast<__half2*>(shift_state)[base2 + p] = y2;
  }
}

template <int Threads, bool VecStats, bool VecOut>
__global__ __launch_bounds__(Threads, 1) void add_last_layer_norm_f16_small_kernel(
    const dtype* __restrict__ x,
    const dtype* __restrict__ residual,
    const dtype* __restrict__ weight,
    const dtype* __restrict__ bias,
    dtype* __restrict__ y,
    int64_t B,
    int64_t T,
    float eps) {
  const int64_t bidx = blockIdx.x;
  if (bidx >= B) {
    return;
  }
  const int64_t src = (bidx * T + (T - 1)) * LN_SMALL_C;
  const int64_t dst = bidx * LN_SMALL_C;
  float sum = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      sum += xv.x + rv.x + xv.y + rv.y;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      sum += v;
    }
  }
  sum = block_sum_t<Threads>(sum);
  const float mean = sum * (1.0f / static_cast<float>(LN_SMALL_C));
  float sum_var = 0.0f;
  if constexpr (VecStats) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      const float dx = xv.x + rv.x - mean;
      const float dy = xv.y + rv.y - mean;
      sum_var += dx * dx + dy * dy;
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      const float d = v - mean;
      sum_var += d * d;
    }
  }
  sum_var = block_sum_t<Threads>(sum_var);
  const float rstd = rsqrtf(sum_var * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);
  if constexpr (VecOut) {
#pragma unroll
    for (int k = 0; k < (LN_SMALL_C / 2) / Threads; ++k) {
      const int idx = threadIdx.x + k * Threads;
      const float2 xv = __half22float2(reinterpret_cast<const __half2*>(x + src)[idx]);
      const float2 rv = __half22float2(reinterpret_cast<const __half2*>(residual + src)[idx]);
      const float sx = xv.x + rv.x;
      const float sy = xv.y + rv.y;
      const float2 w = __half22float2(reinterpret_cast<const __half2*>(weight)[idx]);
      const float2 bb = __half22float2(reinterpret_cast<const __half2*>(bias)[idx]);
      reinterpret_cast<__half2*>(y + dst)[idx] = __floats2half2_rn(
          (sx - mean) * rstd * w.x + bb.x,
          (sy - mean) * rstd * w.y + bb.y);
    }
  } else {
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / Threads; ++k) {
      const int c = threadIdx.x + k * Threads;
      const float v = __half2float(*reinterpret_cast<const __half*>(x + src + c)) +
                      __half2float(*reinterpret_cast<const __half*>(residual + src + c));
      const float w = __half2float(*reinterpret_cast<const __half*>(weight + c));
      const float bb = __half2float(*reinterpret_cast<const __half*>(bias + c));
      *reinterpret_cast<__half*>(y + dst + c) = __float2half_rn((v - mean) * rstd * w + bb);
    }
  }
}

} // namespace

at::Tensor identity_cuda(at::Tensor x) {
  TORCH_CHECK((x.numel() % 8) == 0, "x.numel() must be divisible by 8");
  auto y = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t n_vec4 = x.numel() / 8;
  auto stream = at::cuda::getCurrentCUDAStream();
  identity_kernel<<<static_cast<int>(ceil_div(n_vec4, threads)), threads, 0, stream>>>(
      x.data_ptr<dtype>(), y.data_ptr<dtype>(), n_vec4);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor add_f16_cuda(at::Tensor x, at::Tensor y) {
  TORCH_CHECK((x.numel() % 2) == 0, "add_f16 requires even numel");
  auto out = at::empty_like(x);
  constexpr int threads = 256;
  const int64_t n_pairs = x.numel() / 2;
  auto stream = at::cuda::getCurrentCUDAStream();
  add_f16_kernel<<<static_cast<int>(ceil_div(n_pairs, threads)), threads, 0, stream>>>(
      x.data_ptr<dtype>(), y.data_ptr<dtype>(), out.data_ptr<dtype>(), n_pairs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void advance_i32_cuda(at::Tensor x, int64_t amount) {
  TORCH_CHECK(amount >= INT_MIN && amount <= INT_MAX, "advance_i32 amount out of int range");
  constexpr int threads = 256;
  const int64_t n = x.numel();
  auto stream = at::cuda::getCurrentCUDAStream();
  advance_i32_kernel<<<static_cast<int>(ceil_div(n, threads)), threads, 0, stream>>>(
      x.data_ptr<int>(), static_cast<int>(amount), n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void copy_m1_to_shift_f16_cuda(at::Tensor x, at::Tensor shift) {
  TORCH_CHECK((x.numel() & 1) == 0 && shift.numel() == x.numel(), "copy_m1_to_shift_f16 shape mismatch");
  auto stream = at::cuda::getCurrentCUDAStream();
  C10_CUDA_CHECK(cudaMemcpyAsync(
      shift.data_ptr<dtype>(), x.data_ptr<dtype>(), static_cast<size_t>(x.numel()) * sizeof(dtype),
      cudaMemcpyDeviceToDevice, stream));
}

at::Tensor layer_norm_f16_cuda(at::Tensor x, at::Tensor weight, at::Tensor bias, double eps) {
  auto y = at::empty_like(x);
  const int64_t c64 = x.size(-1);
  TORCH_CHECK(c64 <= INT_MAX, "C too large");
  const int C = static_cast<int>(c64);
  const int64_t rows = x.numel() / C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C == LN_SMALL_C) {
    if (rows >= 1024) {
      layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(),
          weight.data_ptr<dtype>(),
          bias.data_ptr<dtype>(),
          y.data_ptr<dtype>(),
          rows,
          static_cast<float>(eps));
    } else if (rows >= 512) {
      layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(),
          weight.data_ptr<dtype>(),
          bias.data_ptr<dtype>(),
          y.data_ptr<dtype>(),
          rows,
          static_cast<float>(eps));
    } else {
      layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(),
          weight.data_ptr<dtype>(),
          bias.data_ptr<dtype>(),
          y.data_ptr<dtype>(),
          rows,
          static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
  }
  layer_norm_f16_kernel<<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
      C,
      x.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      y.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor layer_norm_f16_small_cuda(at::Tensor x, at::Tensor weight, at::Tensor bias, double eps) {
  auto y = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
      x.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      y.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor layer_norm_f16_small512_cuda(at::Tensor x, at::Tensor weight, at::Tensor bias, double eps) {
  auto y = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
      x.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      y.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::vector<at::Tensor> add_layer_norm_f16_cuda(at::Tensor x, at::Tensor residual, at::Tensor weight, at::Tensor bias, double eps) {
  auto x_out = at::empty_like(x);
  auto y = at::empty_like(x);
  const int64_t c64 = x.size(-1);
  TORCH_CHECK(c64 <= INT_MAX, "C too large");
  const int C = static_cast<int>(c64);
  const int64_t rows = x.numel() / C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (C == LN_SMALL_C) {
    if (rows >= 1024) {
      add_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows, static_cast<float>(eps));
    } else if (rows >= 512) {
      add_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL512_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows, static_cast<float>(eps));
    } else {
      add_layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
          x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
          x_out.data_ptr<dtype>(), y.data_ptr<dtype>(), rows, static_cast<float>(eps));
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {x_out, y};
  }
  add_layer_norm_f16_kernel<<<static_cast<int>(rows), LN_THREADS, 0, stream>>>(
      C,
      x.data_ptr<dtype>(),
      residual.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      x_out.data_ptr<dtype>(),
      y.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, y};
}

at::Tensor add_last_layer_norm_f16_cuda(at::Tensor x, at::Tensor residual, at::Tensor weight, at::Tensor bias, double eps) {
  const int64_t B = x.size(0);
  const int64_t T = x.size(1);
  const int64_t C = x.size(2);
  TORCH_CHECK(C == LN_SMALL_C, "add_last_layer_norm_f16 currently requires C=4096");
  auto y = at::empty({B, C}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (B >= 1024) {
    add_last_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, true, true><<<static_cast<int>(B), LN_SMALL512_THREADS, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
  } else if (B >= 512) {
    add_last_layer_norm_f16_small_kernel<LN_SMALL512_THREADS, false, false><<<static_cast<int>(B), LN_SMALL512_THREADS, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
  } else {
    add_last_layer_norm_f16_small_kernel<LN_SMALL_THREADS, false, false><<<static_cast<int>(B), LN_SMALL_THREADS, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), weight.data_ptr<dtype>(), bias.data_ptr<dtype>(),
        y.data_ptr<dtype>(), B, T, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

std::vector<at::Tensor> add_layer_norm_cmix_mix_f16_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor shift_state,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor x_k,
    double eps) {
  auto x_out = at::empty_like(x);
  auto mixed = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  add_layer_norm_cmix_mix_f16_kernel<LN_SMALL_THREADS><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
      x.data_ptr<dtype>(),
      residual.data_ptr<dtype>(),
      shift_state.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      x_k.data_ptr<dtype>(),
      x_out.data_ptr<dtype>(),
      mixed.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, mixed};
}

std::vector<at::Tensor> add_layer_norm_tmix_mix6_f16_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor shift_state,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor x_r,
    at::Tensor x_w,
    at::Tensor x_k,
    at::Tensor x_v,
    at::Tensor x_a,
    at::Tensor x_g,
    double eps) {
  auto x_out = at::empty_like(x);
  auto out_r = at::empty_like(x);
  auto out_w = at::empty_like(x);
  auto out_k = at::empty_like(x);
  auto out_v = at::empty_like(x);
  auto out_a = at::empty_like(x);
  auto out_g = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  add_layer_norm_tmix_mix6_f16_kernel<LN_SMALL_THREADS><<<static_cast<int>(rows), LN_SMALL_THREADS, 0, stream>>>(
      x.data_ptr<dtype>(),
      residual.data_ptr<dtype>(),
      shift_state.data_ptr<dtype>(),
      weight.data_ptr<dtype>(),
      bias.data_ptr<dtype>(),
      x_r.data_ptr<dtype>(),
      x_w.data_ptr<dtype>(),
      x_k.data_ptr<dtype>(),
      x_v.data_ptr<dtype>(),
      x_a.data_ptr<dtype>(),
      x_g.data_ptr<dtype>(),
      x_out.data_ptr<dtype>(),
      out_r.data_ptr<dtype>(),
      out_w.data_ptr<dtype>(),
      out_k.data_ptr<dtype>(),
      out_v.data_ptr<dtype>(),
      out_a.data_ptr<dtype>(),
      out_g.data_ptr<dtype>(),
      rows,
      static_cast<float>(eps));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, out_r, out_w, out_k, out_v, out_a, out_g};
}

std::vector<at::Tensor> add_layer_norm_cmix_mix_f16_cfg_cuda(
    at::Tensor x,
    at::Tensor residual,
    at::Tensor shift_state,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor x_k,
    double eps,
    int threads) {
  auto x_out = at::empty_like(x);
  auto mixed = at::empty_like(x);
  const int64_t rows = x.numel() / LN_SMALL_C;
  auto stream = at::cuda::getCurrentCUDAStream();
  if (threads == 256) {
    add_layer_norm_cmix_mix_f16_kernel<256><<<static_cast<int>(rows), 256, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(), bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(), rows, static_cast<float>(eps));
  } else if (threads == 512) {
    add_layer_norm_cmix_mix_f16_kernel<512><<<static_cast<int>(rows), 512, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(), bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(), rows, static_cast<float>(eps));
  } else {
    add_layer_norm_cmix_mix_f16_kernel<1024><<<static_cast<int>(rows), 1024, 0, stream>>>(
        x.data_ptr<dtype>(), residual.data_ptr<dtype>(), shift_state.data_ptr<dtype>(),
        weight.data_ptr<dtype>(), bias.data_ptr<dtype>(), x_k.data_ptr<dtype>(),
        x_out.data_ptr<dtype>(), mixed.data_ptr<dtype>(), rows, static_cast<float>(eps));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {x_out, mixed};
}

at::Tensor linear_f16_cuda(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16 K/N too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16 M too large");
  const int m = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (m == 0 || n == 0 || k == 0) {
    return y;
  }

  // Row-major y[M,N] = x[M,K] @ weight[K,N] is column-major
  // y^T[N,M] = weight^T[N,K] @ x^T[K,M].
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
  check_cublas(cublasGemmEx(
      handle,
      CUBLAS_OP_N,
      CUBLAS_OP_N,
      n,
      m,
      k,
      &alpha,
      weight.data_ptr<dtype>(),
      CUDA_R_16F,
      n,
      x.data_ptr<dtype>(),
      CUDA_R_16F,
      k,
      &beta,
      y.data_ptr<dtype>(),
      CUDA_R_16F,
      n,
      CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT_TENSOR_OP),
      "linear_f16 cublasGemmEx");
  return y;
}

at::Tensor linear_f16_lt_cuda(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16_lt K/N too large");
  const int k = static_cast<int>(k64);
  const int n = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16_lt M too large");
  const int m = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (m == 0 || n == 0 || k == 0) {
    return y;
  }

  static cublasLtHandle_t lt_handle = nullptr;
  if (lt_handle == nullptr) {
    check_cublaslt(cublasLtCreate(&lt_handle), "cublasLtCreate");
  }

  cublasLtMatmulDesc_t op_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  check_cublaslt(cublasLtMatmulDescCreate(&op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate");
  const cublasOperation_t trans = CUBLAS_OP_N;
  check_cublaslt(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof(trans)), "cublasLt set transa");
  check_cublaslt(cublasLtMatmulDescSetAttribute(op_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans, sizeof(trans)), "cublasLt set transb");
  check_cublaslt(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16F, n, k, n), "cublasLt a layout");
  check_cublaslt(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16F, k, m, k), "cublasLt b layout");
  check_cublaslt(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16F, n, m, n), "cublasLt c layout");
  check_cublaslt(cublasLtMatmulPreferenceCreate(&pref), "cublasLt preference");
  const size_t workspace_size = 0;
  check_cublaslt(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)),
                 "cublasLt set workspace");

  cublasLtMatmulHeuristicResult_t heuristic = {};
  int returned = 0;
  check_cublaslt(cublasLtMatmulAlgoGetHeuristic(lt_handle, op_desc, a_desc, b_desc, c_desc, c_desc, pref, 1, &heuristic, &returned),
                 "cublasLt heuristic");
  TORCH_CHECK(returned > 0, "cublasLt found no algorithm");
  const float alpha = 1.0f;
  const float beta = 0.0f;
  check_cublaslt(cublasLtMatmul(
      lt_handle,
      op_desc,
      &alpha,
      weight.data_ptr<dtype>(),
      a_desc,
      x.data_ptr<dtype>(),
      b_desc,
      &beta,
      y.data_ptr<dtype>(),
      c_desc,
      y.data_ptr<dtype>(),
      c_desc,
      &heuristic.algo,
      nullptr,
      0,
      at::cuda::getCurrentCUDAStream()),
      "cublasLtMatmul");
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(c_desc);
  cublasLtMatrixLayoutDestroy(b_desc);
  cublasLtMatrixLayoutDestroy(a_desc);
  cublasLtMatmulDescDestroy(op_desc);
  return y;
}

template <int ChunkK, int Warps>
at::Tensor linear_f16_m1_splitk_cuda_impl(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16_m1_splitk K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  TORCH_CHECK(x.numel() == k64, "linear_f16_m1_splitk requires M=1");
  TORCH_CHECK((N % 64) == 0, "linear_f16_m1_splitk requires N multiple of 64");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (K == 0 || N == 0) {
    return y;
  }
  const int chunks = static_cast<int>(ceil_div(K, ChunkK));
  auto partial = at::empty({chunks, n64}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_f16_m1_splitk_partial_kernel<ChunkK, Warps><<<dim3(ceil_div(N, Warps * 64), chunks, 1), Warps * 32, 0, stream>>>(
      K, N, x.data_ptr<dtype>(), weight.data_ptr<dtype>(), partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_f16_m1_splitk_reduce_kernel<<<static_cast<int>(ceil_div(N / 2, 128)), 128, 0, stream>>>(
      chunks, N, partial.data_ptr<float>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_f16_m1_splitk_cuda(at::Tensor x, at::Tensor weight) {
  const int64_t K = x.size(-1);
  const int64_t N = weight.size(1);
  if (K == 4096 && N == 4096) {
    return linear_f16_m1_splitk_cuda_impl<128, 2>(x, weight);
  }
  if (N >= 65536) {
    return linear_f16_m1_splitk_cuda_impl<512, 4>(x, weight);
  }
  if (K == 4096 && N == 16384) {
    return linear_f16_m1_splitk_cuda_impl<512, 2>(x, weight);
  }
  if (K >= 8192) {
    return linear_f16_m1_splitk_cuda_impl<512, 2>(x, weight);
  }
  return linear_f16_m1_splitk_cuda_impl<256, 4>(x, weight);
}

template <int ChunkK, int Warps>
at::Tensor linear_mix_f16_m1_splitk_cuda_impl(at::Tensor x, at::Tensor shift, at::Tensor mix, at::Tensor weight) {
  const int64_t k64 = x.numel();
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_mix_f16_m1_splitk K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  TORCH_CHECK(shift.numel() == k64 && mix.numel() == k64, "linear_mix_f16_m1_splitk shape mismatch");
  TORCH_CHECK(weight.size(0) == k64, "linear_mix_f16_m1_splitk weight K mismatch");
  TORCH_CHECK((N % 64) == 0, "linear_mix_f16_m1_splitk requires N multiple of 64");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  const int chunks = static_cast<int>(ceil_div(K, ChunkK));
  auto partial = at::empty({chunks, n64}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_mix_f16_m1_splitk_partial_kernel<ChunkK, Warps><<<dim3(ceil_div(N, Warps * 64), chunks, 1), Warps * 32, 0, stream>>>(
      K, N, x.data_ptr<dtype>(), shift.data_ptr<dtype>(), mix.data_ptr<dtype>(), weight.data_ptr<dtype>(), partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_f16_m1_splitk_reduce_kernel<<<static_cast<int>(ceil_div(N / 2, 128)), 128, 0, stream>>>(
      chunks, N, partial.data_ptr<float>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_mix_f16_m1_splitk_cuda(at::Tensor x, at::Tensor shift, at::Tensor mix, at::Tensor weight) {
  const int64_t K = x.numel();
  const int64_t N = weight.size(1);
  if (K == 4096 && N == 4096) {
    return linear_mix_f16_m1_splitk_cuda_impl<128, 2>(x, shift, mix, weight);
  }
  if (N >= 65536) {
    return linear_mix_f16_m1_splitk_cuda_impl<512, 4>(x, shift, mix, weight);
  }
  if (K == 4096 && N == 16384) {
    return linear_mix_f16_m1_splitk_cuda_impl<512, 2>(x, shift, mix, weight);
  }
  if (K >= 8192) {
    return linear_mix_f16_m1_splitk_cuda_impl<512, 2>(x, shift, mix, weight);
  }
  return linear_mix_f16_m1_splitk_cuda_impl<256, 4>(x, shift, mix, weight);
}

at::Tensor linear_f16_m1_splitk_cfg_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k) {
  switch (chunk_k) {
    case 128:
      return linear_f16_m1_splitk_cuda_impl<128, 4>(x, weight);
    case 256:
      return linear_f16_m1_splitk_cuda_impl<256, 4>(x, weight);
    case 512:
      return linear_f16_m1_splitk_cuda_impl<512, 4>(x, weight);
    case 1024:
      return linear_f16_m1_splitk_cuda_impl<1024, 4>(x, weight);
    default:
      TORCH_CHECK(false, "unsupported chunk_k");
  }
}

at::Tensor linear_f16_m1_splitk_tile_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k, int64_t tile_cols) {
  if (tile_cols == 128) {
    switch (chunk_k) {
      case 128:
        return linear_f16_m1_splitk_cuda_impl<128, 2>(x, weight);
      case 256:
        return linear_f16_m1_splitk_cuda_impl<256, 2>(x, weight);
      case 512:
        return linear_f16_m1_splitk_cuda_impl<512, 2>(x, weight);
      case 1024:
        return linear_f16_m1_splitk_cuda_impl<1024, 2>(x, weight);
      default:
        TORCH_CHECK(false, "unsupported chunk_k");
    }
  }
  TORCH_CHECK(tile_cols == 256, "unsupported tile_cols");
  return linear_f16_m1_splitk_cfg_cuda(x, weight, chunk_k);
}

template <int ChunkK, int Warps>
at::Tensor linear_f16_rows_splitk_cuda_impl(at::Tensor x, at::Tensor weight) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight.size(1);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_f16_rows_splitk K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_f16_rows_splitk M too large");
  const int M = static_cast<int>(m64);
  TORCH_CHECK((N % 64) == 0, "linear_f16_rows_splitk requires N multiple of 64");
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || K == 0 || N == 0) {
    return y;
  }
  const int chunks = static_cast<int>(ceil_div(K, ChunkK));
  auto partial = at::empty({m64, chunks, n64}, x.options().dtype(at::kFloat));
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_f16_rows_splitk_partial_kernel<ChunkK, Warps><<<dim3(ceil_div(N, Warps * 64), chunks, M), Warps * 32, 0, stream>>>(
      K, N, chunks, x.data_ptr<dtype>(), weight.data_ptr<dtype>(), partial.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  linear_f16_rows_splitk_reduce_kernel<<<dim3(static_cast<int>(ceil_div(N / 2, 128)), M, 1), 128, 0, stream>>>(
      chunks, N, partial.data_ptr<float>(), y.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_f16_rows_splitk_cuda(at::Tensor x, at::Tensor weight, int64_t chunk_k) {
  switch (chunk_k) {
    case 128:
      return linear_f16_rows_splitk_cuda_impl<128, 2>(x, weight);
    case 256:
      return linear_f16_rows_splitk_cuda_impl<256, 2>(x, weight);
    case 512:
      return linear_f16_rows_splitk_cuda_impl<512, 2>(x, weight);
    case 1024:
      return linear_f16_rows_splitk_cuda_impl<1024, 2>(x, weight);
    default:
      TORCH_CHECK(false, "unsupported chunk_k");
  }
}

at::Tensor linear_t_f16_cuda(at::Tensor x, at::Tensor weight_t) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_t.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_t_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_t_f16 M too large");
  const int M = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (K <= 512 && N >= 1024 && M <= 4) {
    if (M == 1) {
      linear_t_f16_ntile_scalar_kernel<128, 2><<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
          M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
    } else {
      linear_t_f16_ntile_kernel<128, 4><<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
          M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
    }
  } else if (K >= 1024) {
    linear_t_f16_kernel<256><<<dim3(N, M, 1), 256, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    linear_t_f16_kernel<128><<<dim3(N, M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

template <int Act>
at::Tensor linear_t_act_f16_cuda_impl(at::Tensor x, at::Tensor weight_t) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_t.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_t_act_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_t_act_f16 M too large");
  const int M = static_cast<int>(m64);
  std::vector<int64_t> out_sizes(x.sizes().begin(), x.sizes().end());
  out_sizes.back() = n64;
  auto y = at::empty(out_sizes, x.options());
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(K <= 512 && N >= 1024 && M <= 4, "linear_t_act_f16 currently supports only small-rank rank-out");
  if (M == 1) {
    linear_t_act_f16_ntile_scalar_kernel<128, 2, Act><<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    linear_t_act_f16_ntile_kernel<128, 4, Act><<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

at::Tensor linear_t_act_f16_cuda(at::Tensor x, at::Tensor weight_t, int64_t act) {
  if (act == 1) {
    return linear_t_act_f16_cuda_impl<1>(x, weight_t);
  }
  return linear_t_act_f16_cuda_impl<2>(x, weight_t);
}

std::vector<at::Tensor> linear_wag_rank_in_f16_cuda(
    at::Tensor xw,
    at::Tensor xa,
    at::Tensor xg,
    at::Tensor w1_t,
    at::Tensor a1_t,
    at::Tensor g1_t) {
  const int64_t k64 = xw.size(-1);
  const int64_t rw64 = w1_t.size(0);
  const int64_t ra64 = a1_t.size(0);
  const int64_t rg64 = g1_t.size(0);
  const int64_t m64 = xw.numel() / k64;
  TORCH_CHECK(k64 <= INT_MAX && rw64 <= INT_MAX && ra64 <= INT_MAX && rg64 <= INT_MAX && m64 <= INT_MAX,
              "linear_wag_rank_in_f16 shape too large");
  const int K = static_cast<int>(k64);
  const int Rw = static_cast<int>(rw64);
  const int Ra = static_cast<int>(ra64);
  const int Rg = static_cast<int>(rg64);
  const int Rmax = std::max(Rw, std::max(Ra, Rg));
  const int M = static_cast<int>(m64);
  TORCH_CHECK(K >= 1024 && Rmax <= 512 && M <= 8, "linear_wag_rank_in_f16 supports only K>=1024,R<=512,M<=8");
  std::vector<int64_t> w_sizes(xw.sizes().begin(), xw.sizes().end());
  std::vector<int64_t> a_sizes = w_sizes;
  std::vector<int64_t> g_sizes = w_sizes;
  w_sizes.back() = rw64;
  a_sizes.back() = ra64;
  g_sizes.back() = rg64;
  auto w1 = at::empty(w_sizes, xw.options());
  auto a1 = at::empty(a_sizes, xw.options());
  auto g1 = at::empty(g_sizes, xw.options());
  if (M == 0 || K == 0 || Rmax == 0) {
    return {w1, a1, g1};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_wag_rank_in_f16_kernel<256><<<dim3(Rmax, M, 3), 256, 0, stream>>>(
      M, K, Rw, Ra, Rg, Rmax,
      xw.data_ptr<dtype>(), xa.data_ptr<dtype>(), xg.data_ptr<dtype>(),
      w1_t.data_ptr<dtype>(), a1_t.data_ptr<dtype>(), g1_t.data_ptr<dtype>(),
      w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w1, a1, g1};
}

std::vector<at::Tensor> linear_wag_rank_in_mix_f16_cuda(
    at::Tensor x,
    at::Tensor shift,
    at::Tensor x_w,
    at::Tensor x_a,
    at::Tensor x_g,
    at::Tensor w1_t,
    at::Tensor a1_t,
    at::Tensor g1_t) {
  const int64_t k64 = x.numel();
  const int64_t rw64 = w1_t.size(0);
  const int64_t ra64 = a1_t.size(0);
  const int64_t rg64 = g1_t.size(0);
  TORCH_CHECK(k64 <= INT_MAX && rw64 <= INT_MAX && ra64 <= INT_MAX && rg64 <= INT_MAX,
              "linear_wag_rank_in_mix_f16 shape too large");
  const int K = static_cast<int>(k64);
  const int Rw = static_cast<int>(rw64);
  const int Ra = static_cast<int>(ra64);
  const int Rg = static_cast<int>(rg64);
  const int Rmax = std::max(Rw, std::max(Ra, Rg));
  TORCH_CHECK(shift.numel() == k64 && x_w.numel() == k64 && x_a.numel() == k64 && x_g.numel() == k64,
              "linear_wag_rank_in_mix_f16 input shape mismatch");
  TORCH_CHECK(w1_t.size(1) == k64 && a1_t.size(1) == k64 && g1_t.size(1) == k64,
              "linear_wag_rank_in_mix_f16 weight shape mismatch");
  std::vector<int64_t> w_sizes(x.sizes().begin(), x.sizes().end());
  std::vector<int64_t> a_sizes = w_sizes;
  std::vector<int64_t> g_sizes = w_sizes;
  w_sizes.back() = Rw;
  a_sizes.back() = Ra;
  g_sizes.back() = Rg;
  auto w1 = at::empty(w_sizes, x.options());
  auto a1 = at::empty(a_sizes, x.options());
  auto g1 = at::empty(g_sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_wag_rank_in_mix_f16_kernel<256><<<dim3(Rmax, 1, 3), 256, 0, stream>>>(
      K, Rw, Ra, Rg, Rmax,
      x.data_ptr<dtype>(), shift.data_ptr<dtype>(),
      x_w.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(),
      w1_t.data_ptr<dtype>(), a1_t.data_ptr<dtype>(), g1_t.data_ptr<dtype>(),
      w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w1, a1, g1};
}

std::vector<at::Tensor> linear_wagv_rank_in_f16_cuda(
    at::Tensor xw,
    at::Tensor xa,
    at::Tensor xg,
    at::Tensor xv,
    at::Tensor w1_t,
    at::Tensor a1_t,
    at::Tensor g1_t,
    at::Tensor v1_t) {
  const int64_t k64 = xw.size(-1);
  const int64_t rw64 = w1_t.size(0);
  const int64_t ra64 = a1_t.size(0);
  const int64_t rg64 = g1_t.size(0);
  const int64_t rv64 = v1_t.size(0);
  const int64_t m64 = xw.numel() / k64;
  TORCH_CHECK(k64 <= INT_MAX && rw64 <= INT_MAX && ra64 <= INT_MAX && rg64 <= INT_MAX && rv64 <= INT_MAX && m64 <= INT_MAX,
              "linear_wagv_rank_in_f16 shape too large");
  const int K = static_cast<int>(k64);
  const int Rw = static_cast<int>(rw64);
  const int Ra = static_cast<int>(ra64);
  const int Rg = static_cast<int>(rg64);
  const int Rv = static_cast<int>(rv64);
  const int Rmax = std::max(std::max(Rw, Ra), std::max(Rg, Rv));
  const int M = static_cast<int>(m64);
  TORCH_CHECK(K >= 1024 && Rmax <= 512 && M <= 8, "linear_wagv_rank_in_f16 supports only K>=1024,R<=512,M<=8");
  std::vector<int64_t> w_sizes(xw.sizes().begin(), xw.sizes().end());
  std::vector<int64_t> a_sizes = w_sizes;
  std::vector<int64_t> g_sizes = w_sizes;
  std::vector<int64_t> v_sizes = w_sizes;
  w_sizes.back() = rw64;
  a_sizes.back() = ra64;
  g_sizes.back() = rg64;
  v_sizes.back() = rv64;
  auto w1 = at::empty(w_sizes, xw.options());
  auto a1 = at::empty(a_sizes, xw.options());
  auto g1 = at::empty(g_sizes, xw.options());
  auto v1 = at::empty(v_sizes, xw.options());
  if (M == 0 || K == 0 || Rmax == 0) {
    return {w1, a1, g1, v1};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_wagv_rank_in_f16_kernel<256><<<dim3(Rmax, M, 4), 256, 0, stream>>>(
      M, K, Rw, Ra, Rg, Rv, Rmax,
      xw.data_ptr<dtype>(), xa.data_ptr<dtype>(), xg.data_ptr<dtype>(), xv.data_ptr<dtype>(),
      w1_t.data_ptr<dtype>(), a1_t.data_ptr<dtype>(), g1_t.data_ptr<dtype>(), v1_t.data_ptr<dtype>(),
      w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(), v1.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w1, a1, g1, v1};
}

std::vector<at::Tensor> linear_wagv_rank_in_mix_f16_cuda(
    at::Tensor x,
    at::Tensor shift,
    at::Tensor x_w,
    at::Tensor x_a,
    at::Tensor x_g,
    at::Tensor x_v,
    at::Tensor w1_t,
    at::Tensor a1_t,
    at::Tensor g1_t,
    at::Tensor v1_t) {
  const int64_t k64 = x.numel();
  const int64_t rw64 = w1_t.size(0);
  const int64_t ra64 = a1_t.size(0);
  const int64_t rg64 = g1_t.size(0);
  const int64_t rv64 = v1_t.size(0);
  TORCH_CHECK(k64 <= INT_MAX && rw64 <= INT_MAX && ra64 <= INT_MAX && rg64 <= INT_MAX && rv64 <= INT_MAX,
              "linear_wagv_rank_in_mix_f16 shape too large");
  const int K = static_cast<int>(k64);
  const int Rw = static_cast<int>(rw64);
  const int Ra = static_cast<int>(ra64);
  const int Rg = static_cast<int>(rg64);
  const int Rv = static_cast<int>(rv64);
  const int Rmax = std::max(std::max(Rw, Ra), std::max(Rg, Rv));
  TORCH_CHECK(shift.numel() == k64 && x_w.numel() == k64 && x_a.numel() == k64 && x_g.numel() == k64 && x_v.numel() == k64,
              "linear_wagv_rank_in_mix_f16 input shape mismatch");
  TORCH_CHECK(w1_t.size(1) == k64 && a1_t.size(1) == k64 && g1_t.size(1) == k64 && v1_t.size(1) == k64,
              "linear_wagv_rank_in_mix_f16 weight shape mismatch");
  std::vector<int64_t> w_sizes(x.sizes().begin(), x.sizes().end());
  std::vector<int64_t> a_sizes = w_sizes;
  std::vector<int64_t> g_sizes = w_sizes;
  std::vector<int64_t> v_sizes = w_sizes;
  w_sizes.back() = Rw;
  a_sizes.back() = Ra;
  g_sizes.back() = Rg;
  v_sizes.back() = Rv;
  auto w1 = at::empty(w_sizes, x.options());
  auto a1 = at::empty(a_sizes, x.options());
  auto g1 = at::empty(g_sizes, x.options());
  auto v1 = at::empty(v_sizes, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  linear_wagv_rank_in_mix_f16_kernel<256><<<dim3(Rmax, 1, 4), 256, 0, stream>>>(
      K, Rw, Ra, Rg, Rv, Rmax,
      x.data_ptr<dtype>(), shift.data_ptr<dtype>(),
      x_w.data_ptr<dtype>(), x_a.data_ptr<dtype>(), x_g.data_ptr<dtype>(), x_v.data_ptr<dtype>(),
      w1_t.data_ptr<dtype>(), a1_t.data_ptr<dtype>(), g1_t.data_ptr<dtype>(), v1_t.data_ptr<dtype>(),
      w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(), v1.data_ptr<dtype>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w1, a1, g1, v1};
}

std::vector<at::Tensor> linear_wag_rank_out_f16_cuda(
    at::Tensor w1,
    at::Tensor a1,
    at::Tensor g1,
    at::Tensor w2_t,
    at::Tensor a2_t,
    at::Tensor g2_t) {
  const int64_t kw64 = w1.size(-1);
  const int64_t ka64 = a1.size(-1);
  const int64_t kg64 = g1.size(-1);
  const int64_t c64 = w2_t.size(0);
  const int64_t m64 = w1.numel() / kw64;
  TORCH_CHECK(kw64 <= INT_MAX && ka64 <= INT_MAX && kg64 <= INT_MAX && c64 <= INT_MAX && m64 <= INT_MAX,
              "linear_wag_rank_out_f16 shape too large");
  const int Kw = static_cast<int>(kw64);
  const int Ka = static_cast<int>(ka64);
  const int Kg = static_cast<int>(kg64);
  const int C = static_cast<int>(c64);
  const int M = static_cast<int>(m64);
  TORCH_CHECK(Kw <= 512 && Ka <= 512 && Kg <= 512 && C >= 1024 && M <= 4,
              "linear_wag_rank_out_f16 supports only small-rank M<=4");
  std::vector<int64_t> out_sizes(w1.sizes().begin(), w1.sizes().end());
  out_sizes.back() = c64;
  auto w = at::empty(out_sizes, w1.options());
  auto a = at::empty(out_sizes, w1.options());
  auto g = at::empty(out_sizes, w1.options());
  if (M == 0 || C == 0 || Kw == 0 || Ka == 0 || Kg == 0) {
    return {w, a, g};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (M == 1) {
    linear_wag_rank_out_f16_kernel<128, 4><<<dim3(ceil_div(C, 4), M, 3), 128, 0, stream>>>(
        M, C, Kw, Ka, Kg,
        w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(),
        w2_t.data_ptr<dtype>(), a2_t.data_ptr<dtype>(), g2_t.data_ptr<dtype>(),
        w.data_ptr<dtype>(), a.data_ptr<dtype>(), g.data_ptr<dtype>());
  } else {
    linear_wag_rank_out_f16_kernel<128, 4><<<dim3(ceil_div(C, 4), M, 3), 128, 0, stream>>>(
        M, C, Kw, Ka, Kg,
        w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(),
        w2_t.data_ptr<dtype>(), a2_t.data_ptr<dtype>(), g2_t.data_ptr<dtype>(),
        w.data_ptr<dtype>(), a.data_ptr<dtype>(), g.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w, a, g};
}

std::vector<at::Tensor> linear_wagv_rank_out_f16_cuda(
    at::Tensor w1,
    at::Tensor a1,
    at::Tensor g1,
    at::Tensor v1,
    at::Tensor w2_t,
    at::Tensor a2_t,
    at::Tensor g2_t,
    at::Tensor v2_t,
    at::Tensor v,
    at::Tensor v_first,
    at::Tensor v0) {
  const int64_t kw64 = w1.size(-1);
  const int64_t ka64 = a1.size(-1);
  const int64_t kg64 = g1.size(-1);
  const int64_t kv64 = v1.size(-1);
  const int64_t c64 = w2_t.size(0);
  const int64_t m64 = w1.numel() / kw64;
  TORCH_CHECK(kw64 <= INT_MAX && ka64 <= INT_MAX && kg64 <= INT_MAX && kv64 <= INT_MAX && c64 <= INT_MAX && m64 <= INT_MAX,
              "linear_wagv_rank_out_f16 shape too large");
  const int Kw = static_cast<int>(kw64);
  const int Ka = static_cast<int>(ka64);
  const int Kg = static_cast<int>(kg64);
  const int Kv = static_cast<int>(kv64);
  const int C = static_cast<int>(c64);
  const int M = static_cast<int>(m64);
  TORCH_CHECK(Kw <= 512 && Ka <= 512 && Kg <= 512 && Kv <= 512 && C >= 1024 && M <= 4,
              "linear_wagv_rank_out_f16 supports only small-rank M<=4");
  std::vector<int64_t> out_sizes(w1.sizes().begin(), w1.sizes().end());
  out_sizes.back() = c64;
  auto w = at::empty(out_sizes, w1.options());
  auto a = at::empty(out_sizes, w1.options());
  auto g = at::empty(out_sizes, w1.options());
  auto v_out = at::empty(out_sizes, w1.options());
  if (M == 0 || C == 0 || Kw == 0 || Ka == 0 || Kg == 0 || Kv == 0) {
    return {w, a, g, v_out};
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  if (M == 1) {
    linear_wagv_rank_out_f16_kernel<128, 4><<<dim3(ceil_div(C, 4), M, 4), 128, 0, stream>>>(
        M, C, Kw, Ka, Kg, Kv,
        w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(), v1.data_ptr<dtype>(),
        w2_t.data_ptr<dtype>(), a2_t.data_ptr<dtype>(), g2_t.data_ptr<dtype>(), v2_t.data_ptr<dtype>(),
        v.data_ptr<dtype>(), v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(),
        w.data_ptr<dtype>(), a.data_ptr<dtype>(), g.data_ptr<dtype>(), v_out.data_ptr<dtype>());
  } else {
    linear_wagv_rank_out_f16_kernel<128, 4><<<dim3(ceil_div(C, 4), M, 4), 128, 0, stream>>>(
        M, C, Kw, Ka, Kg, Kv,
        w1.data_ptr<dtype>(), a1.data_ptr<dtype>(), g1.data_ptr<dtype>(), v1.data_ptr<dtype>(),
        w2_t.data_ptr<dtype>(), a2_t.data_ptr<dtype>(), g2_t.data_ptr<dtype>(), v2_t.data_ptr<dtype>(),
        v.data_ptr<dtype>(), v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(),
        w.data_ptr<dtype>(), a.data_ptr<dtype>(), g.data_ptr<dtype>(), v_out.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {w, a, g, v_out};
}

at::Tensor linear_t_vres_f16_cuda(at::Tensor x, at::Tensor weight_t, at::Tensor v, at::Tensor v_first, at::Tensor v0) {
  const int64_t k64 = x.size(-1);
  const int64_t n64 = weight_t.size(0);
  TORCH_CHECK(k64 <= INT_MAX && n64 <= INT_MAX, "linear_t_vres_f16 K/N too large");
  const int K = static_cast<int>(k64);
  const int N = static_cast<int>(n64);
  const int64_t m64 = x.numel() / k64;
  TORCH_CHECK(m64 <= INT_MAX, "linear_t_vres_f16 M too large");
  const int M = static_cast<int>(m64);
  auto y = at::empty_like(v);
  if (M == 0 || N == 0 || K == 0) {
    return y;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  TORCH_CHECK(K <= 512 && N >= 1024 && M <= 4, "linear_t_vres_f16 currently supports only small-rank rank-out");
  if (M == 1) {
    linear_t_vres_f16_ntile_scalar_kernel<128, 2><<<dim3(ceil_div(N, 2), M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), v.data_ptr<dtype>(), v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(), y.data_ptr<dtype>());
  } else {
    linear_t_vres_f16_ntile_kernel<128, 4><<<dim3(ceil_div(N, 4), M, 1), 128, 0, stream>>>(
        M, K, N, x.data_ptr<dtype>(), weight_t.data_ptr<dtype>(), v.data_ptr<dtype>(), v_first.data_ptr<dtype>(), v0.data_ptr<dtype>(), y.data_ptr<dtype>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}
