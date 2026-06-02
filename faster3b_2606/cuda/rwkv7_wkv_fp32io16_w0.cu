#include <assert.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

namespace {

constexpr int N = 64;
constexpr int WARP_THREADS = 32;
constexpr int BLOCK_THREADS = 32;
constexpr float KK_NORMALIZE_EPS = 1.0e-12f;
constexpr float TMIX_LN_X_EPS = 64.0e-5f;
constexpr float W_SCALE_LOG2_E = -0.8750387749145276f;
constexpr float NLOG2_E = -1.4426950408889634f;

#ifdef _IO_FP16_
using io_t = __half;
__device__ __forceinline__ float io_to_float(io_t x) { return __half2float(x); }
__device__ __forceinline__ io_t float_to_io(float x) { return __float2half_rn(x); }
#else
using io_t = float;
__device__ __forceinline__ float io_to_float(float x) { return x; }
__device__ __forceinline__ float float_to_io(float x) { return x; }
#endif

__device__ __forceinline__ float w_eff(float w) {
  return exp2f(W_SCALE_LOG2_E / (1.0f + exp2f(NLOG2_E * w)));
}

__device__ __forceinline__ float load_io(const io_t* ptr, int64_t idx) {
  return io_to_float(__ldg(ptr + idx));
}

__device__ __forceinline__ float load_w_eff(const io_t* w_ptr, const io_t* w0_ptr, int64_t idx, int c) {
  return w_eff(load_io(w_ptr, idx) + load_io(w0_ptr, c));
}

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    x += __shfl_down_sync(0xffffffffu, x, offset);
  }
  return x;
}

__device__ __forceinline__ float warp_sum_broadcast(float x) {
  return __shfl_sync(0xffffffffu, warp_sum(x), 0);
}

__device__ __forceinline__ float block_sum_broadcast(float x) {
  __shared__ float partial[BLOCK_THREADS / WARP_THREADS];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  x = warp_sum(x);
  if (lane == 0) {
    partial[warp] = x;
  }
  __syncthreads();
  x = (threadIdx.x < (BLOCK_THREADS / WARP_THREADS)) ? partial[lane] : 0.0f;
  if (warp == 0) {
    x = warp_sum(x);
  }
  if (threadIdx.x == 0) {
    partial[0] = x;
  }
  __syncthreads();
  return partial[0];
}

__device__ __forceinline__ float sigmoid_fast(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float block64_sum_broadcast(float x) {
  __shared__ float partial[2];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  x = warp_sum(x);
  if (lane == 0) {
    partial[warp] = x;
  }
  __syncthreads();
  x = (threadIdx.x < 2) ? partial[lane] : 0.0f;
  if (warp == 0) {
    x = warp_sum(x);
  }
  if (threadIdx.x == 0) {
    partial[0] = x;
  }
  __syncthreads();
  return partial[0];
}

__global__ __launch_bounds__(N, 2) void wkv_fp32io16_w0_kk_t1_kernel(
    int C,
    int H,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ w0_ptr,
    const io_t* __restrict__ k_raw_ptr,
    const io_t* __restrict__ k_k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a0_ptr,
    const io_t* __restrict__ a12_ptr,
    const io_t* __restrict__ k_a_ptr,
    io_t* __restrict__ y_ptr,
    io_t* __restrict__ new_k_ptr) {
  const int bh = blockIdx.x;
  const int b_id = bh / H;
  const int h = bh - b_id * H;
  const int i = threadIdx.x;
  const int c_base = h * N;
  const int c = c_base + i;
  const int64_t idx = static_cast<int64_t>(b_id) * C + c;
  float* state_base = state_ptr + (static_cast<int64_t>(b_id) * H * N * N + h * N * N + i * N);

  __shared__ float r[N];
  __shared__ float w[N];
  __shared__ float k[N];
  __shared__ float a[N];
  __shared__ float bb[N];

  const float k_raw = load_io(k_raw_ptr, idx);
  const float kk_u = k_raw * load_io(k_k_ptr, c);
  const float sum_sq = block64_sum_broadcast(kk_u * kk_u);
  const float inv_d = 1.0f / fmaxf(sqrtf(sum_sq), KK_NORMALIZE_EPS);
  const float kk = kk_u * inv_d;
  const float gate = sigmoid_fast(load_io(a0_ptr, c) + load_io(a12_ptr, idx));
  const float ka = load_io(k_a_ptr, c);
  const float new_k = k_raw * fmaf(gate, ka, 1.0f - ka);

  r[i] = load_io(r_ptr, idx);
  w[i] = load_w_eff(w_ptr, w0_ptr, idx, c);
  k[i] = new_k;
  a[i] = -kk;
  bb[i] = kk * gate;
  new_k_ptr[idx] = float_to_io(new_k);
  __syncthreads();

  float state[N];
#pragma unroll
  for (int j = 0; j < N; ++j) {
    state[j] = state_base[j];
  }

  float sa = 0.0f;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    sa += state[j] * a[j];
  }

  const float vi = load_io(v_ptr, idx);
  float y = 0.0f;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    float s = state[j];
    s = s * w[j] + sa * bb[j] + k[j] * vi;
    y += s * r[j];
    state[j] = s;
  }
  y_ptr[idx] = float_to_io(y);

#pragma unroll
  for (int j = 0; j < N; ++j) {
    state_base[j] = state[j];
  }
}

__global__ __launch_bounds__(WARP_THREADS, 4) void wkv_fp32io16_w0_kk_t1_row_kernel(
    int C,
    int H,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ w0_ptr,
    const io_t* __restrict__ k_raw_ptr,
    const io_t* __restrict__ k_k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a0_ptr,
    const io_t* __restrict__ a12_ptr,
    const io_t* __restrict__ k_a_ptr,
    io_t* __restrict__ y_ptr,
    io_t* __restrict__ new_k_ptr) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int b_id = blockIdx.z;
  const int lane = threadIdx.x;
  const int c_base = h * N;
  const int token = b_id * C + c_base;
  const int state_base = ((b_id * H + h) * N + row) * N;

  float sum_sq = 0.0f;
  for (int j = lane; j < N; j += WARP_THREADS) {
    const int idx = token + j;
    const float u = load_io(k_raw_ptr, idx) * load_io(k_k_ptr, c_base + j);
    sum_sq += u * u;
  }
  const float inv_d = 1.0f / fmaxf(sqrtf(warp_sum_broadcast(sum_sq)), KK_NORMALIZE_EPS);

  float sa = 0.0f;
  for (int j = lane; j < N; j += WARP_THREADS) {
    const int idx = token + j;
    const float kk = load_io(k_raw_ptr, idx) * load_io(k_k_ptr, c_base + j) * inv_d;
    sa += state_ptr[state_base + j] * (-kk);
  }
  sa = warp_sum_broadcast(sa);

  float yy = 0.0f;
  const float vv = load_io(v_ptr, token + row);
  for (int j = lane; j < N; j += WARP_THREADS) {
    const int idx = token + j;
    const float k_raw = load_io(k_raw_ptr, idx);
    const float kk = k_raw * load_io(k_k_ptr, c_base + j) * inv_d;
    const float gate = sigmoid_fast(load_io(a0_ptr, c_base + j) + load_io(a12_ptr, idx));
    const float ka = load_io(k_a_ptr, c_base + j);
    const float nk = k_raw * fmaf(gate, ka, 1.0f - ka);
    if (j == row) {
      new_k_ptr[idx] = float_to_io(nk);
    }
    const float s = state_ptr[state_base + j] * load_w_eff(w_ptr, w0_ptr, idx, c_base + j) + vv * nk + sa * (kk * gate);
    state_ptr[state_base + j] = s;
    yy += s * load_io(r_ptr, idx);
  }
  yy = warp_sum(yy);
  if (lane == 0) {
    y_ptr[token + row] = float_to_io(yy);
  }
}

__global__ __launch_bounds__(WARP_THREADS, 4) void wkv_fp32io16_w0_kk_pc_t1_kernel(
    int C,
    int H,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ w0_ptr,
    const io_t* __restrict__ k_raw_ptr,
    const io_t* __restrict__ k_k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a0_ptr,
    const io_t* __restrict__ a12_ptr,
    const io_t* __restrict__ k_a_ptr,
    io_t* __restrict__ y_ptr,
    io_t* __restrict__ new_k_ptr,
    io_t* __restrict__ neg_kk_ptr,
    io_t* __restrict__ kka_ptr,
    int* __restrict__ ready_ptr) {
  const int role = blockIdx.x;
  const int h = blockIdx.y;
  const int b_id = blockIdx.z;
  const int lane = threadIdx.x;
  const int c_base = h * N;
  const int token = b_id * C + c_base;
  const int bh = b_id * H + h;

  if (role == 0) {
    float sum_sq = 0.0f;
    for (int j = lane; j < N; j += WARP_THREADS) {
      const float u = load_io(k_raw_ptr, token + j) * load_io(k_k_ptr, c_base + j);
      sum_sq += u * u;
    }
    const float inv_d = 1.0f / fmaxf(sqrtf(warp_sum_broadcast(sum_sq)), KK_NORMALIZE_EPS);
    for (int j = lane; j < N; j += WARP_THREADS) {
      const int idx = token + j;
      const float k_raw = load_io(k_raw_ptr, idx);
      const float kk = k_raw * load_io(k_k_ptr, c_base + j) * inv_d;
      const float gate = sigmoid_fast(load_io(a0_ptr, c_base + j) + load_io(a12_ptr, idx));
      const float ka = load_io(k_a_ptr, c_base + j);
      new_k_ptr[idx] = float_to_io(k_raw * fmaf(gate, ka, 1.0f - ka));
      neg_kk_ptr[idx] = float_to_io(-kk);
      kka_ptr[idx] = float_to_io(kk * gate);
    }
    __threadfence();
    if (lane == 0) {
      atomicExch(ready_ptr + bh, 1);
    }
    return;
  }

  const volatile int* ready_v = ready_ptr;
  while (ready_v[bh] == 0) {
    __nanosleep(64);
  }

  const int row = role - 1;
  const int state_base = ((b_id * H + h) * N + row) * N;
  float sa = 0.0f;
  for (int j = lane; j < N; j += WARP_THREADS) {
    sa += state_ptr[state_base + j] * load_io(neg_kk_ptr, token + j);
  }
  sa = warp_sum_broadcast(sa);

  float yy = 0.0f;
  const float vv = load_io(v_ptr, token + row);
  for (int j = lane; j < N; j += WARP_THREADS) {
    const int idx = token + j;
    const float s = state_ptr[state_base + j] * load_w_eff(w_ptr, w0_ptr, idx, c_base + j) +
                    vv * load_io(new_k_ptr, idx) + sa * load_io(kka_ptr, idx);
    state_ptr[state_base + j] = s;
    yy += s * load_io(r_ptr, idx);
  }
  yy = warp_sum(yy);
  if (lane == 0) {
    y_ptr[token + row] = float_to_io(yy);
  }
}

template <int ROW_TILE>
__global__ __launch_bounds__(WARP_THREADS, 4) void wkv_fp32io16_w0_kk_t1_tile_kernel(
    int C,
    int H,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ w0_ptr,
    const io_t* __restrict__ k_raw_ptr,
    const io_t* __restrict__ k_k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a0_ptr,
    const io_t* __restrict__ a12_ptr,
    const io_t* __restrict__ k_a_ptr,
    io_t* __restrict__ y_ptr,
    io_t* __restrict__ new_k_ptr) {
  const int row0 = blockIdx.x * ROW_TILE;
  const int h = blockIdx.y;
  const int b_id = blockIdx.z;
  const int lane = threadIdx.x;
  const int c_base = h * N;
  const int token = b_id * C + c_base;

  float sum_sq = 0.0f;
  for (int j = lane; j < N; j += WARP_THREADS) {
    const float u = load_io(k_raw_ptr, token + j) * load_io(k_k_ptr, c_base + j);
    sum_sq += u * u;
  }
  const float inv_d = 1.0f / fmaxf(sqrtf(warp_sum_broadcast(sum_sq)), KK_NORMALIZE_EPS);

  float sa[ROW_TILE];
  float yy[ROW_TILE];
#pragma unroll
  for (int t = 0; t < ROW_TILE; ++t) {
    sa[t] = 0.0f;
    yy[t] = 0.0f;
  }

  for (int j = lane; j < N; j += WARP_THREADS) {
    const int idx = token + j;
    const float kk = load_io(k_raw_ptr, idx) * load_io(k_k_ptr, c_base + j) * inv_d;
#pragma unroll
    for (int t = 0; t < ROW_TILE; ++t) {
      const int row = row0 + t;
      if (row < N) {
        const int state_base = ((b_id * H + h) * N + row) * N;
        sa[t] += state_ptr[state_base + j] * (-kk);
      }
    }
  }
#pragma unroll
  for (int t = 0; t < ROW_TILE; ++t) {
    sa[t] = warp_sum_broadcast(sa[t]);
  }

  for (int j = lane; j < N; j += WARP_THREADS) {
    const int idx = token + j;
    const float k_raw = load_io(k_raw_ptr, idx);
    const float kk = k_raw * load_io(k_k_ptr, c_base + j) * inv_d;
    const float gate = sigmoid_fast(load_io(a0_ptr, c_base + j) + load_io(a12_ptr, idx));
    const float ka = load_io(k_a_ptr, c_base + j);
    const float nk = k_raw * fmaf(gate, ka, 1.0f - ka);
    const float kka = kk * gate;
    if (row0 == 0) {
      new_k_ptr[idx] = float_to_io(nk);
    }
#pragma unroll
    for (int t = 0; t < ROW_TILE; ++t) {
      const int row = row0 + t;
      if (row < N) {
        const int state_base = ((b_id * H + h) * N + row) * N;
        const float s = state_ptr[state_base + j] * load_w_eff(w_ptr, w0_ptr, idx, c_base + j) +
                        load_io(v_ptr, token + row) * nk + sa[t] * kka;
        state_ptr[state_base + j] = s;
        yy[t] += s * load_io(r_ptr, idx);
      }
    }
  }
#pragma unroll
  for (int t = 0; t < ROW_TILE; ++t) {
    const int row = row0 + t;
    if (row < N) {
      const float sum = warp_sum(yy[t]);
      if (lane == 0) {
        y_ptr[token + row] = float_to_io(sum);
      }
    }
  }
}

__global__ __launch_bounds__(N, 2) void wkv_fp32io16_w0_lnx_t1_kernel(
    int C,
    int H,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ w0_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ neg_kk_ptr,
    const io_t* __restrict__ kka_ptr,
    const io_t* __restrict__ r_k_ptr,
    const io_t* __restrict__ weight_ptr,
    const io_t* __restrict__ bias_ptr,
    const io_t* __restrict__ g_ptr,
    io_t* __restrict__ out_ptr) {
  const int bh = blockIdx.x;
  const int b_id = bh / H;
  const int h = bh - b_id * H;
  const int i = threadIdx.x;
  const int c_base = h * N;
  const int c = c_base + i;
  const int64_t idx = static_cast<int64_t>(b_id) * C + c;
  float* state_base = state_ptr + (static_cast<int64_t>(b_id) * H * N * N + h * N * N + i * N);

  __shared__ float r[N];
  __shared__ float w[N];
  __shared__ float k[N];
  __shared__ float a[N];
  __shared__ float bb[N];
  __shared__ float partial[2];
  r[i] = load_io(r_ptr, idx);
  w[i] = load_w_eff(w_ptr, w0_ptr, idx, c);
  k[i] = load_io(k_ptr, idx);
  a[i] = load_io(neg_kk_ptr, idx);
  bb[i] = load_io(kka_ptr, idx);
  __syncthreads();

  float state[N];
#pragma unroll
  for (int j = 0; j < N; ++j) {
    state[j] = state_base[j];
  }

  float sa = 0.0f;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    sa += state[j] * a[j];
  }

  const float vi = load_io(v_ptr, idx);
  float y = 0.0f;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    float s = state[j];
    s = s * w[j] + sa * bb[j] + k[j] * vi;
    y += s * r[j];
    state[j] = s;
  }
#pragma unroll
  for (int j = 0; j < N; ++j) {
    state_base[j] = state[j];
  }
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float sum = warp_sum(y);
  if (lane == 0) {
    partial[warp] = sum;
  }
  __syncthreads();
  const float mean = (partial[0] + partial[1]) * (1.0f / static_cast<float>(N));
  __syncthreads();

  const float d = y - mean;
  sum = warp_sum(d * d);
  if (lane == 0) {
    partial[warp] = sum;
  }
  __syncthreads();
  const float rstd = rsqrtf((partial[0] + partial[1]) * (1.0f / static_cast<float>(N)) + TMIX_LN_X_EPS);
  __syncthreads();

  sum = warp_sum(load_io(r_ptr, idx) * load_io(k_ptr, idx) * load_io(r_k_ptr, c));
  if (lane == 0) {
    partial[warp] = sum;
  }
  __syncthreads();
  const float rkv = partial[0] + partial[1];
  __syncthreads();

  const float out = (d * rstd * load_io(weight_ptr, c) + load_io(bias_ptr, c) + rkv * load_io(v_ptr, idx)) * load_io(g_ptr, idx);
  out_ptr[idx] = float_to_io(out);
}

__global__ __launch_bounds__(N, 2) void wkv_fp32io16_w0_kk_lnx_t1_kernel(
    int C,
    int H,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ w0_ptr,
    const io_t* __restrict__ k_raw_ptr,
    const io_t* __restrict__ k_k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a0_ptr,
    const io_t* __restrict__ a12_ptr,
    const io_t* __restrict__ k_a_ptr,
    const io_t* __restrict__ r_k_ptr,
    const io_t* __restrict__ weight_ptr,
    const io_t* __restrict__ bias_ptr,
    const io_t* __restrict__ g_ptr,
    io_t* __restrict__ out_ptr) {
  const int bh = blockIdx.x;
  const int b_id = bh / H;
  const int h = bh - b_id * H;
  const int i = threadIdx.x;
  const int lane = i & 31;
  const int warp = i >> 5;
  const int c_base = h * N;
  const int c = c_base + i;
  const int64_t idx = static_cast<int64_t>(b_id) * C + c;
  float* state_base = state_ptr + (static_cast<int64_t>(b_id) * H * N * N + h * N * N + i * N);

  __shared__ float r[N];
  __shared__ float w[N];
  __shared__ float k[N];
  __shared__ float a[N];
  __shared__ float bb[N];
  __shared__ float partial[2];

  const float k_raw = load_io(k_raw_ptr, idx);
  const float kk_u = k_raw * load_io(k_k_ptr, c);
  float sum = warp_sum(kk_u * kk_u);
  if (lane == 0) {
    partial[warp] = sum;
  }
  __syncthreads();
  const float inv_d = 1.0f / fmaxf(sqrtf(partial[0] + partial[1]), KK_NORMALIZE_EPS);
  __syncthreads();

  const float kk = kk_u * inv_d;
  const float gate = sigmoid_fast(load_io(a0_ptr, c) + load_io(a12_ptr, idx));
  const float ka = load_io(k_a_ptr, c);
  const float new_k = k_raw * fmaf(gate, ka, 1.0f - ka);
  r[i] = load_io(r_ptr, idx);
  w[i] = load_w_eff(w_ptr, w0_ptr, idx, c);
  k[i] = new_k;
  a[i] = -kk;
  bb[i] = kk * gate;
  __syncthreads();

  float state[N];
#pragma unroll
  for (int j = 0; j < N; ++j) {
    state[j] = state_base[j];
  }

  float sa = 0.0f;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    sa += state[j] * a[j];
  }

  const float vi = load_io(v_ptr, idx);
  float y = 0.0f;
#pragma unroll
  for (int j = 0; j < N; ++j) {
    float s = state[j];
    s = s * w[j] + sa * bb[j] + k[j] * vi;
    y += s * r[j];
    state[j] = s;
  }
#pragma unroll
  for (int j = 0; j < N; ++j) {
    state_base[j] = state[j];
  }
  __syncthreads();

  sum = warp_sum(y);
  if (lane == 0) {
    partial[warp] = sum;
  }
  __syncthreads();
  const float mean = (partial[0] + partial[1]) * (1.0f / static_cast<float>(N));
  __syncthreads();

  const float d = y - mean;
  sum = warp_sum(d * d);
  if (lane == 0) {
    partial[warp] = sum;
  }
  __syncthreads();
  const float rstd = rsqrtf((partial[0] + partial[1]) * (1.0f / static_cast<float>(N)) + TMIX_LN_X_EPS);
  __syncthreads();

  sum = warp_sum(r[i] * new_k * load_io(r_k_ptr, c));
  if (lane == 0) {
    partial[warp] = sum;
  }
  __syncthreads();
  const float rkv = partial[0] + partial[1];
  __syncthreads();

  const float out = (d * rstd * load_io(weight_ptr, c) + load_io(bias_ptr, c) + rkv * vi) * load_io(g_ptr, idx);
  out_ptr[idx] = float_to_io(out);
}

template <int HeadSize>
__launch_bounds__(HeadSize, 2)
__global__ void wkv_fp32_v2_kernel(
    int T,
    int C,
    int H,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ w0_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ y_ptr) {
  const int bh = blockIdx.x;
  const int b_id = bh / H;
  const int h = bh - b_id * H;
  const int i = threadIdx.x;
  const int c_base = h * HeadSize;
  const int64_t bt_base = static_cast<int64_t>(b_id) * T * C + c_base;
  float* state_base = state_ptr + (static_cast<int64_t>(b_id) * H * HeadSize * HeadSize + h * HeadSize * HeadSize + i * HeadSize);

  float state[HeadSize];
#pragma unroll
  for (int j = 0; j < HeadSize; ++j) {
    state[j] = state_base[j];
  }

  __shared__ float r[HeadSize];
  __shared__ float w[HeadSize];
  __shared__ float k[HeadSize];
  __shared__ float a[HeadSize];
  __shared__ float b[HeadSize];

  for (int t = 0; t < T; ++t) {
    const int64_t idx = bt_base + static_cast<int64_t>(t) * C + i;
    __syncthreads();
    r[i] = load_io(r_ptr, idx);
    w[i] = load_w_eff(w_ptr, w0_ptr, idx, c_base + i);
    k[i] = load_io(k_ptr, idx);
    a[i] = load_io(a_ptr, idx);
    b[i] = load_io(b_ptr, idx);
    __syncthreads();

    float sa = 0.0f;
#pragma unroll
    for (int j = 0; j < HeadSize; ++j) {
      sa += state[j] * a[j];
    }

    const float vi = load_io(v_ptr, idx);
    float y = 0.0f;
#pragma unroll
    for (int j = 0; j < HeadSize; ++j) {
      float s = state[j];
      s = s * w[j] + sa * b[j] + k[j] * vi;
      y += s * r[j];
      state[j] = s;
    }
    y_ptr[idx] = float_to_io(y);
  }

#pragma unroll
  for (int j = 0; j < HeadSize; ++j) {
    state_base[j] = state[j];
  }
}

__global__ __launch_bounds__(WARP_THREADS, 4) void wkv_fp32_v2_small_warp_kernel(
    int T,
    int C,
    int H,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ w0_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ y_ptr) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int b_id = blockIdx.z;
  const int lane = threadIdx.x;
  const int c_base = h * N;
  const int state_base = ((b_id * H + h) * N + row) * N;

  for (int t = 0; t < T; ++t) {
    const int token = (b_id * T + t) * C + c_base;
    float sa = 0.0f;
    for (int j = lane; j < N; j += WARP_THREADS) {
      sa += state_ptr[state_base + j] * load_io(a_ptr, token + j);
    }
    sa = warp_sum_broadcast(sa);

    float yy = 0.0f;
    const float vv = load_io(v_ptr, token + row);
    for (int j = lane; j < N; j += WARP_THREADS) {
      const int idx = token + j;
      const float s = state_ptr[state_base + j] * load_w_eff(w_ptr, w0_ptr, idx, c_base + j) + vv * load_io(k_ptr, idx) + sa * load_io(b_ptr, idx);
      state_ptr[state_base + j] = s;
      yy += s * load_io(r_ptr, idx);
    }
    yy = warp_sum(yy);
    if (lane == 0) {
      y_ptr[token + row] = float_to_io(yy);
    }
  }
}

__global__ __launch_bounds__(BLOCK_THREADS, 4) void wkv_fp32_v2_short_block_kernel(
    int T,
    int C,
    int H,
    float* __restrict__ state_ptr,
    const io_t* __restrict__ r_ptr,
    const io_t* __restrict__ w_ptr,
    const io_t* __restrict__ w0_ptr,
    const io_t* __restrict__ k_ptr,
    const io_t* __restrict__ v_ptr,
    const io_t* __restrict__ a_ptr,
    const io_t* __restrict__ b_ptr,
    io_t* __restrict__ y_ptr) {
  const int row = blockIdx.x;
  const int h = blockIdx.y;
  const int b_id = blockIdx.z;
  const int tid = threadIdx.x;
  const int c_base = h * N;
  const int state_base = ((b_id * H + h) * N + row) * N;

  for (int t = 0; t < T; ++t) {
    const int token = (b_id * T + t) * C + c_base;
    float sa = 0.0f;
    for (int j = tid; j < N; j += BLOCK_THREADS) {
      sa += state_ptr[state_base + j] * load_io(a_ptr, token + j);
    }
    sa = block_sum_broadcast(sa);

    float yy = 0.0f;
    const float vv = load_io(v_ptr, token + row);
    for (int j = tid; j < N; j += BLOCK_THREADS) {
      const int idx = token + j;
      const float s = state_ptr[state_base + j] * load_w_eff(w_ptr, w0_ptr, idx, c_base + j) + vv * load_io(k_ptr, idx) + sa * load_io(b_ptr, idx);
      state_ptr[state_base + j] = s;
      yy += s * load_io(r_ptr, idx);
    }
    yy = block_sum_broadcast(yy);
    if (tid == 0) {
      y_ptr[token + row] = float_to_io(yy);
    }
    __syncthreads();
  }
}

bool use_small_auto(int B, int T) {
#ifdef _IO_FP16_
  return (T == 1 && B <= 96) ||
         (T == 2 && B <= 21) ||
         (T == 3 && B <= 3) ||
         (T == 4 && (B == 1 || B == 3)) ||
         (B == 1 && T >= 5 && T <= 11);
#else
  return (T == 1) ||
         (T == 2 && B <= 96) ||
         (T == 3 && (B <= 4 || B == 6)) ||
         (T == 4 && (B == 1 || B == 3)) ||
         (B == 1 && T >= 5 && T <= 9);
#endif
}

}  // namespace

void wkv_fp32io16_w0_cuda(
    int B,
    int T,
    int C,
    int H,
    int mode,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor a,
    at::Tensor b,
    at::Tensor y) {
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  const bool use_small = (mode == 2) || (mode == 0 && use_small_auto(B, T));
  if (mode == 3) {
    wkv_fp32_v2_short_block_kernel<<<dim3(N, H, B), dim3(BLOCK_THREADS), 0, stream>>>(
        T,
        C,
        H,
        state.data_ptr<float>(),
        reinterpret_cast<io_t*>(r.data_ptr()),
        reinterpret_cast<io_t*>(w.data_ptr()),
        reinterpret_cast<io_t*>(w0.data_ptr()),
        reinterpret_cast<io_t*>(k.data_ptr()),
        reinterpret_cast<io_t*>(v.data_ptr()),
        reinterpret_cast<io_t*>(a.data_ptr()),
        reinterpret_cast<io_t*>(b.data_ptr()),
        reinterpret_cast<io_t*>(y.data_ptr()));
  } else if (use_small) {
    wkv_fp32_v2_small_warp_kernel<<<dim3(N, H, B), dim3(WARP_THREADS), 0, stream>>>(
        T,
        C,
        H,
        state.data_ptr<float>(),
        reinterpret_cast<io_t*>(r.data_ptr()),
        reinterpret_cast<io_t*>(w.data_ptr()),
        reinterpret_cast<io_t*>(w0.data_ptr()),
        reinterpret_cast<io_t*>(k.data_ptr()),
        reinterpret_cast<io_t*>(v.data_ptr()),
        reinterpret_cast<io_t*>(a.data_ptr()),
        reinterpret_cast<io_t*>(b.data_ptr()),
        reinterpret_cast<io_t*>(y.data_ptr()));
  } else {
    wkv_fp32_v2_kernel<N><<<dim3(B * H), dim3(N), 0, stream>>>(
        T,
        C,
        H,
        state.data_ptr<float>(),
        reinterpret_cast<io_t*>(r.data_ptr()),
        reinterpret_cast<io_t*>(w.data_ptr()),
        reinterpret_cast<io_t*>(w0.data_ptr()),
        reinterpret_cast<io_t*>(k.data_ptr()),
        reinterpret_cast<io_t*>(v.data_ptr()),
        reinterpret_cast<io_t*>(a.data_ptr()),
        reinterpret_cast<io_t*>(b.data_ptr()),
        reinterpret_cast<io_t*>(y.data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor wkv_fp32io16_w0_kk_t1_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k_raw,
    at::Tensor k_k,
    at::Tensor v,
    at::Tensor a0,
    at::Tensor a12,
    at::Tensor k_a,
    at::Tensor y) {
  assert(T == 1);
  assert(C == H * N);
  auto new_k = at::empty_like(k_raw);
  auto stream = at::cuda::getCurrentCUDAStream();
  wkv_fp32io16_w0_kk_t1_kernel<<<dim3(B * H), dim3(N), 0, stream>>>(
      C,
      H,
      state.data_ptr<float>(),
      reinterpret_cast<const io_t*>(r.data_ptr()),
      reinterpret_cast<const io_t*>(w.data_ptr()),
      reinterpret_cast<const io_t*>(w0.data_ptr()),
      reinterpret_cast<const io_t*>(k_raw.data_ptr()),
      reinterpret_cast<const io_t*>(k_k.data_ptr()),
      reinterpret_cast<const io_t*>(v.data_ptr()),
      reinterpret_cast<const io_t*>(a0.data_ptr()),
      reinterpret_cast<const io_t*>(a12.data_ptr()),
      reinterpret_cast<const io_t*>(k_a.data_ptr()),
      reinterpret_cast<io_t*>(y.data_ptr()),
      reinterpret_cast<io_t*>(new_k.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return new_k;
}

at::Tensor wkv_fp32io16_w0_kk_t1_row_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k_raw,
    at::Tensor k_k,
    at::Tensor v,
    at::Tensor a0,
    at::Tensor a12,
    at::Tensor k_a,
    at::Tensor y) {
  assert(T == 1);
  assert(C == H * N);
  auto new_k = at::empty_like(k_raw);
  auto stream = at::cuda::getCurrentCUDAStream();
  wkv_fp32io16_w0_kk_t1_row_kernel<<<dim3(N, H, B), dim3(WARP_THREADS), 0, stream>>>(
      C,
      H,
      state.data_ptr<float>(),
      reinterpret_cast<const io_t*>(r.data_ptr()),
      reinterpret_cast<const io_t*>(w.data_ptr()),
      reinterpret_cast<const io_t*>(w0.data_ptr()),
      reinterpret_cast<const io_t*>(k_raw.data_ptr()),
      reinterpret_cast<const io_t*>(k_k.data_ptr()),
      reinterpret_cast<const io_t*>(v.data_ptr()),
      reinterpret_cast<const io_t*>(a0.data_ptr()),
      reinterpret_cast<const io_t*>(a12.data_ptr()),
      reinterpret_cast<const io_t*>(k_a.data_ptr()),
      reinterpret_cast<io_t*>(y.data_ptr()),
      reinterpret_cast<io_t*>(new_k.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return new_k;
}

void wkv_fp32io16_w0_kk_pc_t1_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k_raw,
    at::Tensor k_k,
    at::Tensor v,
    at::Tensor a0,
    at::Tensor a12,
    at::Tensor k_a,
    at::Tensor y,
    at::Tensor new_k,
    at::Tensor neg_kk,
    at::Tensor kka,
    at::Tensor ready) {
  assert(T == 1);
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  wkv_fp32io16_w0_kk_pc_t1_kernel<<<dim3(N + 1, H, B), dim3(WARP_THREADS), 0, stream>>>(
      C,
      H,
      state.data_ptr<float>(),
      reinterpret_cast<const io_t*>(r.data_ptr()),
      reinterpret_cast<const io_t*>(w.data_ptr()),
      reinterpret_cast<const io_t*>(w0.data_ptr()),
      reinterpret_cast<const io_t*>(k_raw.data_ptr()),
      reinterpret_cast<const io_t*>(k_k.data_ptr()),
      reinterpret_cast<const io_t*>(v.data_ptr()),
      reinterpret_cast<const io_t*>(a0.data_ptr()),
      reinterpret_cast<const io_t*>(a12.data_ptr()),
      reinterpret_cast<const io_t*>(k_a.data_ptr()),
      reinterpret_cast<io_t*>(y.data_ptr()),
      reinterpret_cast<io_t*>(new_k.data_ptr()),
      reinterpret_cast<io_t*>(neg_kk.data_ptr()),
      reinterpret_cast<io_t*>(kka.data_ptr()),
      ready.data_ptr<int>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor wkv_fp32io16_w0_kk_t1_tile_cuda(
    int B,
    int T,
    int C,
    int H,
    int row_tile,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k_raw,
    at::Tensor k_k,
    at::Tensor v,
    at::Tensor a0,
    at::Tensor a12,
    at::Tensor k_a,
    at::Tensor y) {
  assert(T == 1);
  assert(C == H * N);
  auto new_k = at::empty_like(k_raw);
  auto stream = at::cuda::getCurrentCUDAStream();
  if (row_tile == 8) {
    wkv_fp32io16_w0_kk_t1_tile_kernel<8><<<dim3((N + 7) / 8, H, B), dim3(WARP_THREADS), 0, stream>>>(
        C,
        H,
        state.data_ptr<float>(),
        reinterpret_cast<const io_t*>(r.data_ptr()),
        reinterpret_cast<const io_t*>(w.data_ptr()),
        reinterpret_cast<const io_t*>(w0.data_ptr()),
        reinterpret_cast<const io_t*>(k_raw.data_ptr()),
        reinterpret_cast<const io_t*>(k_k.data_ptr()),
        reinterpret_cast<const io_t*>(v.data_ptr()),
        reinterpret_cast<const io_t*>(a0.data_ptr()),
        reinterpret_cast<const io_t*>(a12.data_ptr()),
        reinterpret_cast<const io_t*>(k_a.data_ptr()),
        reinterpret_cast<io_t*>(y.data_ptr()),
        reinterpret_cast<io_t*>(new_k.data_ptr()));
  } else {
    wkv_fp32io16_w0_kk_t1_tile_kernel<4><<<dim3((N + 3) / 4, H, B), dim3(WARP_THREADS), 0, stream>>>(
        C,
        H,
        state.data_ptr<float>(),
        reinterpret_cast<const io_t*>(r.data_ptr()),
        reinterpret_cast<const io_t*>(w.data_ptr()),
        reinterpret_cast<const io_t*>(w0.data_ptr()),
        reinterpret_cast<const io_t*>(k_raw.data_ptr()),
        reinterpret_cast<const io_t*>(k_k.data_ptr()),
        reinterpret_cast<const io_t*>(v.data_ptr()),
        reinterpret_cast<const io_t*>(a0.data_ptr()),
        reinterpret_cast<const io_t*>(a12.data_ptr()),
        reinterpret_cast<const io_t*>(k_a.data_ptr()),
        reinterpret_cast<io_t*>(y.data_ptr()),
        reinterpret_cast<io_t*>(new_k.data_ptr()));
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return new_k;
}

at::Tensor wkv_fp32io16_w0_lnx_t1_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k,
    at::Tensor v,
    at::Tensor neg_kk,
    at::Tensor kka,
    at::Tensor r_k,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor g) {
  assert(T == 1);
  assert(C == H * N);
  auto out = at::empty_like(r);
  auto stream = at::cuda::getCurrentCUDAStream();
  wkv_fp32io16_w0_lnx_t1_kernel<<<dim3(B * H), dim3(N), 0, stream>>>(
      C,
      H,
      state.data_ptr<float>(),
      reinterpret_cast<const io_t*>(r.data_ptr()),
      reinterpret_cast<const io_t*>(w.data_ptr()),
      reinterpret_cast<const io_t*>(w0.data_ptr()),
      reinterpret_cast<const io_t*>(k.data_ptr()),
      reinterpret_cast<const io_t*>(v.data_ptr()),
      reinterpret_cast<const io_t*>(neg_kk.data_ptr()),
      reinterpret_cast<const io_t*>(kka.data_ptr()),
      reinterpret_cast<const io_t*>(r_k.data_ptr()),
      reinterpret_cast<const io_t*>(weight.data_ptr()),
      reinterpret_cast<const io_t*>(bias.data_ptr()),
      reinterpret_cast<const io_t*>(g.data_ptr()),
      reinterpret_cast<io_t*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

at::Tensor wkv_fp32io16_w0_kk_lnx_t1_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k_raw,
    at::Tensor k_k,
    at::Tensor v,
    at::Tensor a0,
    at::Tensor a12,
    at::Tensor k_a,
    at::Tensor r_k,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor g) {
  assert(T == 1);
  assert(C == H * N);
  auto out = at::empty_like(r);
  auto stream = at::cuda::getCurrentCUDAStream();
  wkv_fp32io16_w0_kk_lnx_t1_kernel<<<dim3(B * H), dim3(N), 0, stream>>>(
      C,
      H,
      state.data_ptr<float>(),
      reinterpret_cast<const io_t*>(r.data_ptr()),
      reinterpret_cast<const io_t*>(w.data_ptr()),
      reinterpret_cast<const io_t*>(w0.data_ptr()),
      reinterpret_cast<const io_t*>(k_raw.data_ptr()),
      reinterpret_cast<const io_t*>(k_k.data_ptr()),
      reinterpret_cast<const io_t*>(v.data_ptr()),
      reinterpret_cast<const io_t*>(a0.data_ptr()),
      reinterpret_cast<const io_t*>(a12.data_ptr()),
      reinterpret_cast<const io_t*>(k_a.data_ptr()),
      reinterpret_cast<const io_t*>(r_k.data_ptr()),
      reinterpret_cast<const io_t*>(weight.data_ptr()),
      reinterpret_cast<const io_t*>(bias.data_ptr()),
      reinterpret_cast<const io_t*>(g.data_ptr()),
      reinterpret_cast<io_t*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

void wkv_fp32io16_w0_kk_lnx_t1_into_cuda(
    int B,
    int T,
    int C,
    int H,
    at::Tensor state,
    at::Tensor r,
    at::Tensor w,
    at::Tensor w0,
    at::Tensor k_raw,
    at::Tensor k_k,
    at::Tensor v,
    at::Tensor a0,
    at::Tensor a12,
    at::Tensor k_a,
    at::Tensor r_k,
    at::Tensor weight,
    at::Tensor bias,
    at::Tensor g,
    at::Tensor out) {
  assert(T == 1);
  assert(C == H * N);
  auto stream = at::cuda::getCurrentCUDAStream();
  wkv_fp32io16_w0_kk_lnx_t1_kernel<<<dim3(B * H), dim3(N), 0, stream>>>(
      C,
      H,
      state.data_ptr<float>(),
      reinterpret_cast<const io_t*>(r.data_ptr()),
      reinterpret_cast<const io_t*>(w.data_ptr()),
      reinterpret_cast<const io_t*>(w0.data_ptr()),
      reinterpret_cast<const io_t*>(k_raw.data_ptr()),
      reinterpret_cast<const io_t*>(k_k.data_ptr()),
      reinterpret_cast<const io_t*>(v.data_ptr()),
      reinterpret_cast<const io_t*>(a0.data_ptr()),
      reinterpret_cast<const io_t*>(a12.data_ptr()),
      reinterpret_cast<const io_t*>(k_a.data_ptr()),
      reinterpret_cast<const io_t*>(r_k.data_ptr()),
      reinterpret_cast<const io_t*>(weight.data_ptr()),
      reinterpret_cast<const io_t*>(bias.data_ptr()),
      reinterpret_cast<const io_t*>(g.data_ptr()),
      reinterpret_cast<io_t*>(out.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
