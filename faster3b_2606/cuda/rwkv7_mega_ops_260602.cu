// 2026-06-02 B1T1 selected-kernel CUDA implementation.


#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <cuda_fp16.h>

#include "rwkv7_mega_config.cuh"

namespace {

constexpr int HEAD_SIZE = 64;
constexpr float KK_NORMALIZE_EPS = 1.0e-12f;
constexpr float TMIX_LN_X_EPS = 64.0e-5f;
constexpr int LN_SMALL_C = 4096;
constexpr int LN_SMALL_THREADS = 1024;
constexpr int FFN_SPMV_THREADS = 128;
constexpr int FFN_TILE = 128;


__device__ __forceinline__ float warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__device__ __forceinline__ float sigmoid_fast(float x) {
    return 1.0f / (1.0f + __expf(-x));
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t x) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = static_cast<uint32_t>(x) << 16;
    return v.f;
}

__device__ __forceinline__ half2 half2_from_u32(uint32_t x) {
    union {
        uint32_t u;
        half2 h;
    } v;
    v.u = x;
    return v.h;
}

__device__ __forceinline__ void load_half4_float2_u64(const half* __restrict__ p, float2& lo, float2& hi) {
    union {
        unsigned long long u64;
        uint2 u32;
    } v;
    v.u64 = *reinterpret_cast<const unsigned long long*>(p);
    lo = __half22float2(half2_from_u32(v.u32.x));
    hi = __half22float2(half2_from_u32(v.u32.y));
}

template <int THREADS>
__device__ __forceinline__ float block_sum_all(float v) {
    __shared__ float partial[THREADS / 32];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) {
        partial[warp] = v;
    }
    __syncthreads();
    v = threadIdx.x < (THREADS / 32) ? partial[lane] : 0.0f;
    if (warp == 0) {
        v = warp_sum(v);
    }
    if (threadIdx.x == 0) {
        partial[0] = v;
    }
    __syncthreads();
    return partial[0];
}

__global__ void emb_ln0_bf16_to_f16_kernel(
    int V,
    int C,
    const uint16_t* __restrict__ emb,
    const uint16_t* __restrict__ weight,
    const uint16_t* __restrict__ bias,
    half* __restrict__ out,
    float eps) {
    int tok = blockIdx.x;
    if (tok >= V) {
        return;
    }
    const uint16_t* x = emb + static_cast<int64_t>(tok) * C;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        sum += bf16_bits_to_float(x[c]);
    }
    float mean = block_sum_all<256>(sum) / static_cast<float>(C);

    float var = 0.0f;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float d = bf16_bits_to_float(x[c]) - mean;
        var += d * d;
    }
    float rstd = rsqrtf(block_sum_all<256>(var) / static_cast<float>(C) + eps);

    half* y = out + static_cast<int64_t>(tok) * C;
    for (int c = threadIdx.x; c < C; c += blockDim.x) {
        float v = (bf16_bits_to_float(x[c]) - mean) * rstd * bf16_bits_to_float(weight[c]) + bf16_bits_to_float(bias[c]);
        y[c] = __float2half_rn(v);
    }
}

__global__ void emb_lookup_f16_kernel(
    int rows,
    int V,
    int C,
    const half* __restrict__ emb,
    const int64_t* __restrict__ tokens,
    half* __restrict__ out) {
    int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int64_t tok64 = tokens[row];
    if (tok64 < 0) {
        tok64 = 0;
    }
    if (tok64 >= V) {
        tok64 = V - 1;
    }
    const half2* src = reinterpret_cast<const half2*>(emb + tok64 * static_cast<int64_t>(C));
    half2* dst = reinterpret_cast<half2*>(out + row * static_cast<int64_t>(C));
    int C2 = C >> 1;
    for (int i = threadIdx.x; i < C2; i += blockDim.x) {
        dst[i] = src[i];
    }
}

template <int THREADS>
__global__ __launch_bounds__(THREADS, 1) void add_last_layer_norm_f16_kernel(
    const half* __restrict__ x,
    const half* __restrict__ residual,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ out,
    int B,
    int T,
    int C,
    float eps) {
    int b = blockIdx.x;
    if (b >= B) {
        return;
    }
    int64_t src = (static_cast<int64_t>(b) * T + (T - 1)) * C;
    int64_t dst = static_cast<int64_t>(b) * C;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < C; c += THREADS) {
        sum += __half2float(x[src + c]) + __half2float(residual[src + c]);
    }
    float mean = block_sum_all<THREADS>(sum) / static_cast<float>(C);

    float var = 0.0f;
    for (int c = threadIdx.x; c < C; c += THREADS) {
        float v = __half2float(x[src + c]) + __half2float(residual[src + c]);
        float d = v - mean;
        var += d * d;
    }
    float rstd = rsqrtf(block_sum_all<THREADS>(var) / static_cast<float>(C) + eps);

    for (int c = threadIdx.x; c < C; c += THREADS) {
        float v = __half2float(x[src + c]) + __half2float(residual[src + c]);
        out[dst + c] = __float2half_rn((v - mean) * rstd * __half2float(weight[c]) + __half2float(bias[c]));
    }
}


__global__ void lnx_rkvres_xg_kernel(
    int H,
    const half* __restrict__ x,
    const half* __restrict__ r,
    const half* __restrict__ k,
    const half* __restrict__ v,
    const half* __restrict__ r_k,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ g,
    half* __restrict__ out,
    int64_t bth_size) {
    __shared__ float partial[2];
    int bth = blockIdx.x;
    if (bth >= bth_size) {
        return;
    }
    int lane = threadIdx.x;
    int warp = lane >> 5;
    int warp_lane = lane & 31;
    int h = bth % H;
    int64_t base = static_cast<int64_t>(bth) * HEAD_SIZE;
    int64_t cbase = static_cast<int64_t>(h) * HEAD_SIZE;
    int64_t idx = base + lane;
    int64_t c = cbase + lane;

    float xv = __half2float(x[idx]);
    float sum = warp_sum(xv);
    if (warp_lane == 0) {
        partial[warp] = sum;
    }
    __syncthreads();
    float mean = (partial[0] + partial[1]) * (1.0f / 64.0f);
    __syncthreads();

    float d = xv - mean;
    float ss = warp_sum(d * d);
    if (warp_lane == 0) {
        partial[warp] = ss;
    }
    __syncthreads();
    float rstd = rsqrtf((partial[0] + partial[1]) * (1.0f / 64.0f) + TMIX_LN_X_EPS);
    __syncthreads();

    float rv = __half2float(r[idx]);
    float kv = __half2float(k[idx]);
    float vv = __half2float(v[idx]);
    float dot = warp_sum(rv * kv * __half2float(r_k[c]));
    if (warp_lane == 0) {
        partial[warp] = dot;
    }
    __syncthreads();
    float rkv = partial[0] + partial[1];
    float y = (d * rstd * __half2float(weight[c]) + __half2float(bias[c]) + rkv * vv) * __half2float(g[idx]);
    out[idx] = __float2half_rn(y);
}


__global__ void zero_vec4_kernel(half* __restrict__ out, int64_t n_vec4) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n_vec4) {
        reinterpret_cast<int4*>(out)[i] = make_int4(0, 0, 0, 0);
    }
}


__global__ __launch_bounds__(FFN_SPMV_THREADS, 4) void cmix_sparse_down_relu_one_vtile_hfma2_split2_kernel(
    int C,
    const half* __restrict__ preact,
    const half* __restrict__ value_weight_tiled,
    half* __restrict__ out) {
    __shared__ __align__(256) half vec_slice[FFN_TILE];
    __shared__ __align__(256) int nnz_ids[FFN_TILE];
    __shared__ int nnz_count;
    __shared__ int warp_counts[FFN_TILE / 32];
    __shared__ int warp_prefix[FFN_TILE / 32];

    int f_block = blockIdx.x;
    int c_block = blockIdx.y;
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int start_f = f_block * FFN_TILE;

    half relu2_h = __float2half_rn(0.0f);
    if (tid < FFN_TILE) {
        float v = fmaxf(__half2float(preact[start_f + tid]), 0.0f);
        relu2_h = __float2half_rn(v * v);
        vec_slice[tid] = relu2_h;
    }

    bool nonzero = false;
    int local_pos = 0;
    if (tid < FFN_TILE) {
        nonzero = bool(__half_as_ushort(relu2_h) << 1);
        unsigned mask = __ballot_sync(0xffffffffu, nonzero);
        local_pos = __popc(mask & ((1u << lane) - 1u));
        if (lane == 0) {
            warp_counts[warp_id] = __popc(mask);
        }
    }
    __syncthreads();

    if (tid == 0) {
        int s = 0;
#pragma unroll
        for (int w = 0; w < FFN_TILE / 32; ++w) {
            warp_prefix[w] = s;
            s += warp_counts[w];
        }
        nnz_count = s;
    }
    __syncthreads();

    if (tid < FFN_TILE && nonzero) {
        nnz_ids[warp_prefix[warp_id] + local_pos] = tid;
    }
    __syncthreads();

    half2 acc0;
    half2 acc1;
    *reinterpret_cast<int*>(&acc0) = 0;
    *reinterpret_cast<int*>(&acc1) = 0;
    constexpr int C_TILE = 2 * FFN_SPMV_THREADS;
    int c_blocks = C / C_TILE;
    int c0 = c_block * C_TILE + tid * 2;
    int tile_base = ((f_block * c_blocks + c_block) * FFN_TILE) * C_TILE;
    for (int i = 0; i < nnz_count; i += 2) {
        int local_f0 = nnz_ids[i];
        half2 mat0 = *reinterpret_cast<const half2*>(
            value_weight_tiled + static_cast<int64_t>(tile_base) + local_f0 * C_TILE + tid * 2);
        acc0 = __hfma2(__half2half2(vec_slice[local_f0]), mat0, acc0);
        if (i + 1 < nnz_count) {
            int local_f1 = nnz_ids[i + 1];
            half2 mat1 = *reinterpret_cast<const half2*>(
                value_weight_tiled + static_cast<int64_t>(tile_base) + local_f1 * C_TILE + tid * 2);
            acc1 = __hfma2(__half2half2(vec_slice[local_f1]), mat1, acc1);
        }
    }
    half2 acc = __hadd2(acc0, acc1);
    atomicAdd(reinterpret_cast<half2*>(out + c0), acc);
}


template <int THREADS>
__global__ __launch_bounds__(THREADS, 1) void ln_mix6_c4096_kernel(
    const half* __restrict__ x,
    const half* __restrict__ residual,
    half* __restrict__ shift_state,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ x_r,
    const half* __restrict__ x_w,
    const half* __restrict__ x_k,
    const half* __restrict__ x_v,
    const half* __restrict__ x_a,
    const half* __restrict__ x_g,
    half* __restrict__ x_out,
    half* __restrict__ out_r,
    half* __restrict__ out_w,
    half* __restrict__ out_k,
    half* __restrict__ out_v,
    half* __restrict__ out_a,
    half* __restrict__ out_g,
    int64_t rows,
    float eps) {
    int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int64_t base = row * static_cast<int64_t>(LN_SMALL_C);
    int64_t base2 = row * static_cast<int64_t>(LN_SMALL_C >> 1);
    constexpr int pairs = LN_SMALL_C >> 1;

    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / THREADS; ++k) {
        int c = threadIdx.x + k * THREADS;
        sum += __half2float(x[base + c]) + __half2float(residual[base + c]);
    }
    float mean = block_sum_all<THREADS>(sum) * (1.0f / static_cast<float>(LN_SMALL_C));

    float sum_var = 0.0f;
#pragma unroll
    for (int k = 0; k < LN_SMALL_C / THREADS; ++k) {
        int c = threadIdx.x + k * THREADS;
        float v = __half2float(x[base + c]) + __half2float(residual[base + c]);
        float d = v - mean;
        sum_var += d * d;
    }
    float rstd = rsqrtf(block_sum_all<THREADS>(sum_var) * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);

    const half2* x2 = reinterpret_cast<const half2*>(x);
    const half2* residual2 = reinterpret_cast<const half2*>(residual);
    half2* shift2 = reinterpret_cast<half2*>(shift_state);
    const half2* weight2 = reinterpret_cast<const half2*>(weight);
    const half2* bias2 = reinterpret_cast<const half2*>(bias);
    const half2* xr2 = reinterpret_cast<const half2*>(x_r);
    const half2* xw2 = reinterpret_cast<const half2*>(x_w);
    const half2* xk2 = reinterpret_cast<const half2*>(x_k);
    const half2* xv2 = reinterpret_cast<const half2*>(x_v);
    const half2* xa2 = reinterpret_cast<const half2*>(x_a);
    const half2* xg2 = reinterpret_cast<const half2*>(x_g);
    half2* xo2 = reinterpret_cast<half2*>(x_out);
    half2* r2 = reinterpret_cast<half2*>(out_r);
    half2* w2 = reinterpret_cast<half2*>(out_w);
    half2* k2 = reinterpret_cast<half2*>(out_k);
    half2* v2 = reinterpret_cast<half2*>(out_v);
    half2* a2 = reinterpret_cast<half2*>(out_a);
    half2* g2 = reinterpret_cast<half2*>(out_g);
#pragma unroll
    for (int k = 0; k < pairs / THREADS; ++k) {
        int p = threadIdx.x + k * THREADS;
        float2 xv = __half22float2(x2[base2 + p]);
        float2 rv = __half22float2(residual2[base2 + p]);
        float2 ww = __half22float2(weight2[p]);
        float2 bb = __half22float2(bias2[p]);
        float2 prev = __half22float2(shift2[base2 + p]);
        float x0 = xv.x + rv.x;
        float x1 = xv.y + rv.y;
        half2 y2 = __floats2half2_rn((x0 - mean) * rstd * ww.x + bb.x, (x1 - mean) * rstd * ww.y + bb.y);
        float2 yv = __half22float2(y2);
        float dx0 = prev.x - yv.x;
        float dx1 = prev.y - yv.y;
        float2 mr = __half22float2(xr2[p]);
        float2 mw = __half22float2(xw2[p]);
        float2 mk = __half22float2(xk2[p]);
        float2 mv = __half22float2(xv2[p]);
        float2 ma = __half22float2(xa2[p]);
        float2 mg = __half22float2(xg2[p]);
        xo2[base2 + p] = __floats2half2_rn(x0, x1);
        r2[base2 + p] = __floats2half2_rn(yv.x + dx0 * mr.x, yv.y + dx1 * mr.y);
        w2[base2 + p] = __floats2half2_rn(yv.x + dx0 * mw.x, yv.y + dx1 * mw.y);
        k2[base2 + p] = __floats2half2_rn(yv.x + dx0 * mk.x, yv.y + dx1 * mk.y);
        v2[base2 + p] = __floats2half2_rn(yv.x + dx0 * mv.x, yv.y + dx1 * mv.y);
        a2[base2 + p] = __floats2half2_rn(yv.x + dx0 * ma.x, yv.y + dx1 * ma.y);
        g2[base2 + p] = __floats2half2_rn(yv.x + dx0 * mg.x, yv.y + dx1 * mg.y);
        shift2[base2 + p] = y2;
    }
}


template <int THREADS>
__global__ __launch_bounds__(THREADS, 1) void add_ln_cmix_mix_c4096_kernel(
    const half* __restrict__ x,
    const half* __restrict__ residual,
    half* __restrict__ shift_state,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    const half* __restrict__ x_k,
    half* __restrict__ x_out,
    half* __restrict__ mixed,
    int64_t rows,
    float eps) {
    int64_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    int64_t base = row * static_cast<int64_t>(LN_SMALL_C);
    int64_t base2 = base >> 1;
    constexpr int pairs = LN_SMALL_C >> 1;
    const half2* x2 = reinterpret_cast<const half2*>(x);
    const half2* residual2 = reinterpret_cast<const half2*>(residual);

    float sum = 0.0f;
#pragma unroll
    for (int k = 0; k < pairs / THREADS; ++k) {
        int p = threadIdx.x + k * THREADS;
        float2 xv = __half22float2(x2[base2 + p]);
        float2 rv = __half22float2(residual2[base2 + p]);
        sum += xv.x + rv.x + xv.y + rv.y;
    }
    float mean = block_sum_all<THREADS>(sum) * (1.0f / static_cast<float>(LN_SMALL_C));

    float sum_var = 0.0f;
#pragma unroll
    for (int k = 0; k < pairs / THREADS; ++k) {
        int p = threadIdx.x + k * THREADS;
        float2 xv = __half22float2(x2[base2 + p]);
        float2 rv = __half22float2(residual2[base2 + p]);
        float d0 = xv.x + rv.x - mean;
        float d1 = xv.y + rv.y - mean;
        sum_var += d0 * d0 + d1 * d1;
    }
    float rstd = rsqrtf(block_sum_all<THREADS>(sum_var) * (1.0f / static_cast<float>(LN_SMALL_C)) + eps);

    half2* shift2 = reinterpret_cast<half2*>(shift_state);
    const half2* weight2 = reinterpret_cast<const half2*>(weight);
    const half2* bias2 = reinterpret_cast<const half2*>(bias);
    const half2* xk2 = reinterpret_cast<const half2*>(x_k);
    half2* xo2 = reinterpret_cast<half2*>(x_out);
    half2* mixed2 = reinterpret_cast<half2*>(mixed);
#pragma unroll
    for (int k = 0; k < pairs / THREADS; ++k) {
        int p = threadIdx.x + k * THREADS;
        float2 xv = __half22float2(x2[base2 + p]);
        float2 rv = __half22float2(residual2[base2 + p]);
        float2 ww = __half22float2(weight2[p]);
        float2 bb = __half22float2(bias2[p]);
        float2 prev = __half22float2(shift2[base2 + p]);
        float2 mix = __half22float2(xk2[p]);
        float x0 = xv.x + rv.x;
        float x1 = xv.y + rv.y;
        half2 y2 = __floats2half2_rn((x0 - mean) * rstd * ww.x + bb.x, (x1 - mean) * rstd * ww.y + bb.y);
        float2 yv = __half22float2(y2);
        xo2[base2 + p] = __floats2half2_rn(x0, x1);
        mixed2[base2 + p] = __floats2half2_rn(yv.x + (prev.x - yv.x) * mix.x, yv.y + (prev.y - yv.y) * mix.y);
        shift2[base2 + p] = y2;
    }
}


template <int THREADS>
__device__ __forceinline__ float row_dot_half2(
    const half* __restrict__ x,
    const half* __restrict__ w,
    int row,
    int C) {
    const half2* x2 = reinterpret_cast<const half2*>(x);
    const half2* w2 = reinterpret_cast<const half2*>(w + static_cast<int64_t>(row) * C);
    int C2 = C >> 1;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < C2; i += THREADS) {
        float2 xv = __half22float2(x2[i]);
        float2 wv = __half22float2(w2[i]);
        sum = fmaf(xv.x, wv.x, sum);
        sum = fmaf(xv.y, wv.y, sum);
    }
    if ((C & 1) && threadIdx.x == 0) {
        int i = C - 1;
        sum = fmaf(__half2float(x[i]), __half2float(w[static_cast<int64_t>(row) * C + i]), sum);
    }
    return block_sum_all<THREADS>(sum);
}


template <int THREADS, int OUT_TILE>
__global__ __launch_bounds__(THREADS, 1) void row1_linear_exact4_kernel(
    int K,
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ y) {
    int n0 = blockIdx.x * OUT_TILE;
    float acc[OUT_TILE];
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
        acc[j] = 0.0f;
    }
    for (int k = threadIdx.x << 2; k < K; k += THREADS << 2) {
        float2 x0 = __half22float2(*reinterpret_cast<const half2*>(x + k));
        float2 x1 = __half22float2(*reinterpret_cast<const half2*>(x + k + 2));
#pragma unroll
        for (int j = 0; j < OUT_TILE; ++j) {
            const half* wj = w + static_cast<int64_t>(n0 + j) * K + k;
            float2 w0 = __half22float2(*reinterpret_cast<const half2*>(wj));
            float2 w1 = __half22float2(*reinterpret_cast<const half2*>(wj + 2));
            acc[j] = fmaf(x0.x, w0.x, acc[j]);
            acc[j] = fmaf(x0.y, w0.y, acc[j]);
            acc[j] = fmaf(x1.x, w1.x, acc[j]);
            acc[j] = fmaf(x1.y, w1.y, acc[j]);
        }
    }
    __shared__ float partial[THREADS / 32][OUT_TILE];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
        float v = warp_sum(acc[j]);
        if (lane == 0) {
            partial[warp][j] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < OUT_TILE; ++j) {
            float sum = 0.0f;
#pragma unroll
            for (int widx = 0; widx < THREADS / 32; ++widx) {
                sum += partial[widx][j];
            }
            y[n0 + j] = __float2half_rn(sum);
        }
    }
}

template <int THREADS, int OUT_TILE>
__global__ __launch_bounds__(THREADS, 1) void row1_linear_exact4_vec4_kernel(
    int K,
    const half* __restrict__ x,
    const half* __restrict__ w,
    half* __restrict__ y) {
    int n0 = blockIdx.x * OUT_TILE;
    float acc[OUT_TILE];
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
        acc[j] = 0.0f;
    }
    for (int k = threadIdx.x << 2; k < K; k += THREADS << 2) {
        float2 x0;
        float2 x1;
        load_half4_float2_u64(x + k, x0, x1);
#pragma unroll
        for (int j = 0; j < OUT_TILE; ++j) {
            const half* wj = w + static_cast<int64_t>(n0 + j) * K + k;
            float2 w0;
            float2 w1;
            load_half4_float2_u64(wj, w0, w1);
            acc[j] = fmaf(x0.x, w0.x, acc[j]);
            acc[j] = fmaf(x0.y, w0.y, acc[j]);
            acc[j] = fmaf(x1.x, w1.x, acc[j]);
            acc[j] = fmaf(x1.y, w1.y, acc[j]);
        }
    }
    __shared__ float partial[THREADS / 32][OUT_TILE];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
        float v = warp_sum(acc[j]);
        if (lane == 0) {
            partial[warp][j] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < OUT_TILE; ++j) {
            float sum = 0.0f;
#pragma unroll
            for (int widx = 0; widx < THREADS / 32; ++widx) {
                sum += partial[widx][j];
            }
            y[n0 + j] = __float2half_rn(sum);
        }
    }
}


template <int THREADS, int RKV_OUT_TILE>
__device__ __forceinline__ void rkv_executor_tile_body(
    const half* __restrict__ xr,
    const half* __restrict__ xk,
    const half* __restrict__ xv,
    const half* __restrict__ wr,
    const half* __restrict__ wk,
    const half* __restrict__ wv,
    half* __restrict__ yr,
    half* __restrict__ yk,
    half* __restrict__ yv,
    int task,
    int C) {
    __shared__ float partial[THREADS / 32][RKV_OUT_TILE];
    int rows_per_group = C / RKV_OUT_TILE;
    int group = task / rows_per_group;
    int row0 = (task - group * rows_per_group) * RKV_OUT_TILE;
    const half* inp = group == 0 ? xr : (group == 1 ? xk : xv);
    const half* wt = group == 0 ? wr : (group == 1 ? wk : wv);
    half* out = group == 0 ? yr : (group == 1 ? yk : yv);

    float acc[RKV_OUT_TILE];
#pragma unroll
    for (int j = 0; j < RKV_OUT_TILE; ++j) {
        acc[j] = 0.0f;
    }
    for (int k = threadIdx.x << 2; k < C; k += THREADS << 2) {
        float2 x0 = __half22float2(*reinterpret_cast<const half2*>(inp + k));
        float2 x1 = __half22float2(*reinterpret_cast<const half2*>(inp + k + 2));
#pragma unroll
        for (int j = 0; j < RKV_OUT_TILE; ++j) {
            const half* wj = wt + static_cast<int64_t>(row0 + j) * C + k;
            float2 w0 = __half22float2(*reinterpret_cast<const half2*>(wj));
            float2 w1v = __half22float2(*reinterpret_cast<const half2*>(wj + 2));
            acc[j] = fmaf(x0.x, w0.x, acc[j]);
            acc[j] = fmaf(x0.y, w0.y, acc[j]);
            acc[j] = fmaf(x1.x, w1v.x, acc[j]);
            acc[j] = fmaf(x1.y, w1v.y, acc[j]);
        }
    }

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < RKV_OUT_TILE; ++j) {
        float v = warp_sum(acc[j]);
        if (lane == 0) {
            partial[warp][j] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < RKV_OUT_TILE; ++j) {
            float sum = 0.0f;
#pragma unroll
            for (int widx = 0; widx < THREADS / 32; ++widx) {
                sum += partial[widx][j];
            }
            out[row0 + j] = __float2half_rn(sum);
        }
    }
}


template <int THREADS>
__device__ __forceinline__ void rkv_executor_tile_body_h2stage_hfma2_splitacc_k2pipe(
    const half* __restrict__ xr,
    const half* __restrict__ xk,
    const half* __restrict__ xv,
    const half* __restrict__ wr,
    const half* __restrict__ wk,
    const half* __restrict__ wv,
    half* __restrict__ yr,
    half* __restrict__ yk,
    half* __restrict__ yv,
    int task,
    int C) {
    constexpr int RKV_OUT_TILE = 2;
    __shared__ float partial[THREADS / 32][RKV_OUT_TILE];
    int rows_per_group = C / RKV_OUT_TILE;
    int group = task / rows_per_group;
    int row0 = (task - group * rows_per_group) * RKV_OUT_TILE;
    const half* inp = group == 0 ? xr : (group == 1 ? xk : xv);
    const half* wt = group == 0 ? wr : (group == 1 ? wk : wv);
    half* out = group == 0 ? yr : (group == 1 ? yk : yv);

    half2 acc00h;
    half2 acc01h;
    half2 acc10h;
    half2 acc11h;
    *reinterpret_cast<int*>(&acc00h) = 0;
    *reinterpret_cast<int*>(&acc01h) = 0;
    *reinterpret_cast<int*>(&acc10h) = 0;
    *reinterpret_cast<int*>(&acc11h) = 0;
    for (int k0 = threadIdx.x << 1; k0 < C; k0 += THREADS << 2) {
        const int k1 = k0 + (THREADS << 1);
        const half* w00 = wt + static_cast<int64_t>(row0) * C + k0;
        const half* w10 = w00 + C;
        half2 hx0 = *reinterpret_cast<const half2*>(inp + k0);
        half2 hw00 = *reinterpret_cast<const half2*>(w00);
        half2 hw10 = *reinterpret_cast<const half2*>(w10);
        if (k1 < C) {
            const half* w01 = wt + static_cast<int64_t>(row0) * C + k1;
            const half* w11 = w01 + C;
            half2 hx1 = *reinterpret_cast<const half2*>(inp + k1);
            half2 hw01 = *reinterpret_cast<const half2*>(w01);
            half2 hw11 = *reinterpret_cast<const half2*>(w11);
            acc00h = __hfma2(hx0, hw00, acc00h);
            acc01h = __hfma2(hx1, hw01, acc01h);
            acc10h = __hfma2(hx0, hw10, acc10h);
            acc11h = __hfma2(hx1, hw11, acc11h);
        } else {
            acc00h = __hfma2(hx0, hw00, acc00h);
            acc10h = __hfma2(hx0, hw10, acc10h);
        }
    }

    float2 acc0f = __half22float2(__hadd2(acc00h, acc01h));
    float2 acc1f = __half22float2(__hadd2(acc10h, acc11h));
    float acc0 = acc0f.x + acc0f.y;
    float acc1 = acc1f.x + acc1f.y;
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    float v0 = warp_sum(acc0);
    float v1 = warp_sum(acc1);
    if (lane == 0) {
        partial[warp][0] = v0;
        partial[warp][1] = v1;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum0 = 0.0f;
        float sum1 = 0.0f;
#pragma unroll
        for (int widx = 0; widx < THREADS / 32; ++widx) {
            sum0 += partial[widx][0];
            sum1 += partial[widx][1];
        }
        out[row0] = __float2half_rn(sum0);
        out[row0 + 1] = __float2half_rn(sum1);
    }
}


template <int THREADS>
__device__ __forceinline__ void rkv_executor_tile_body_h2stage_hfma2_splitacc_k2pipe_group_interleave(
    const half* __restrict__ xr,
    const half* __restrict__ xk,
    const half* __restrict__ xv,
    const half* __restrict__ wr,
    const half* __restrict__ wk,
    const half* __restrict__ wv,
    half* __restrict__ yr,
    half* __restrict__ yk,
    half* __restrict__ yv,
    int task,
    int C) {
    constexpr int RKV_OUT_TILE = 2;
    int rows_per_group = C / RKV_OUT_TILE;
    int row_pair = task / 3;
    int group = task - row_pair * 3;
    int mapped_task = group * rows_per_group + row_pair;
    rkv_executor_tile_body_h2stage_hfma2_splitacc_k2pipe<THREADS>(xr, xk, xv, wr, wk, wv, yr, yk, yv, mapped_task, C);
}

template <int THREADS>
__device__ __forceinline__ void lowrank_pre_compact_body(
    const half* __restrict__ xw,
    const half* __restrict__ xa,
    const half* __restrict__ xg,
    const half* __restrict__ xlr_v,
    const half* __restrict__ w1_t,
    const half* __restrict__ a1_t,
    const half* __restrict__ g1_t,
    const half* __restrict__ v1_t,
    half* __restrict__ w1,
    half* __restrict__ a1,
    half* __restrict__ g1,
    half* __restrict__ v1,
    int task,
    int M,
    int C,
    int Rw,
    int Ra,
    int Rg,
    int Rv) {
    int n0 = M * Rw;
    int n1 = n0 + M * Ra;
    int n2 = n1 + M * Rg;
    const half* x = xw;
    const half* wt = w1_t;
    half* y = w1;
    int R = Rw;
    int base = 0;
    if (task >= n2) {
        x = xlr_v;
        wt = v1_t;
        y = v1;
        R = Rv;
        base = n2;
    } else if (task >= n1) {
        x = xg;
        wt = g1_t;
        y = g1;
        R = Rg;
        base = n1;
    } else if (task >= n0) {
        x = xa;
        wt = a1_t;
        y = a1;
        R = Ra;
        base = n0;
    }
    int local = task - base;
    int m = local / R;
    int r = local - m * R;
    if (m < M) {
        float sum = row_dot_half2<THREADS>(x + static_cast<int64_t>(m) * C, wt, r, C);
        if (threadIdx.x == 0) {
            y[static_cast<int64_t>(m) * R + r] = __float2half_rn(sum);
        }
    }
}

template <int THREADS, int OUT_TILE>
__device__ __forceinline__ int lowrank_rank_out4_body(
    const half* __restrict__ w1,
    const half* __restrict__ a1,
    const half* __restrict__ g1,
    const half* __restrict__ v1,
    const half* __restrict__ w2_t,
    const half* __restrict__ a2_t,
    const half* __restrict__ g2_t,
    const half* __restrict__ v2_t,
    const half* __restrict__ v,
    const half* __restrict__ v_first,
    const half* __restrict__ v0,
    half* __restrict__ w,
    half* __restrict__ a,
    half* __restrict__ g,
    half* __restrict__ v_out,
    int task,
    int M,
    int C,
    int Rw,
    int Ra,
    int Rg,
    int Rv) {
    int tiles = (C + OUT_TILE - 1) / OUT_TILE;
    int tile = task % tiles;
    int rem = task / tiles;
    int group = rem % 4;
    int m = rem / 4;
    int n0 = tile * OUT_TILE;
    int R = Rw;
    const half* x = w1;
    const half* wt = w2_t;
    half* y = w;
    if (group == 1) {
        R = Ra;
        x = a1;
        wt = a2_t;
        y = a;
    } else if (group == 2) {
        R = Rg;
        x = g1;
        wt = g2_t;
        y = g;
    } else if (group == 3) {
        R = Rv;
        x = v1;
        wt = v2_t;
        y = v_out;
    }
    if (m >= M) {
        return group;
    }
    float acc[OUT_TILE];
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
        acc[j] = 0.0f;
    }
    const half* x_row = x + static_cast<int64_t>(m) * R;
    for (int r = threadIdx.x; r < R; r += THREADS) {
        float xv0 = __half2float(x_row[r]);
        if (group == 0) {
            xv0 = tanhf(xv0);
        } else if (group == 2) {
            xv0 = sigmoid_fast(xv0);
        }
#pragma unroll
        for (int j = 0; j < OUT_TILE; ++j) {
            int n = n0 + j;
            if (n < C) {
                acc[j] = fmaf(xv0, __half2float(wt[static_cast<int64_t>(n) * R + r]), acc[j]);
            }
        }
    }
    __shared__ float partial[THREADS / 32][OUT_TILE];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
        acc[j] = warp_sum(acc[j]);
        if (lane == 0) {
            partial[warp][j] = acc[j];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < OUT_TILE; ++j) {
            float sum = 0.0f;
#pragma unroll
            for (int u = 0; u < THREADS / 32; ++u) {
                sum += partial[u][j];
            }
            int n = n0 + j;
            if (n < C) {
                int64_t idx = static_cast<int64_t>(m) * C + n;
                if (group == 3) {
                    float vv = __half2float(v[idx]);
                    float vf = __half2float(v_first[idx]);
                    float gate = sigmoid_fast(__half2float(v0[n]) + sum);
                    y[idx] = __float2half_rn(fmaf(vf - vv, gate, vv));
                } else {
                    y[idx] = __float2half_rn(sum);
                }
            }
        }
    }
    return group;
}

template <int THREADS, int OUT_TILE, bool KK_LANES = true>
__device__ __forceinline__ int lowrank_rank_out4_kk_body(
    const half* __restrict__ w1,
    const half* __restrict__ a1,
    const half* __restrict__ g1,
    const half* __restrict__ v1,
    const half* __restrict__ w2_t,
    const half* __restrict__ a2_t,
    const half* __restrict__ g2_t,
    const half* __restrict__ v2_t,
    const half* __restrict__ v,
    const half* __restrict__ v_first,
    const half* __restrict__ v0,
    const half* __restrict__ k_raw,
    const half* __restrict__ k_k,
    const half* __restrict__ a0,
    const half* __restrict__ k_a,
    half* __restrict__ w,
    half* __restrict__ a,
    half* __restrict__ g,
    half* __restrict__ v_out,
    half* __restrict__ new_k,
    half* __restrict__ neg_kk,
    half* __restrict__ kka,
    int task,
    int M,
    int C,
    int Rw,
    int Ra,
    int Rg,
    int Rv) {
    int tiles = (C + OUT_TILE - 1) / OUT_TILE;
    int tile = task % tiles;
    int rem = task / tiles;
    int group = rem % 4;
    int m = rem / 4;
    int n0 = tile * OUT_TILE;
    int R = Rw;
    const half* x = w1;
    const half* wt = w2_t;
    half* y = w;
    if (group == 1) {
        R = Ra;
        x = a1;
        wt = a2_t;
        y = a;
    } else if (group == 2) {
        R = Rg;
        x = g1;
        wt = g2_t;
        y = g;
    } else if (group == 3) {
        R = Rv;
        x = v1;
        wt = v2_t;
        y = v_out;
    }
    if (m >= M) {
        return group;
    }
    float acc[OUT_TILE];
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
        acc[j] = 0.0f;
    }
    const half* x_row = x + static_cast<int64_t>(m) * R;
    for (int r = threadIdx.x; r < R; r += THREADS) {
        float xv0 = __half2float(x_row[r]);
        if (group == 0) {
            xv0 = tanhf(xv0);
        } else if (group == 2) {
            xv0 = sigmoid_fast(xv0);
        }
#pragma unroll
        for (int j = 0; j < OUT_TILE; ++j) {
            int n = n0 + j;
            if (n < C) {
                acc[j] = fmaf(xv0, __half2float(wt[static_cast<int64_t>(n) * R + r]), acc[j]);
            }
        }
    }
    __shared__ float partial[THREADS / 32][OUT_TILE];
    __shared__ float total[OUT_TILE];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
        acc[j] = warp_sum(acc[j]);
        if (lane == 0) {
            partial[warp][j] = acc[j];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < OUT_TILE; ++j) {
            float sum = 0.0f;
#pragma unroll
            for (int u = 0; u < THREADS / 32; ++u) {
                sum += partial[u][j];
            }
            total[j] = sum;
            int n = n0 + j;
            if (n < C) {
                int64_t idx = static_cast<int64_t>(m) * C + n;
                if (group == 3) {
                    float vv = __half2float(v[idx]);
                    float vf = __half2float(v_first[idx]);
                    float gate = sigmoid_fast(__half2float(v0[n]) + sum);
                    y[idx] = __float2half_rn(fmaf(vf - vv, gate, vv));
                } else {
                    y[idx] = __float2half_rn(sum);
                }
            }
        }
    }
    if (group != 1) {
        return group;
    }
    __syncthreads();
    int head_base = (n0 / HEAD_SIZE) * HEAD_SIZE;
    int64_t row_base = static_cast<int64_t>(m) * C;
    float norm_local = 0.0f;
    for (int q = threadIdx.x; q < HEAD_SIZE; q += THREADS) {
        int c = head_base + q;
        if (c < C) {
            float u = __half2float(k_raw[row_base + c]) * __half2float(k_k[c]);
            norm_local = fmaf(u, u, norm_local);
        }
    }
    float norm = block_sum_all<THREADS>(norm_local);
    float inv_norm = 1.0f / fmaxf(sqrtf(norm), KK_NORMALIZE_EPS);
    if constexpr (KK_LANES) {
        int j = threadIdx.x;
        if (j < OUT_TILE) {
            int n = n0 + j;
            if (n < C) {
                int64_t idx = row_base + n;
                half a_half = __float2half_rn(total[j]);
                float a_val = __half2float(a_half);
                float kr = __half2float(k_raw[idx]);
                float kk = kr * __half2float(k_k[n]) * inv_norm;
                float gate = sigmoid_fast(__half2float(a0[n]) + a_val);
                float ka = __half2float(k_a[n]);
                new_k[idx] = __float2half_rn(kr * fmaf(gate, ka, 1.0f - ka));
                neg_kk[idx] = __float2half_rn(-kk);
                kka[idx] = __float2half_rn(kk * gate);
            }
        }
        return group;
    }
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < OUT_TILE; ++j) {
            int n = n0 + j;
            if (n < C) {
                int64_t idx = row_base + n;
                half a_half = __float2half_rn(total[j]);
                float a_val = __half2float(a_half);
                float kr = __half2float(k_raw[idx]);
                float kk = kr * __half2float(k_k[n]) * inv_norm;
                float gate = sigmoid_fast(__half2float(a0[n]) + a_val);
                float ka = __half2float(k_a[n]);
                new_k[idx] = __float2half_rn(kr * fmaf(gate, ka, 1.0f - ka));
                neg_kk[idx] = __float2half_rn(-kk);
                kka[idx] = __float2half_rn(kk * gate);
            }
        }
    }
    return group;
}

template <int THREADS, int RKV_OUT_TILE, bool FORCE_TASK_SYNC, int MIN_BLOCKS = 1, int ROLE_ORDER = -1>
__global__ __launch_bounds__(THREADS, MIN_BLOCKS) void rkv_lowrank_pre_executor_kernel(
    const half* __restrict__ xr,
    const half* __restrict__ xk,
    const half* __restrict__ xv,
    const half* __restrict__ wr,
    const half* __restrict__ wk,
    const half* __restrict__ wv,
    half* __restrict__ yr,
    half* __restrict__ yk,
    half* __restrict__ yv,
    const half* __restrict__ xw,
    const half* __restrict__ xa,
    const half* __restrict__ xg,
    const half* __restrict__ xlr_v,
    const half* __restrict__ w1_t,
    const half* __restrict__ a1_t,
    const half* __restrict__ g1_t,
    const half* __restrict__ v1_t,
    const half* __restrict__ w2_t,
    const half* __restrict__ g2_t,
    half* __restrict__ w1,
    half* __restrict__ a1,
    half* __restrict__ g1,
    half* __restrict__ v1,
    half* __restrict__ w,
    half* __restrict__ g,
    int M,
    int C,
    int Rw,
    int Ra,
    int Rg,
    int Rv,
    int lowrank_worker_budget,
    int role_order) {
    const int rkv_tasks = 3 * (C / RKV_OUT_TILE);
    const int lowrank_tasks = M * (Rw + Ra + Rg + Rv);
    int lowrank_workers = lowrank_worker_budget > 0 ? lowrank_worker_budget : static_cast<int>(gridDim.x) / 4;
    if (lowrank_workers < 0) {
        lowrank_workers = 0;
    }
    if (lowrank_workers > lowrank_tasks) {
        lowrank_workers = lowrank_tasks;
    }
    if (lowrank_workers >= static_cast<int>(gridDim.x)) {
        lowrank_workers = static_cast<int>(gridDim.x) - 1;
    }
    const int bid = static_cast<int>(blockIdx.x);
    const int effective_role_order = ROLE_ORDER >= 0 ? ROLE_ORDER : role_order;
    const bool lowrank_first = effective_role_order == 1;
    const bool lowrank_interleave = effective_role_order == 2;
    const int interleave_stride = 8;
    if (lowrank_interleave) {
        const int interleave_slots = (static_cast<int>(gridDim.x) + interleave_stride - 1) / interleave_stride;
        if (lowrank_workers > interleave_slots) {
            lowrank_workers = interleave_slots;
        }
    }
    int rkv_workers = static_cast<int>(gridDim.x) - lowrank_workers;
    const int interleave_span = lowrank_interleave ? lowrank_workers * interleave_stride : 0;
    const bool interleave_lowrank = lowrank_interleave && bid < interleave_span && (bid % interleave_stride) == 0;
    const int interleave_lowrank_before = bid < interleave_span
        ? (bid + interleave_stride - 1) / interleave_stride
        : lowrank_workers;
    const bool lowrank_worker = lowrank_interleave
        ? interleave_lowrank
        : (lowrank_first ? bid < lowrank_workers : bid >= rkv_workers);
    const int worker_rank = lowrank_worker
        ? (lowrank_interleave ? bid / interleave_stride : (lowrank_first ? bid : bid - rkv_workers))
        : (lowrank_interleave ? bid - interleave_lowrank_before : (lowrank_first ? bid - lowrank_workers : bid));
    const int worker_count = lowrank_worker ? lowrank_workers : rkv_workers;
    const bool needs_task_sync = FORCE_TASK_SYNC || worker_count < (lowrank_worker ? lowrank_tasks : rkv_tasks);
    if (lowrank_worker) {
        for (int task = worker_rank; task < lowrank_tasks; task += worker_count) {
            lowrank_pre_compact_body<THREADS>(
                xw, xa, xg, xlr_v, w1_t, a1_t, g1_t, v1_t, w1, a1, g1, v1,
                task, M, C, Rw, Ra, Rg, Rv);
            if (needs_task_sync) {
                __syncthreads();
            }
        }
    } else {
        for (int task = worker_rank; task < rkv_tasks; task += worker_count) {
            rkv_executor_tile_body_h2stage_hfma2_splitacc_k2pipe_group_interleave<THREADS>(
                xr, xk, xv, wr, wk, wv, yr, yk, yv, task, C);
            if (needs_task_sync) {
                __syncthreads();
            }
        }
    }
}


template <int THREADS, int OUT_TILE, bool KK_LANES = false>
__global__ __launch_bounds__(THREADS, 2) void lowrank_rank_out4_kk_kernel(
    const half* __restrict__ w1,
    const half* __restrict__ a1,
    const half* __restrict__ g1,
    const half* __restrict__ v1,
    const half* __restrict__ w2_t,
    const half* __restrict__ a2_t,
    const half* __restrict__ g2_t,
    const half* __restrict__ v2_t,
    const half* __restrict__ v,
    const half* __restrict__ v_first,
    const half* __restrict__ v0,
    const half* __restrict__ k_raw,
    const half* __restrict__ k_k,
    const half* __restrict__ a0,
    const half* __restrict__ k_a,
    half* __restrict__ w,
    half* __restrict__ a,
    half* __restrict__ g,
    half* __restrict__ v_out,
    half* __restrict__ new_k,
    half* __restrict__ neg_kk,
    half* __restrict__ kka,
    int M,
    int C,
    int Rw,
    int Ra,
    int Rg,
    int Rv) {
    int tile = blockIdx.x;
    int m = blockIdx.y;
    int group = blockIdx.z;
    int n0 = tile * OUT_TILE;
    int R = Rw;
    const half* x = w1;
    const half* wt = w2_t;
    half* y = w;
    if (group == 1) {
        R = Ra;
        x = a1;
        wt = a2_t;
        y = a;
    } else if (group == 2) {
        R = Rg;
        x = g1;
        wt = g2_t;
        y = g;
    } else if (group == 3) {
        R = Rv;
        x = v1;
        wt = v2_t;
        y = v_out;
    }
    if (m >= M) {
        return;
    }
    float acc[OUT_TILE];
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
        acc[j] = 0.0f;
    }
    const half* x_row = x + static_cast<int64_t>(m) * R;
    for (int r = threadIdx.x; r < R; r += THREADS) {
        float xv0 = __half2float(x_row[r]);
        if (group == 0) {
            xv0 = tanhf(xv0);
        } else if (group == 2) {
            xv0 = sigmoid_fast(xv0);
        }
#pragma unroll
        for (int j = 0; j < OUT_TILE; ++j) {
            int n = n0 + j;
            if (n < C) {
                acc[j] = fmaf(xv0, __half2float(wt[static_cast<int64_t>(n) * R + r]), acc[j]);
            }
        }
    }
    __shared__ float partial[THREADS / 32][OUT_TILE];
    __shared__ float total[OUT_TILE];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
        acc[j] = warp_sum(acc[j]);
        if (lane == 0) {
            partial[warp][j] = acc[j];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
#pragma unroll
        for (int j = 0; j < OUT_TILE; ++j) {
            float sum = 0.0f;
#pragma unroll
            for (int u = 0; u < THREADS / 32; ++u) {
                sum += partial[u][j];
            }
            total[j] = sum;
            int n = n0 + j;
            if (n < C) {
                int64_t idx = static_cast<int64_t>(m) * C + n;
                if (group == 3) {
                    float vv = __half2float(v[idx]);
                    float vf = __half2float(v_first[idx]);
                    float gate = sigmoid_fast(__half2float(v0[n]) + sum);
                    y[idx] = __float2half_rn(fmaf(vf - vv, gate, vv));
                } else {
                    y[idx] = __float2half_rn(sum);
                }
            }
        }
    }
    if (group != 1) {
        return;
    }
    __syncthreads();
    int head_base = (n0 / HEAD_SIZE) * HEAD_SIZE;
    int64_t row_base = static_cast<int64_t>(m) * C;
    float norm_local = 0.0f;
    for (int q = threadIdx.x; q < HEAD_SIZE; q += THREADS) {
        int c = head_base + q;
        if (c < C) {
            float u = __half2float(k_raw[row_base + c]) * __half2float(k_k[c]);
            norm_local = fmaf(u, u, norm_local);
        }
    }
    float norm = block_sum_all<THREADS>(norm_local);
    float inv_norm = 1.0f / fmaxf(sqrtf(norm), KK_NORMALIZE_EPS);
    if (KK_LANES) {
        int j = threadIdx.x;
        if (j < OUT_TILE) {
            int n = n0 + j;
            if (n < C) {
                int64_t idx = row_base + n;
                half a_half = __float2half_rn(total[j]);
                float a_val = __half2float(a_half);
                float kr = __half2float(k_raw[idx]);
                float kk = kr * __half2float(k_k[n]) * inv_norm;
                float gate = sigmoid_fast(__half2float(a0[n]) + a_val);
                float ka = __half2float(k_a[n]);
                new_k[idx] = __float2half_rn(kr * fmaf(gate, ka, 1.0f - ka));
                neg_kk[idx] = __float2half_rn(-kk);
                kka[idx] = __float2half_rn(kk * gate);
            }
        }
        return;
    }
    if (threadIdx.x != 0) {
        return;
    }
#pragma unroll
    for (int j = 0; j < OUT_TILE; ++j) {
        int n = n0 + j;
        if (n >= C) {
            continue;
        }
        int64_t idx = row_base + n;
        half a_half = __float2half_rn(total[j]);
        float a_val = __half2float(a_half);
        float kr = __half2float(k_raw[idx]);
        float kk = kr * __half2float(k_k[n]) * inv_norm;
        float gate = sigmoid_fast(__half2float(a0[n]) + a_val);
        float ka = __half2float(k_a[n]);
        new_k[idx] = __float2half_rn(kr * fmaf(gate, ka, 1.0f - ka));
        neg_kk[idx] = __float2half_rn(-kk);
        kka[idx] = __float2half_rn(kk * gate);
    }
}


} // namespace



torch::Tensor rwkv7_mega_emb_ln0_bf16_to_f16_cuda(torch::Tensor emb, torch::Tensor weight, torch::Tensor bias, double eps) {
    auto out = torch::empty(emb.sizes(), emb.options().dtype(torch::kFloat16));
    int64_t V64 = emb.size(0);
    int64_t C64 = emb.size(1);
    TORCH_CHECK(V64 <= INT_MAX && C64 <= INT_MAX, "emb_ln0 shape too large");
    auto stream = at::cuda::getCurrentCUDAStream();
    emb_ln0_bf16_to_f16_kernel<<<static_cast<unsigned int>(V64), 256, 0, stream>>>(
        static_cast<int>(V64),
        static_cast<int>(C64),
        reinterpret_cast<const uint16_t*>(emb.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(weight.data_ptr<at::BFloat16>()),
        reinterpret_cast<const uint16_t*>(bias.data_ptr<at::BFloat16>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

void rwkv7_mega_emb_lookup_f16_into_cuda(torch::Tensor emb, torch::Tensor tokens, torch::Tensor out) {
    int64_t B = tokens.size(0);
    int64_t T = tokens.size(1);
    int64_t V = emb.size(0);
    int64_t C = emb.size(1);
    TORCH_CHECK(B * T <= INT_MAX && V <= INT_MAX && C <= INT_MAX, "emb lookup shape too large");
    auto stream = at::cuda::getCurrentCUDAStream();
    emb_lookup_f16_kernel<<<static_cast<unsigned int>(B * T), 256, 0, stream>>>(
        static_cast<int>(B * T),
        static_cast<int>(V),
        static_cast<int>(C),
        reinterpret_cast<const half*>(emb.data_ptr<at::Half>()),
        tokens.data_ptr<int64_t>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rwkv7_mega_ln_mix6_into_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor shift_state,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor x_r,
    torch::Tensor x_w,
    torch::Tensor x_k,
    torch::Tensor x_v,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor x_out,
    torch::Tensor out_r,
    torch::Tensor out_w,
    torch::Tensor out_k,
    torch::Tensor out_v,
    torch::Tensor out_a,
    torch::Tensor out_g,
    double eps,
    int64_t threads) {
    int64_t C = x.size(-1);
    int64_t rows = x.numel() / C;
    TORCH_CHECK(C == LN_SMALL_C && threads == LN_SMALL_THREADS, "260602 ln_mix6 requires C=4096 threads=1024");
    auto stream = at::cuda::getCurrentCUDAStream();
    ln_mix6_c4096_kernel<LN_SMALL_THREADS><<<static_cast<unsigned int>(rows), LN_SMALL_THREADS, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<half*>(shift_state.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x_r.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x_w.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x_k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x_v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x_a.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x_g.data_ptr<at::Half>()),
        reinterpret_cast<half*>(x_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_r.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_w.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_g.data_ptr<at::Half>()),
        rows,
        static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rwkv7_mega_rkv_lowrank_pre_executor_into_cuda(
    torch::Tensor xr,
    torch::Tensor xk,
    torch::Tensor xv,
    torch::Tensor wr,
    torch::Tensor wk,
    torch::Tensor wv,
    torch::Tensor yr,
    torch::Tensor yk,
    torch::Tensor yv,
    torch::Tensor xw,
    torch::Tensor xa,
    torch::Tensor xg,
    torch::Tensor xlr_v,
    torch::Tensor w1_t,
    torch::Tensor a1_t,
    torch::Tensor g1_t,
    torch::Tensor v1_t,
    torch::Tensor w2_t,
    torch::Tensor g2_t,
    torch::Tensor w1,
    torch::Tensor a1,
    torch::Tensor g1,
    torch::Tensor v1,
    torch::Tensor w,
    torch::Tensor g,
    int64_t blocks,
    int64_t threads,
    int64_t lowrank_worker_budget,
    int64_t rkv_out_tile,
    int64_t role_order) {
    (void)w2_t;
    (void)g2_t;
    (void)w;
    (void)g;
    TORCH_CHECK(threads == 128 && rkv_out_tile == 2 && role_order == 2,
                "260602 rkv executor requires threads=128 rkv_out_tile=2 role_order=2");
    int64_t C64 = xr.numel();
    int M = static_cast<int>(xw.numel() / C64);
    int C = static_cast<int>(C64);
    int Rw = static_cast<int>(w1_t.size(0));
    int Ra = static_cast<int>(a1_t.size(0));
    int Rg = static_cast<int>(g1_t.size(0));
    int Rv = static_cast<int>(v1_t.size(0));
    auto stream = at::cuda::getCurrentCUDAStream();
    rkv_lowrank_pre_executor_kernel<128, 2, true, 1, 2><<<static_cast<unsigned int>(blocks), 128, 0, stream>>>(
        reinterpret_cast<const half*>(xr.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xk.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xv.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(wr.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(wk.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(wv.data_ptr<at::Half>()),
        reinterpret_cast<half*>(yr.data_ptr<at::Half>()),
        reinterpret_cast<half*>(yk.data_ptr<at::Half>()),
        reinterpret_cast<half*>(yv.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xw.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xa.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xg.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(xlr_v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w1_t.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(a1_t.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(g1_t.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v1_t.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w2_t.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(g2_t.data_ptr<at::Half>()),
        reinterpret_cast<half*>(w1.data_ptr<at::Half>()),
        reinterpret_cast<half*>(a1.data_ptr<at::Half>()),
        reinterpret_cast<half*>(g1.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v1.data_ptr<at::Half>()),
        reinterpret_cast<half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<half*>(g.data_ptr<at::Half>()),
        M, C, Rw, Ra, Rg, Rv,
        static_cast<int>(lowrank_worker_budget),
        2);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rwkv7_mega_lowrank_rank_out4_kk_lanes_into_cuda(
    torch::Tensor w1,
    torch::Tensor a1,
    torch::Tensor g1,
    torch::Tensor v1,
    torch::Tensor w2_t,
    torch::Tensor a2_t,
    torch::Tensor g2_t,
    torch::Tensor v2_t,
    torch::Tensor v,
    torch::Tensor v_first,
    torch::Tensor v0,
    torch::Tensor k_raw,
    torch::Tensor k_k,
    torch::Tensor a0,
    torch::Tensor k_a,
    torch::Tensor w,
    torch::Tensor a,
    torch::Tensor g,
    torch::Tensor v_out,
    torch::Tensor new_k,
    torch::Tensor neg_kk,
    torch::Tensor kka) {
    int M = static_cast<int>(w1.size(0));
    int Rw = static_cast<int>(w1.size(1));
    int Ra = static_cast<int>(a1.size(1));
    int Rg = static_cast<int>(g1.size(1));
    int Rv = static_cast<int>(v1.size(1));
    int C = static_cast<int>(w2_t.size(0));
    auto stream = at::cuda::getCurrentCUDAStream();
    lowrank_rank_out4_kk_kernel<128, 4, true><<<dim3(static_cast<unsigned int>((C + 3) / 4), static_cast<unsigned int>(M), 4), 128, 0, stream>>>(
        reinterpret_cast<const half*>(w1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(a1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(g1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v1.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w2_t.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(a2_t.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(g2_t.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v2_t.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v_first.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v0.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_raw.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(a0.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k_a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<half*>(a.data_ptr<at::Half>()),
        reinterpret_cast<half*>(g.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(new_k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(neg_kk.data_ptr<at::Half>()),
        reinterpret_cast<half*>(kka.data_ptr<at::Half>()),
        M, C, Rw, Ra, Rg, Rv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rwkv7_mega_lnx_rkvres_xg_into_cuda(
    torch::Tensor x,
    torch::Tensor r,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g,
    torch::Tensor out,
    int64_t H64) {
    int64_t bth_size = x.numel() / HEAD_SIZE;
    auto stream = at::cuda::getCurrentCUDAStream();
    lnx_rkvres_xg_kernel<<<static_cast<unsigned int>(bth_size), HEAD_SIZE, 0, stream>>>(
        static_cast<int>(H64),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(r.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(r_k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(g.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        bth_size);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rwkv7_mega_row1_linear_exact4_into_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor y) {
    int64_t K64 = x.size(0);
    int64_t N64 = w.size(0);
    TORCH_CHECK(K64 <= INT_MAX && N64 <= INT_MAX, "row1 exact4 shape too large");
    auto stream = at::cuda::getCurrentCUDAStream();
    row1_linear_exact4_kernel<128, 2><<<static_cast<unsigned int>(N64 / 2), 128, 0, stream>>>(
        static_cast<int>(K64),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rwkv7_mega_add_ln_cmix_mix_into_cuda(
    torch::Tensor x,
    torch::Tensor residual,
    torch::Tensor shift_state,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor x_k,
    torch::Tensor x_out,
    torch::Tensor mixed,
    double eps,
    int64_t threads) {
    int64_t C = x.size(-1);
    int64_t rows = x.numel() / C;
    TORCH_CHECK(C == LN_SMALL_C && threads == LN_SMALL_THREADS, "260602 add_ln_cmix_mix requires C=4096 threads=1024");
    auto stream = at::cuda::getCurrentCUDAStream();
    add_ln_cmix_mix_c4096_kernel<LN_SMALL_THREADS><<<static_cast<unsigned int>(rows), LN_SMALL_THREADS, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<half*>(shift_state.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(x_k.data_ptr<at::Half>()),
        reinterpret_cast<half*>(x_out.data_ptr<at::Half>()),
        reinterpret_cast<half*>(mixed.data_ptr<at::Half>()),
        rows,
        static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rwkv7_mega_row1_linear_exact4_vec4_threads_tile_into_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t threads, int64_t out_tile) {
    int64_t K64 = x.size(0);
    int64_t N64 = w.size(0);
    TORCH_CHECK(K64 <= INT_MAX && N64 <= INT_MAX, "row1 exact4 vec4 shape too large");
    TORCH_CHECK((K64 % 4) == 0 && threads == 128 && out_tile == 2, "260602 vec4 key requires K%4=0 threads=128 out_tile=2");
    auto stream = at::cuda::getCurrentCUDAStream();
    row1_linear_exact4_vec4_kernel<128, 2><<<static_cast<unsigned int>(N64 / 2), 128, 0, stream>>>(
        static_cast<int>(K64),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(w.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rwkv7_mega_cmix_sparse_down_relu_one_vtile_hfma2_split2_into_cuda(torch::Tensor preact, torch::Tensor value_weight, torch::Tensor out) {
    int64_t F64 = preact.size(0);
    int64_t C64 = value_weight.size(1);
    auto stream = at::cuda::getCurrentCUDAStream();
    zero_vec4_kernel<<<static_cast<unsigned int>(((C64 / 8) + 255) / 256), 256, 0, stream>>>(
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        C64 / 8);
    cmix_sparse_down_relu_one_vtile_hfma2_split2_kernel<<<dim3(static_cast<unsigned int>(F64 / FFN_TILE), static_cast<unsigned int>(C64 / (2 * FFN_SPMV_THREADS)), 1), FFN_SPMV_THREADS, 0, stream>>>(
        static_cast<int>(C64),
        reinterpret_cast<const half*>(preact.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(value_weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void rwkv7_mega_add_last_layer_norm_f16_into_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, double eps) {
    int64_t B64 = x.size(0);
    int64_t T64 = x.size(1);
    int64_t C64 = x.size(2);
    TORCH_CHECK(B64 <= INT_MAX && T64 <= INT_MAX && C64 <= INT_MAX, "add_last_layer_norm shape too large");
    auto stream = at::cuda::getCurrentCUDAStream();
    add_last_layer_norm_f16_kernel<256><<<static_cast<unsigned int>(B64), 256, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(residual.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(bias.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        static_cast<int>(B64),
        static_cast<int>(T64),
        static_cast<int>(C64),
        static_cast<float>(eps));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
