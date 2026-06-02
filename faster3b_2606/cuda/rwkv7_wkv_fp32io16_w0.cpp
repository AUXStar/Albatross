#include <torch/extension.h>

void wkv_fp32io16_w0_cuda(
    int B,
    int T,
    int C,
    int H,
    int mode,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y);
torch::Tensor wkv_fp32io16_w0_kk_t1_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k_raw,
    torch::Tensor k_k,
    torch::Tensor v,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor y);
torch::Tensor wkv_fp32io16_w0_kk_t1_row_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k_raw,
    torch::Tensor k_k,
    torch::Tensor v,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor y);
torch::Tensor wkv_fp32io16_w0_kk_t1_tile_cuda(
    int B,
    int T,
    int C,
    int H,
    int row_tile,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k_raw,
    torch::Tensor k_k,
    torch::Tensor v,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor y);
void wkv_fp32io16_w0_kk_pc_t1_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k_raw,
    torch::Tensor k_k,
    torch::Tensor v,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor y,
    torch::Tensor new_k,
    torch::Tensor neg_kk,
    torch::Tensor kka,
    torch::Tensor ready);
torch::Tensor wkv_fp32io16_w0_lnx_t1_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor neg_kk,
    torch::Tensor kka,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g);
torch::Tensor wkv_fp32io16_w0_kk_lnx_t1_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k_raw,
    torch::Tensor k_k,
    torch::Tensor v,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g);
void wkv_fp32io16_w0_kk_lnx_t1_into_cuda(
    int B,
    int T,
    int C,
    int H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k_raw,
    torch::Tensor k_k,
    torch::Tensor v,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g,
    torch::Tensor out);

namespace {

#ifdef _IO_FP16_
constexpr auto IO_DTYPE = torch::kFloat16;
constexpr const char* IO_DTYPE_NAME = "fp16";
#else
constexpr auto IO_DTYPE = torch::kFloat32;
constexpr const char* IO_DTYPE_NAME = "fp32";
#endif

void check_float_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat32, name, " must be fp32");
}

void check_io_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == IO_DTYPE, name, " must be ", IO_DTYPE_NAME);
}

void check_int_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kInt32, name, " must be int32");
}

void check_inputs(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y) {
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  check_float_cuda_contig(state, "state");
  check_io_cuda_contig(r, "r");
  check_io_cuda_contig(w, "w");
  check_io_cuda_contig(w0, "w0");
  check_io_cuda_contig(k, "k");
  check_io_cuda_contig(v, "v");
  check_io_cuda_contig(a, "a");
  check_io_cuda_contig(b, "b");
  check_io_cuda_contig(y, "y");
  TORCH_CHECK(state.dim() == 4 && state.size(0) == B && state.size(1) == H && state.size(2) == 64 && state.size(3) == 64,
              "state must have shape [B,H,64,64]");
  TORCH_CHECK(w0.dim() == 1 && w0.size(0) == C, "w0 must have shape [C]");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k.sizes() && r.sizes() == v.sizes() &&
              r.sizes() == a.sizes() && r.sizes() == b.sizes() && r.sizes() == y.sizes(),
              "r,w,k,v,a,b,y shape mismatch");
  TORCH_CHECK(r.dim() == 3 && r.size(0) == B && r.size(1) == T && r.size(2) == C,
              "r must have shape [B,T,C]");
}

}  // namespace

void forward_w0(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y) {
  check_inputs(B, T, C, H, state, r, w, w0, k, v, a, b, y);
  wkv_fp32io16_w0_cuda(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(C),
      static_cast<int>(H),
      0,
      state,
      r,
      w,
      w0,
      k,
      v,
      a,
      b,
      y);
}

void forward_seq_w0(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y) {
  check_inputs(B, T, C, H, state, r, w, w0, k, v, a, b, y);
  wkv_fp32io16_w0_cuda(static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H), 1, state, r, w, w0, k, v, a, b, y);
}

void forward_small_w0(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y) {
  check_inputs(B, T, C, H, state, r, w, w0, k, v, a, b, y);
  wkv_fp32io16_w0_cuda(static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H), 2, state, r, w, w0, k, v, a, b, y);
}

void forward_block_w0(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor a,
    torch::Tensor b,
    torch::Tensor y) {
  check_inputs(B, T, C, H, state, r, w, w0, k, v, a, b, y);
  wkv_fp32io16_w0_cuda(static_cast<int>(B), static_cast<int>(T), static_cast<int>(C), static_cast<int>(H), 3, state, r, w, w0, k, v, a, b, y);
}

torch::Tensor forward_w0_kk_t1(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k_raw,
    torch::Tensor k_k,
    torch::Tensor v,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor y) {
  TORCH_CHECK(T == 1, "forward_w0_kk_t1 only supports T=1");
  check_float_cuda_contig(state, "state");
  check_io_cuda_contig(r, "r");
  check_io_cuda_contig(w, "w");
  check_io_cuda_contig(w0, "w0");
  check_io_cuda_contig(k_raw, "k_raw");
  check_io_cuda_contig(k_k, "k_k");
  check_io_cuda_contig(v, "v");
  check_io_cuda_contig(a0, "a0");
  check_io_cuda_contig(a12, "a12");
  check_io_cuda_contig(k_a, "k_a");
  check_io_cuda_contig(y, "y");
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  TORCH_CHECK(state.dim() == 4 && state.size(0) == B && state.size(1) == H && state.size(2) == 64 && state.size(3) == 64,
              "state must have shape [B,H,64,64]");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k_raw.sizes() && r.sizes() == v.sizes() &&
              r.sizes() == a12.sizes() && r.sizes() == y.sizes(), "r,w,k_raw,v,a12,y shape mismatch");
  TORCH_CHECK(r.dim() == 3 && r.size(0) == B && r.size(1) == T && r.size(2) == C, "r must have shape [B,T,C]");
  TORCH_CHECK(w0.dim() == 1 && w0.size(0) == C, "w0 must have shape [C]");
  TORCH_CHECK(k_k.dim() == 1 && k_k.size(0) == C, "k_k must have shape [C]");
  TORCH_CHECK(a0.dim() == 1 && a0.size(0) == C, "a0 must have shape [C]");
  TORCH_CHECK(k_a.dim() == 1 && k_a.size(0) == C, "k_a must have shape [C]");
  return wkv_fp32io16_w0_kk_t1_cuda(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(C),
      static_cast<int>(H),
      state,
      r,
      w,
      w0,
      k_raw,
      k_k,
      v,
      a0,
      a12,
      k_a,
      y);
}

torch::Tensor forward_w0_kk_t1_row(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k_raw,
    torch::Tensor k_k,
    torch::Tensor v,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor y) {
  TORCH_CHECK(T == 1, "forward_w0_kk_t1_row only supports T=1");
  check_float_cuda_contig(state, "state");
  check_io_cuda_contig(r, "r");
  check_io_cuda_contig(w, "w");
  check_io_cuda_contig(w0, "w0");
  check_io_cuda_contig(k_raw, "k_raw");
  check_io_cuda_contig(k_k, "k_k");
  check_io_cuda_contig(v, "v");
  check_io_cuda_contig(a0, "a0");
  check_io_cuda_contig(a12, "a12");
  check_io_cuda_contig(k_a, "k_a");
  check_io_cuda_contig(y, "y");
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  return wkv_fp32io16_w0_kk_t1_row_cuda(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(C),
      static_cast<int>(H),
      state,
      r,
      w,
      w0,
      k_raw,
      k_k,
      v,
      a0,
      a12,
      k_a,
      y);
}

torch::Tensor forward_w0_kk_t1_tile(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    int64_t row_tile,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k_raw,
    torch::Tensor k_k,
    torch::Tensor v,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor y) {
  TORCH_CHECK(T == 1, "forward_w0_kk_t1_tile only supports T=1");
  TORCH_CHECK(row_tile == 4 || row_tile == 8, "row_tile must be 4 or 8");
  check_float_cuda_contig(state, "state");
  check_io_cuda_contig(r, "r");
  check_io_cuda_contig(w, "w");
  check_io_cuda_contig(w0, "w0");
  check_io_cuda_contig(k_raw, "k_raw");
  check_io_cuda_contig(k_k, "k_k");
  check_io_cuda_contig(v, "v");
  check_io_cuda_contig(a0, "a0");
  check_io_cuda_contig(a12, "a12");
  check_io_cuda_contig(k_a, "k_a");
  check_io_cuda_contig(y, "y");
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k_raw.sizes() && r.sizes() == v.sizes() &&
              r.sizes() == a12.sizes() && r.sizes() == y.sizes(), "r,w,k_raw,v,a12,y shape mismatch");
  return wkv_fp32io16_w0_kk_t1_tile_cuda(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(C),
      static_cast<int>(H),
      static_cast<int>(row_tile),
      state,
      r,
      w,
      w0,
      k_raw,
      k_k,
      v,
      a0,
      a12,
      k_a,
      y);
}

void forward_w0_kk_pc_t1(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k_raw,
    torch::Tensor k_k,
    torch::Tensor v,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor y,
    torch::Tensor new_k,
    torch::Tensor neg_kk,
    torch::Tensor kka,
    torch::Tensor ready) {
  TORCH_CHECK(T == 1, "forward_w0_kk_pc_t1 only supports T=1");
  check_float_cuda_contig(state, "state");
  check_io_cuda_contig(r, "r");
  check_io_cuda_contig(w, "w");
  check_io_cuda_contig(w0, "w0");
  check_io_cuda_contig(k_raw, "k_raw");
  check_io_cuda_contig(k_k, "k_k");
  check_io_cuda_contig(v, "v");
  check_io_cuda_contig(a0, "a0");
  check_io_cuda_contig(a12, "a12");
  check_io_cuda_contig(k_a, "k_a");
  check_io_cuda_contig(y, "y");
  check_io_cuda_contig(new_k, "new_k");
  check_io_cuda_contig(neg_kk, "neg_kk");
  check_io_cuda_contig(kka, "kka");
  check_int_cuda_contig(ready, "ready");
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k_raw.sizes() && r.sizes() == v.sizes() &&
              r.sizes() == a12.sizes() && r.sizes() == y.sizes() && r.sizes() == new_k.sizes() &&
              r.sizes() == neg_kk.sizes() && r.sizes() == kka.sizes(), "B1T1 tensor shape mismatch");
  TORCH_CHECK(ready.dim() == 1 && ready.size(0) >= B * H, "ready must have at least B*H int32 entries");
  wkv_fp32io16_w0_kk_pc_t1_cuda(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(C),
      static_cast<int>(H),
      state,
      r,
      w,
      w0,
      k_raw,
      k_k,
      v,
      a0,
      a12,
      k_a,
      y,
      new_k,
      neg_kk,
      kka,
      ready);
}

torch::Tensor forward_w0_lnx_t1(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k,
    torch::Tensor v,
    torch::Tensor neg_kk,
    torch::Tensor kka,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g) {
  TORCH_CHECK(T == 1, "forward_w0_lnx_t1 only supports T=1");
  check_float_cuda_contig(state, "state");
  check_io_cuda_contig(r, "r");
  check_io_cuda_contig(w, "w");
  check_io_cuda_contig(w0, "w0");
  check_io_cuda_contig(k, "k");
  check_io_cuda_contig(v, "v");
  check_io_cuda_contig(neg_kk, "neg_kk");
  check_io_cuda_contig(kka, "kka");
  check_io_cuda_contig(r_k, "r_k");
  check_io_cuda_contig(weight, "weight");
  check_io_cuda_contig(bias, "bias");
  check_io_cuda_contig(g, "g");
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k.sizes() && r.sizes() == v.sizes() &&
              r.sizes() == neg_kk.sizes() && r.sizes() == kka.sizes() && r.sizes() == g.sizes(),
              "r,w,k,v,neg_kk,kka,g shape mismatch");
  TORCH_CHECK(r.dim() == 3 && r.size(0) == B && r.size(1) == T && r.size(2) == C, "r must have shape [B,T,C]");
  TORCH_CHECK(w0.dim() == 1 && w0.size(0) == C, "w0 must have shape [C]");
  TORCH_CHECK(r_k.dim() == 1 && r_k.size(0) == C, "r_k must have shape [C]");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == C, "weight must have shape [C]");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C, "bias must have shape [C]");
  return wkv_fp32io16_w0_lnx_t1_cuda(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(C),
      static_cast<int>(H),
      state,
      r,
      w,
      w0,
      k,
      v,
      neg_kk,
      kka,
      r_k,
      weight,
      bias,
      g);
}

torch::Tensor forward_w0_kk_lnx_t1(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k_raw,
    torch::Tensor k_k,
    torch::Tensor v,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g) {
  TORCH_CHECK(T == 1, "forward_w0_kk_lnx_t1 only supports T=1");
  check_float_cuda_contig(state, "state");
  check_io_cuda_contig(r, "r");
  check_io_cuda_contig(w, "w");
  check_io_cuda_contig(w0, "w0");
  check_io_cuda_contig(k_raw, "k_raw");
  check_io_cuda_contig(k_k, "k_k");
  check_io_cuda_contig(v, "v");
  check_io_cuda_contig(a0, "a0");
  check_io_cuda_contig(a12, "a12");
  check_io_cuda_contig(k_a, "k_a");
  check_io_cuda_contig(r_k, "r_k");
  check_io_cuda_contig(weight, "weight");
  check_io_cuda_contig(bias, "bias");
  check_io_cuda_contig(g, "g");
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k_raw.sizes() && r.sizes() == v.sizes() &&
              r.sizes() == a12.sizes() && r.sizes() == g.sizes(), "r,w,k_raw,v,a12,g shape mismatch");
  TORCH_CHECK(r.dim() == 3 && r.size(0) == B && r.size(1) == T && r.size(2) == C, "r must have shape [B,T,C]");
  TORCH_CHECK(w0.dim() == 1 && w0.size(0) == C, "w0 must have shape [C]");
  TORCH_CHECK(k_k.dim() == 1 && k_k.size(0) == C, "k_k must have shape [C]");
  TORCH_CHECK(a0.dim() == 1 && a0.size(0) == C, "a0 must have shape [C]");
  TORCH_CHECK(k_a.dim() == 1 && k_a.size(0) == C, "k_a must have shape [C]");
  TORCH_CHECK(r_k.dim() == 1 && r_k.size(0) == C, "r_k must have shape [C]");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == C, "weight must have shape [C]");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C, "bias must have shape [C]");
  return wkv_fp32io16_w0_kk_lnx_t1_cuda(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(C),
      static_cast<int>(H),
      state,
      r,
      w,
      w0,
      k_raw,
      k_k,
      v,
      a0,
      a12,
      k_a,
      r_k,
      weight,
      bias,
      g);
}

void forward_w0_kk_lnx_t1_into(
    int64_t B,
    int64_t T,
    int64_t C,
    int64_t H,
    torch::Tensor state,
    torch::Tensor r,
    torch::Tensor w,
    torch::Tensor w0,
    torch::Tensor k_raw,
    torch::Tensor k_k,
    torch::Tensor v,
    torch::Tensor a0,
    torch::Tensor a12,
    torch::Tensor k_a,
    torch::Tensor r_k,
    torch::Tensor weight,
    torch::Tensor bias,
    torch::Tensor g,
    torch::Tensor out) {
  TORCH_CHECK(T == 1, "forward_w0_kk_lnx_t1_into only supports T=1");
  check_float_cuda_contig(state, "state");
  check_io_cuda_contig(r, "r");
  check_io_cuda_contig(w, "w");
  check_io_cuda_contig(w0, "w0");
  check_io_cuda_contig(k_raw, "k_raw");
  check_io_cuda_contig(k_k, "k_k");
  check_io_cuda_contig(v, "v");
  check_io_cuda_contig(a0, "a0");
  check_io_cuda_contig(a12, "a12");
  check_io_cuda_contig(k_a, "k_a");
  check_io_cuda_contig(r_k, "r_k");
  check_io_cuda_contig(weight, "weight");
  check_io_cuda_contig(bias, "bias");
  check_io_cuda_contig(g, "g");
  check_io_cuda_contig(out, "out");
  TORCH_CHECK(C == H * 64, "only head size 64 is supported");
  TORCH_CHECK(r.sizes() == w.sizes() && r.sizes() == k_raw.sizes() && r.sizes() == v.sizes() &&
              r.sizes() == a12.sizes() && r.sizes() == g.sizes() && r.sizes() == out.sizes(),
              "r,w,k_raw,v,a12,g,out shape mismatch");
  TORCH_CHECK(r.dim() == 3 && r.size(0) == B && r.size(1) == T && r.size(2) == C, "r must have shape [B,T,C]");
  TORCH_CHECK(w0.dim() == 1 && w0.size(0) == C, "w0 must have shape [C]");
  TORCH_CHECK(k_k.dim() == 1 && k_k.size(0) == C, "k_k must have shape [C]");
  TORCH_CHECK(a0.dim() == 1 && a0.size(0) == C, "a0 must have shape [C]");
  TORCH_CHECK(k_a.dim() == 1 && k_a.size(0) == C, "k_a must have shape [C]");
  TORCH_CHECK(r_k.dim() == 1 && r_k.size(0) == C, "r_k must have shape [C]");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == C, "weight must have shape [C]");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == C, "bias must have shape [C]");
  wkv_fp32io16_w0_kk_lnx_t1_into_cuda(
      static_cast<int>(B),
      static_cast<int>(T),
      static_cast<int>(C),
      static_cast<int>(H),
      state,
      r,
      w,
      w0,
      k_raw,
      k_k,
      v,
      a0,
      a12,
      k_a,
      r_k,
      weight,
      bias,
      g,
      out);
}

TORCH_LIBRARY(rwkv7_wkv_fp32io16_w0, m) {
  m.def("forward_w0(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.def("forward_seq_w0(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.def("forward_small_w0(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.def("forward_block_w0(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k, Tensor v, Tensor a, Tensor b, Tensor(a!) y) -> ()");
  m.def("forward_w0_kk_t1(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k_raw, Tensor k_k, Tensor v, Tensor a0, Tensor a12, Tensor k_a, Tensor(a!) y) -> Tensor");
  m.def("forward_w0_kk_t1_row(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k_raw, Tensor k_k, Tensor v, Tensor a0, Tensor a12, Tensor k_a, Tensor(a!) y) -> Tensor");
  m.def("forward_w0_kk_t1_tile(int B, int T, int C, int H, int row_tile, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k_raw, Tensor k_k, Tensor v, Tensor a0, Tensor a12, Tensor k_a, Tensor(a!) y) -> Tensor");
  m.def("forward_w0_kk_pc_t1(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k_raw, Tensor k_k, Tensor v, Tensor a0, Tensor a12, Tensor k_a, Tensor(a!) y, Tensor(a!) new_k, Tensor(a!) neg_kk, Tensor(a!) kka, Tensor(a!) ready) -> ()");
  m.def("forward_w0_lnx_t1(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k, Tensor v, Tensor neg_kk, Tensor kka, Tensor r_k, Tensor weight, Tensor bias, Tensor g) -> Tensor");
  m.def("forward_w0_kk_lnx_t1(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k_raw, Tensor k_k, Tensor v, Tensor a0, Tensor a12, Tensor k_a, Tensor r_k, Tensor weight, Tensor bias, Tensor g) -> Tensor");
  m.def("forward_w0_kk_lnx_t1_into(int B, int T, int C, int H, Tensor(a!) state, Tensor r, Tensor w, Tensor w0, Tensor k_raw, Tensor k_k, Tensor v, Tensor a0, Tensor a12, Tensor k_a, Tensor r_k, Tensor weight, Tensor bias, Tensor g, Tensor(b!) out) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_wkv_fp32io16_w0, CUDA, m) {
  m.impl("forward_w0", &forward_w0);
  m.impl("forward_seq_w0", &forward_seq_w0);
  m.impl("forward_small_w0", &forward_small_w0);
  m.impl("forward_block_w0", &forward_block_w0);
  m.impl("forward_w0_kk_t1", &forward_w0_kk_t1);
  m.impl("forward_w0_kk_t1_row", &forward_w0_kk_t1_row);
  m.impl("forward_w0_kk_t1_tile", &forward_w0_kk_t1_tile);
  m.impl("forward_w0_kk_pc_t1", &forward_w0_kk_pc_t1);
  m.impl("forward_w0_lnx_t1", &forward_w0_lnx_t1);
  m.impl("forward_w0_kk_lnx_t1", &forward_w0_kk_lnx_t1);
  m.impl("forward_w0_kk_lnx_t1_into", &forward_w0_kk_lnx_t1_into);
}
