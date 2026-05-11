#include <torch/extension.h>
#include <vector>

torch::Tensor identity_cuda(torch::Tensor x);
torch::Tensor layer_norm_f16_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);
torch::Tensor layer_norm_f16_small_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);
torch::Tensor layer_norm_f16_small512_cuda(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps);
torch::Tensor linear_f16_cuda(torch::Tensor x, torch::Tensor weight);
torch::Tensor linear_f16_lt_cuda(torch::Tensor x, torch::Tensor weight);
torch::Tensor linear_f16_m1_splitk_cuda(torch::Tensor x, torch::Tensor weight);
torch::Tensor linear_mix_f16_m1_splitk_cuda(torch::Tensor x, torch::Tensor shift, torch::Tensor mix, torch::Tensor weight);
torch::Tensor linear_f16_m1_splitk_cfg_cuda(torch::Tensor x, torch::Tensor weight, int64_t chunk_k);
torch::Tensor linear_f16_m1_splitk_tile_cuda(torch::Tensor x, torch::Tensor weight, int64_t chunk_k, int64_t tile_cols);
torch::Tensor linear_f16_rows_splitk_cuda(torch::Tensor x, torch::Tensor weight, int64_t chunk_k);
torch::Tensor linear_t_f16_cuda(torch::Tensor x, torch::Tensor weight_t);
torch::Tensor linear_t_act_f16_cuda(torch::Tensor x, torch::Tensor weight_t, int64_t act);
torch::Tensor linear_t_vres_f16_cuda(torch::Tensor x, torch::Tensor weight_t, torch::Tensor v, torch::Tensor v_first, torch::Tensor v0);
std::vector<torch::Tensor> linear_wag_rank_in_f16_cuda(
    torch::Tensor xw, torch::Tensor xa, torch::Tensor xg,
    torch::Tensor w1_t, torch::Tensor a1_t, torch::Tensor g1_t);
std::vector<torch::Tensor> linear_wag_rank_in_mix_f16_cuda(
    torch::Tensor x, torch::Tensor shift, torch::Tensor x_w, torch::Tensor x_a, torch::Tensor x_g,
    torch::Tensor w1_t, torch::Tensor a1_t, torch::Tensor g1_t);
std::vector<torch::Tensor> linear_wagv_rank_in_f16_cuda(
    torch::Tensor xw, torch::Tensor xa, torch::Tensor xg, torch::Tensor xv,
    torch::Tensor w1_t, torch::Tensor a1_t, torch::Tensor g1_t, torch::Tensor v1_t);
std::vector<torch::Tensor> linear_wagv_rank_in_mix_f16_cuda(
    torch::Tensor x, torch::Tensor shift, torch::Tensor x_w, torch::Tensor x_a, torch::Tensor x_g, torch::Tensor x_v,
    torch::Tensor w1_t, torch::Tensor a1_t, torch::Tensor g1_t, torch::Tensor v1_t);
std::vector<torch::Tensor> linear_wag_rank_out_f16_cuda(
    torch::Tensor w1, torch::Tensor a1, torch::Tensor g1,
    torch::Tensor w2_t, torch::Tensor a2_t, torch::Tensor g2_t);
std::vector<torch::Tensor> linear_wagv_rank_out_f16_cuda(
    torch::Tensor w1, torch::Tensor a1, torch::Tensor g1, torch::Tensor v1,
    torch::Tensor w2_t, torch::Tensor a2_t, torch::Tensor g2_t, torch::Tensor v2_t,
    torch::Tensor v, torch::Tensor v_first, torch::Tensor v0);
torch::Tensor add_f16_cuda(torch::Tensor x, torch::Tensor y);
std::vector<torch::Tensor> add_layer_norm_f16_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, torch::Tensor bias, double eps);
torch::Tensor add_last_layer_norm_f16_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, torch::Tensor bias, double eps);
std::vector<torch::Tensor> add_layer_norm_cmix_mix_f16_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias, torch::Tensor x_k, double eps);
std::vector<torch::Tensor> add_layer_norm_cmix_mix_f16_cfg_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias, torch::Tensor x_k, double eps, int threads);
std::vector<torch::Tensor> add_layer_norm_tmix_mix6_f16_cuda(
    torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias,
    torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a, torch::Tensor x_g,
    double eps);
void advance_i32_cuda(torch::Tensor x, int64_t amount);
void copy_m1_to_shift_f16_cuda(torch::Tensor x, torch::Tensor shift);

namespace {

void check_half_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kFloat16, name, " must be fp16");
}

void check_i32_cuda_contig(const torch::Tensor& x, const char* name) {
  TORCH_CHECK(x.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(x.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(x.scalar_type() == torch::kInt32, name, " must be int32");
}

torch::Tensor identity(torch::Tensor x) {
  check_half_cuda_contig(x, "x");
  return identity_cuda(x);
}

torch::Tensor layer_norm_f16(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim");
  const int64_t c = x.size(-1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(c > 0 && c <= 8192, "unsupported C");
  return layer_norm_f16_cuda(x, weight, bias, eps);
}

torch::Tensor layer_norm_f16_small(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim");
  const int64_t c = x.size(-1);
  TORCH_CHECK(c == 4096, "small LN currently requires C=4096");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  return layer_norm_f16_small_cuda(x, weight, bias, eps);
}

torch::Tensor layer_norm_f16_small512(torch::Tensor x, torch::Tensor weight, torch::Tensor bias, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim");
  const int64_t c = x.size(-1);
  TORCH_CHECK(c == 4096, "small512 LN currently requires C=4096");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  return layer_norm_f16_small512_cuda(x, weight, bias, eps);
}

torch::Tensor linear_f16(torch::Tensor x, torch::Tensor weight) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_f16 shape mismatch");
  return linear_f16_cuda(x, weight);
}

torch::Tensor linear_f16_lt(torch::Tensor x, torch::Tensor weight) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_f16_lt shape mismatch");
  return linear_f16_lt_cuda(x, weight);
}

torch::Tensor linear_f16_m1_splitk(torch::Tensor x, torch::Tensor weight) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_f16_m1_splitk shape mismatch");
  TORCH_CHECK(x.numel() == x.size(-1), "linear_f16_m1_splitk requires M=1");
  return linear_f16_m1_splitk_cuda(x, weight);
}

torch::Tensor linear_mix_f16_m1_splitk(torch::Tensor x, torch::Tensor shift, torch::Tensor mix, torch::Tensor weight) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(shift, "shift");
  check_half_cuda_contig(mix, "mix");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_mix_f16_m1_splitk shape mismatch");
  TORCH_CHECK(x.numel() == x.size(-1), "linear_mix_f16_m1_splitk requires M=1");
  TORCH_CHECK(shift.numel() == x.numel() && mix.numel() == x.numel(), "linear_mix_f16_m1_splitk mix shape mismatch");
  return linear_mix_f16_m1_splitk_cuda(x, shift, mix, weight);
}

torch::Tensor linear_f16_m1_splitk_cfg(torch::Tensor x, torch::Tensor weight, int64_t chunk_k) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_f16_m1_splitk_cfg shape mismatch");
  TORCH_CHECK(x.numel() == x.size(-1), "linear_f16_m1_splitk_cfg requires M=1");
  return linear_f16_m1_splitk_cfg_cuda(x, weight, chunk_k);
}

torch::Tensor linear_f16_m1_splitk_tile(torch::Tensor x, torch::Tensor weight, int64_t chunk_k, int64_t tile_cols) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_f16_m1_splitk_tile shape mismatch");
  TORCH_CHECK(x.numel() == x.size(-1), "linear_f16_m1_splitk_tile requires M=1");
  return linear_f16_m1_splitk_tile_cuda(x, weight, chunk_k, tile_cols);
}

torch::Tensor linear_f16_rows_splitk(torch::Tensor x, torch::Tensor weight, int64_t chunk_k) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight, "weight");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight.dim() == 2, "weight must have shape [K, N]");
  TORCH_CHECK(x.size(-1) == weight.size(0), "linear_f16_rows_splitk shape mismatch");
  return linear_f16_rows_splitk_cuda(x, weight, chunk_k);
}

torch::Tensor linear_t_f16(torch::Tensor x, torch::Tensor weight_t) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_t, "weight_t");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_t.dim() == 2, "weight_t must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_t.size(1), "linear_t_f16 shape mismatch");
  return linear_t_f16_cuda(x, weight_t);
}

torch::Tensor linear_t_act_f16(torch::Tensor x, torch::Tensor weight_t, int64_t act) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_t, "weight_t");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_t.dim() == 2, "weight_t must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_t.size(1), "linear_t_act_f16 shape mismatch");
  TORCH_CHECK(act == 1 || act == 2, "act must be 1=tanh or 2=sigmoid");
  return linear_t_act_f16_cuda(x, weight_t, act);
}

torch::Tensor linear_t_vres_f16(torch::Tensor x, torch::Tensor weight_t, torch::Tensor v, torch::Tensor v_first, torch::Tensor v0) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(weight_t, "weight_t");
  check_half_cuda_contig(v, "v");
  check_half_cuda_contig(v_first, "v_first");
  check_half_cuda_contig(v0, "v0");
  TORCH_CHECK(x.dim() >= 2, "x must have at least 2 dims");
  TORCH_CHECK(weight_t.dim() == 2, "weight_t must have shape [N, K]");
  TORCH_CHECK(x.size(-1) == weight_t.size(1), "linear_t_vres_f16 shape mismatch");
  TORCH_CHECK(v.sizes() == v_first.sizes(), "v/v_first shape mismatch");
  TORCH_CHECK(v.dim() >= 2 && v.size(-1) == weight_t.size(0), "v shape mismatch");
  TORCH_CHECK(v0.dim() == 1 && v0.size(0) == weight_t.size(0), "v0 shape mismatch");
  return linear_t_vres_f16_cuda(x, weight_t, v, v_first, v0);
}

std::vector<torch::Tensor> linear_wag_rank_in_f16(
    torch::Tensor xw,
    torch::Tensor xa,
    torch::Tensor xg,
    torch::Tensor w1_t,
    torch::Tensor a1_t,
    torch::Tensor g1_t) {
  check_half_cuda_contig(xw, "xw");
  check_half_cuda_contig(xa, "xa");
  check_half_cuda_contig(xg, "xg");
  check_half_cuda_contig(w1_t, "w1_t");
  check_half_cuda_contig(a1_t, "a1_t");
  check_half_cuda_contig(g1_t, "g1_t");
  TORCH_CHECK(xw.sizes() == xa.sizes() && xw.sizes() == xg.sizes(), "xw/xa/xg shape mismatch");
  TORCH_CHECK(w1_t.dim() == 2 && a1_t.dim() == 2 && g1_t.dim() == 2, "weight_t must be 2D");
  TORCH_CHECK(xw.size(-1) == w1_t.size(1) && xw.size(-1) == a1_t.size(1) && xw.size(-1) == g1_t.size(1),
              "rank-in K mismatch");
  return linear_wag_rank_in_f16_cuda(xw, xa, xg, w1_t, a1_t, g1_t);
}

std::vector<torch::Tensor> linear_wag_rank_in_mix_f16(
    torch::Tensor x,
    torch::Tensor shift,
    torch::Tensor x_w,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor w1_t,
    torch::Tensor a1_t,
    torch::Tensor g1_t) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(shift, "shift");
  check_half_cuda_contig(x_w, "x_w");
  check_half_cuda_contig(x_a, "x_a");
  check_half_cuda_contig(x_g, "x_g");
  check_half_cuda_contig(w1_t, "w1_t");
  check_half_cuda_contig(a1_t, "a1_t");
  check_half_cuda_contig(g1_t, "g1_t");
  TORCH_CHECK(x.numel() == x.size(-1), "linear_wag_rank_in_mix_f16 requires M=1");
  TORCH_CHECK(shift.numel() == x.numel() && x_w.numel() == x.numel() && x_a.numel() == x.numel() && x_g.numel() == x.numel(),
              "linear_wag_rank_in_mix_f16 input shape mismatch");
  TORCH_CHECK(w1_t.dim() == 2 && a1_t.dim() == 2 && g1_t.dim() == 2, "rank-in weights must be 2D");
  TORCH_CHECK(w1_t.size(1) == x.numel() && a1_t.size(1) == x.numel() && g1_t.size(1) == x.numel(),
              "linear_wag_rank_in_mix_f16 weight shape mismatch");
  return linear_wag_rank_in_mix_f16_cuda(x, shift, x_w, x_a, x_g, w1_t, a1_t, g1_t);
}

std::vector<torch::Tensor> linear_wagv_rank_in_f16(
    torch::Tensor xw,
    torch::Tensor xa,
    torch::Tensor xg,
    torch::Tensor xv,
    torch::Tensor w1_t,
    torch::Tensor a1_t,
    torch::Tensor g1_t,
    torch::Tensor v1_t) {
  check_half_cuda_contig(xw, "xw");
  check_half_cuda_contig(xa, "xa");
  check_half_cuda_contig(xg, "xg");
  check_half_cuda_contig(xv, "xv");
  check_half_cuda_contig(w1_t, "w1_t");
  check_half_cuda_contig(a1_t, "a1_t");
  check_half_cuda_contig(g1_t, "g1_t");
  check_half_cuda_contig(v1_t, "v1_t");
  TORCH_CHECK(xw.sizes() == xa.sizes() && xw.sizes() == xg.sizes() && xw.sizes() == xv.sizes(), "xw/xa/xg/xv shape mismatch");
  TORCH_CHECK(w1_t.dim() == 2 && a1_t.dim() == 2 && g1_t.dim() == 2 && v1_t.dim() == 2, "weight_t must be 2D");
  TORCH_CHECK(xw.size(-1) == w1_t.size(1) && xw.size(-1) == a1_t.size(1) &&
              xw.size(-1) == g1_t.size(1) && xw.size(-1) == v1_t.size(1), "rank-in K mismatch");
  return linear_wagv_rank_in_f16_cuda(xw, xa, xg, xv, w1_t, a1_t, g1_t, v1_t);
}

std::vector<torch::Tensor> linear_wagv_rank_in_mix_f16(
    torch::Tensor x,
    torch::Tensor shift,
    torch::Tensor x_w,
    torch::Tensor x_a,
    torch::Tensor x_g,
    torch::Tensor x_v,
    torch::Tensor w1_t,
    torch::Tensor a1_t,
    torch::Tensor g1_t,
    torch::Tensor v1_t) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(shift, "shift");
  check_half_cuda_contig(x_w, "x_w");
  check_half_cuda_contig(x_a, "x_a");
  check_half_cuda_contig(x_g, "x_g");
  check_half_cuda_contig(x_v, "x_v");
  check_half_cuda_contig(w1_t, "w1_t");
  check_half_cuda_contig(a1_t, "a1_t");
  check_half_cuda_contig(g1_t, "g1_t");
  check_half_cuda_contig(v1_t, "v1_t");
  TORCH_CHECK(x.numel() == x.size(-1), "linear_wagv_rank_in_mix_f16 requires M=1");
  TORCH_CHECK(shift.numel() == x.numel() && x_w.numel() == x.numel() && x_a.numel() == x.numel() && x_g.numel() == x.numel() && x_v.numel() == x.numel(),
              "linear_wagv_rank_in_mix_f16 input shape mismatch");
  TORCH_CHECK(w1_t.dim() == 2 && a1_t.dim() == 2 && g1_t.dim() == 2 && v1_t.dim() == 2, "rank-in weights must be 2D");
  TORCH_CHECK(w1_t.size(1) == x.numel() && a1_t.size(1) == x.numel() && g1_t.size(1) == x.numel() && v1_t.size(1) == x.numel(),
              "linear_wagv_rank_in_mix_f16 weight shape mismatch");
  return linear_wagv_rank_in_mix_f16_cuda(x, shift, x_w, x_a, x_g, x_v, w1_t, a1_t, g1_t, v1_t);
}

std::vector<torch::Tensor> linear_wag_rank_out_f16(
    torch::Tensor w1,
    torch::Tensor a1,
    torch::Tensor g1,
    torch::Tensor w2_t,
    torch::Tensor a2_t,
    torch::Tensor g2_t) {
  check_half_cuda_contig(w1, "w1");
  check_half_cuda_contig(a1, "a1");
  check_half_cuda_contig(g1, "g1");
  check_half_cuda_contig(w2_t, "w2_t");
  check_half_cuda_contig(a2_t, "a2_t");
  check_half_cuda_contig(g2_t, "g2_t");
  TORCH_CHECK(w1.dim() >= 2 && a1.dim() == w1.dim() && g1.dim() == w1.dim(), "w1/a1/g1 dim mismatch");
  TORCH_CHECK(w1.sizes().slice(0, w1.dim() - 1) == a1.sizes().slice(0, a1.dim() - 1), "w1/a1 batch mismatch");
  TORCH_CHECK(w1.sizes().slice(0, w1.dim() - 1) == g1.sizes().slice(0, g1.dim() - 1), "w1/g1 batch mismatch");
  TORCH_CHECK(w2_t.dim() == 2 && a2_t.dim() == 2 && g2_t.dim() == 2, "weight_t must be 2D");
  TORCH_CHECK(w2_t.size(0) == a2_t.size(0) && w2_t.size(0) == g2_t.size(0), "output C mismatch");
  TORCH_CHECK(w1.size(-1) == w2_t.size(1), "w rank mismatch");
  TORCH_CHECK(a1.size(-1) == a2_t.size(1), "a rank mismatch");
  TORCH_CHECK(g1.size(-1) == g2_t.size(1), "g rank mismatch");
  return linear_wag_rank_out_f16_cuda(w1, a1, g1, w2_t, a2_t, g2_t);
}

std::vector<torch::Tensor> linear_wagv_rank_out_f16(
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
    torch::Tensor v0) {
  check_half_cuda_contig(w1, "w1");
  check_half_cuda_contig(a1, "a1");
  check_half_cuda_contig(g1, "g1");
  check_half_cuda_contig(v1, "v1");
  check_half_cuda_contig(w2_t, "w2_t");
  check_half_cuda_contig(a2_t, "a2_t");
  check_half_cuda_contig(g2_t, "g2_t");
  check_half_cuda_contig(v2_t, "v2_t");
  check_half_cuda_contig(v, "v");
  check_half_cuda_contig(v_first, "v_first");
  check_half_cuda_contig(v0, "v0");
  TORCH_CHECK(w1.dim() >= 2 && a1.dim() == w1.dim() && g1.dim() == w1.dim() && v1.dim() == w1.dim(), "rank dim mismatch");
  TORCH_CHECK(w2_t.dim() == 2 && a2_t.dim() == 2 && g2_t.dim() == 2 && v2_t.dim() == 2, "weight_t must be 2D");
  TORCH_CHECK(w2_t.size(0) == a2_t.size(0) && w2_t.size(0) == g2_t.size(0) && w2_t.size(0) == v2_t.size(0), "output C mismatch");
  TORCH_CHECK(w1.size(-1) == w2_t.size(1) && a1.size(-1) == a2_t.size(1) &&
              g1.size(-1) == g2_t.size(1) && v1.size(-1) == v2_t.size(1), "rank mismatch");
  TORCH_CHECK(v.sizes() == v_first.sizes(), "v/v_first shape mismatch");
  TORCH_CHECK(v.dim() >= 2 && v.size(-1) == w2_t.size(0), "v shape mismatch");
  TORCH_CHECK(v0.dim() == 1 && v0.size(0) == w2_t.size(0), "v0 shape mismatch");
  return linear_wagv_rank_out_f16_cuda(w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0);
}

torch::Tensor add_f16(torch::Tensor x, torch::Tensor y) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(y, "y");
  TORCH_CHECK(x.sizes() == y.sizes(), "add_f16 shape mismatch");
  return add_f16_cuda(x, y);
}

std::vector<torch::Tensor> add_layer_norm_f16(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, torch::Tensor bias, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  TORCH_CHECK(x.sizes() == residual.sizes(), "add_layer_norm_f16 x/residual shape mismatch");
  TORCH_CHECK(x.dim() >= 1, "x must have at least 1 dim");
  const int64_t c = x.size(-1);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(c > 0 && c <= 8192, "unsupported C");
  return add_layer_norm_f16_cuda(x, residual, weight, bias, eps);
}

torch::Tensor add_last_layer_norm_f16(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, torch::Tensor bias, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  TORCH_CHECK(x.sizes() == residual.sizes(), "add_last_layer_norm_f16 x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3, "x must have shape [B,T,C]");
  const int64_t c = x.size(2);
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(c > 0 && c <= 8192, "unsupported C");
  return add_last_layer_norm_f16_cuda(x, residual, weight, bias, eps);
}

std::vector<torch::Tensor> add_layer_norm_cmix_mix_f16(torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias, torch::Tensor x_k, double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(shift_state, "shift_state");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  check_half_cuda_contig(x_k, "x_k");
  TORCH_CHECK(x.sizes() == residual.sizes(), "add_layer_norm_cmix_mix_f16 x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1, "add_layer_norm_cmix_mix_f16 requires shape [B,1,C]");
  const int64_t c = x.size(2);
  TORCH_CHECK(c == 4096, "add_layer_norm_cmix_mix_f16 currently requires C=4096");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == x.size(0) && shift_state.size(1) == c,
              "shift_state shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == c, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == c, "bias shape mismatch");
  TORCH_CHECK(x_k.dim() == 1 && x_k.size(0) == c, "x_k shape mismatch");
  return add_layer_norm_cmix_mix_f16_cuda(x, residual, shift_state, weight, bias, x_k, eps);
}

std::vector<torch::Tensor> add_layer_norm_tmix_mix6_f16(
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
    double eps) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(shift_state, "shift_state");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  check_half_cuda_contig(x_r, "x_r");
  check_half_cuda_contig(x_w, "x_w");
  check_half_cuda_contig(x_k, "x_k");
  check_half_cuda_contig(x_v, "x_v");
  check_half_cuda_contig(x_a, "x_a");
  check_half_cuda_contig(x_g, "x_g");
  TORCH_CHECK(x.sizes() == residual.sizes(), "add_layer_norm_tmix_mix6_f16 x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1 && x.size(2) == 4096, "add_layer_norm_tmix_mix6_f16 requires shape [B,1,4096]");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == x.size(0) && shift_state.size(1) == 4096,
              "shift_state shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == 4096, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == 4096, "bias shape mismatch");
  TORCH_CHECK(x_r.numel() == 4096 && x_w.numel() == 4096 && x_k.numel() == 4096 &&
              x_v.numel() == 4096 && x_a.numel() == 4096 && x_g.numel() == 4096,
              "mix vector shape mismatch");
  return add_layer_norm_tmix_mix6_f16_cuda(
      x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, eps);
}

std::vector<torch::Tensor> add_layer_norm_cmix_mix_f16_cfg(torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias, torch::Tensor x_k, double eps, int64_t threads) {
  TORCH_CHECK(threads == 256 || threads == 512 || threads == 1024, "threads must be 256, 512, or 1024");
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(residual, "residual");
  check_half_cuda_contig(shift_state, "shift_state");
  check_half_cuda_contig(weight, "weight");
  check_half_cuda_contig(bias, "bias");
  check_half_cuda_contig(x_k, "x_k");
  TORCH_CHECK(x.sizes() == residual.sizes(), "add_layer_norm_cmix_mix_f16_cfg x/residual shape mismatch");
  TORCH_CHECK(x.dim() == 3 && x.size(1) == 1 && x.size(2) == 4096, "add_layer_norm_cmix_mix_f16_cfg requires shape [B,1,4096]");
  TORCH_CHECK(shift_state.dim() == 2 && shift_state.size(0) == x.size(0) && shift_state.size(1) == 4096,
              "shift_state shape mismatch");
  TORCH_CHECK(weight.dim() == 1 && weight.size(0) == 4096, "weight shape mismatch");
  TORCH_CHECK(bias.dim() == 1 && bias.size(0) == 4096, "bias shape mismatch");
  TORCH_CHECK(x_k.dim() == 1 && x_k.size(0) == 4096, "x_k shape mismatch");
  return add_layer_norm_cmix_mix_f16_cfg_cuda(x, residual, shift_state, weight, bias, x_k, eps, static_cast<int>(threads));
}

void advance_i32(torch::Tensor x, int64_t amount) {
  check_i32_cuda_contig(x, "x");
  TORCH_CHECK(x.dim() == 1, "x must have shape [B]");
  advance_i32_cuda(x, amount);
}

void copy_m1_to_shift_f16(torch::Tensor x, torch::Tensor shift) {
  check_half_cuda_contig(x, "x");
  check_half_cuda_contig(shift, "shift");
  TORCH_CHECK(x.numel() == x.size(-1), "copy_m1_to_shift_f16 requires M=1");
  TORCH_CHECK(shift.numel() == x.numel(), "copy_m1_to_shift_f16 shape mismatch");
  copy_m1_to_shift_f16_cuda(x, shift);
}

} // namespace

TORCH_LIBRARY(rwkv7_v3a_ops, m) {
  m.def("identity(Tensor x) -> Tensor");
  m.def("layer_norm_f16(Tensor x, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor");
  m.def("layer_norm_f16_small(Tensor x, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor");
  m.def("layer_norm_f16_small512(Tensor x, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor");
  m.def("linear_f16(Tensor x, Tensor weight) -> Tensor");
  m.def("linear_f16_lt(Tensor x, Tensor weight) -> Tensor");
  m.def("linear_f16_m1_splitk(Tensor x, Tensor weight) -> Tensor");
  m.def("linear_mix_f16_m1_splitk(Tensor x, Tensor shift, Tensor mix, Tensor weight) -> Tensor");
  m.def("linear_f16_m1_splitk_cfg(Tensor x, Tensor weight, int chunk_k) -> Tensor");
  m.def("linear_f16_m1_splitk_tile(Tensor x, Tensor weight, int chunk_k, int tile_cols) -> Tensor");
  m.def("linear_f16_rows_splitk(Tensor x, Tensor weight, int chunk_k) -> Tensor");
  m.def("linear_t_f16(Tensor x, Tensor weight_t) -> Tensor");
  m.def("linear_t_act_f16(Tensor x, Tensor weight_t, int act) -> Tensor");
  m.def("linear_t_vres_f16(Tensor x, Tensor weight_t, Tensor v, Tensor v_first, Tensor v0) -> Tensor");
  m.def("linear_wag_rank_in_f16(Tensor xw, Tensor xa, Tensor xg, Tensor w1_t, Tensor a1_t, Tensor g1_t) -> Tensor[]");
  m.def("linear_wag_rank_in_mix_f16(Tensor x, Tensor shift, Tensor x_w, Tensor x_a, Tensor x_g, Tensor w1_t, Tensor a1_t, Tensor g1_t) -> Tensor[]");
  m.def("linear_wagv_rank_in_f16(Tensor xw, Tensor xa, Tensor xg, Tensor xv, Tensor w1_t, Tensor a1_t, Tensor g1_t, Tensor v1_t) -> Tensor[]");
  m.def("linear_wagv_rank_in_mix_f16(Tensor x, Tensor shift, Tensor x_w, Tensor x_a, Tensor x_g, Tensor x_v, Tensor w1_t, Tensor a1_t, Tensor g1_t, Tensor v1_t) -> Tensor[]");
  m.def("linear_wag_rank_out_f16(Tensor w1, Tensor a1, Tensor g1, Tensor w2_t, Tensor a2_t, Tensor g2_t) -> Tensor[]");
  m.def("linear_wagv_rank_out_f16(Tensor w1, Tensor a1, Tensor g1, Tensor v1, Tensor w2_t, Tensor a2_t, Tensor g2_t, Tensor v2_t, Tensor v, Tensor v_first, Tensor v0) -> Tensor[]");
  m.def("add_f16(Tensor x, Tensor y) -> Tensor");
  m.def("add_layer_norm_f16(Tensor x, Tensor residual, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor[]");
  m.def("add_last_layer_norm_f16(Tensor x, Tensor residual, Tensor weight, Tensor bias, float eps=1e-5) -> Tensor");
  m.def("add_layer_norm_cmix_mix_f16(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_k, float eps=1e-5) -> Tensor[]");
  m.def("add_layer_norm_tmix_mix6_f16(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, float eps=1e-5) -> Tensor[]");
  m.def("add_layer_norm_cmix_mix_f16_cfg(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_k, float eps, int threads) -> Tensor[]");
  m.def("advance_i32(Tensor(a!) x, int amount) -> ()");
  m.def("copy_m1_to_shift_f16(Tensor x, Tensor(a!) shift) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_v3a_ops, CUDA, m) {
  m.impl("identity", &identity);
  m.impl("layer_norm_f16", &layer_norm_f16);
  m.impl("layer_norm_f16_small", &layer_norm_f16_small);
  m.impl("layer_norm_f16_small512", &layer_norm_f16_small512);
  m.impl("linear_f16", &linear_f16);
  m.impl("linear_f16_lt", &linear_f16_lt);
  m.impl("linear_f16_m1_splitk", &linear_f16_m1_splitk);
  m.impl("linear_mix_f16_m1_splitk", &linear_mix_f16_m1_splitk);
  m.impl("linear_f16_m1_splitk_cfg", &linear_f16_m1_splitk_cfg);
  m.impl("linear_f16_m1_splitk_tile", &linear_f16_m1_splitk_tile);
  m.impl("linear_f16_rows_splitk", &linear_f16_rows_splitk);
  m.impl("linear_t_f16", &linear_t_f16);
  m.impl("linear_t_act_f16", &linear_t_act_f16);
  m.impl("linear_t_vres_f16", &linear_t_vres_f16);
  m.impl("linear_wag_rank_in_f16", &linear_wag_rank_in_f16);
  m.impl("linear_wag_rank_in_mix_f16", &linear_wag_rank_in_mix_f16);
  m.impl("linear_wagv_rank_in_f16", &linear_wagv_rank_in_f16);
  m.impl("linear_wagv_rank_in_mix_f16", &linear_wagv_rank_in_mix_f16);
  m.impl("linear_wag_rank_out_f16", &linear_wag_rank_out_f16);
  m.impl("linear_wagv_rank_out_f16", &linear_wagv_rank_out_f16);
  m.impl("add_f16", &add_f16);
  m.impl("add_layer_norm_f16", &add_layer_norm_f16);
  m.impl("add_last_layer_norm_f16", &add_last_layer_norm_f16);
  m.impl("add_layer_norm_cmix_mix_f16", &add_layer_norm_cmix_mix_f16);
  m.impl("add_layer_norm_tmix_mix6_f16", &add_layer_norm_tmix_mix6_f16);
  m.impl("add_layer_norm_cmix_mix_f16_cfg", &add_layer_norm_cmix_mix_f16_cfg);
  m.impl("advance_i32", &advance_i32);
  m.impl("copy_m1_to_shift_f16", &copy_m1_to_shift_f16);
}
