#include <torch/extension.h>

torch::Tensor rwkv7_mega_emb_ln0_bf16_to_f16_cuda(torch::Tensor emb, torch::Tensor weight, torch::Tensor bias, double eps);
void rwkv7_mega_emb_lookup_f16_into_cuda(torch::Tensor emb, torch::Tensor tokens, torch::Tensor out);
void rwkv7_mega_ln_mix6_into_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias, torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a, torch::Tensor x_g, torch::Tensor x_out, torch::Tensor out_r, torch::Tensor out_w, torch::Tensor out_k, torch::Tensor out_v, torch::Tensor out_a, torch::Tensor out_g, double eps, int64_t threads);
void rwkv7_mega_rkv_lowrank_pre_executor_into_cuda(torch::Tensor xr, torch::Tensor xk, torch::Tensor xv, torch::Tensor wr, torch::Tensor wk, torch::Tensor wv, torch::Tensor yr, torch::Tensor yk, torch::Tensor yv, torch::Tensor xw, torch::Tensor xa, torch::Tensor xg, torch::Tensor xlr_v, torch::Tensor w1_t, torch::Tensor a1_t, torch::Tensor g1_t, torch::Tensor v1_t, torch::Tensor w2_t, torch::Tensor g2_t, torch::Tensor w1, torch::Tensor a1, torch::Tensor g1, torch::Tensor v1, torch::Tensor w, torch::Tensor g, int64_t blocks, int64_t threads, int64_t lowrank_worker_budget, int64_t rkv_out_tile, int64_t role_order);
void rwkv7_mega_lowrank_rank_out4_kk_lanes_into_cuda(torch::Tensor w1, torch::Tensor a1, torch::Tensor g1, torch::Tensor v1, torch::Tensor w2_t, torch::Tensor a2_t, torch::Tensor g2_t, torch::Tensor v2_t, torch::Tensor v, torch::Tensor v_first, torch::Tensor v0, torch::Tensor k_raw, torch::Tensor k_k, torch::Tensor a0, torch::Tensor k_a, torch::Tensor w, torch::Tensor a, torch::Tensor g, torch::Tensor v_out, torch::Tensor new_k, torch::Tensor neg_kk, torch::Tensor kka);
void rwkv7_mega_lnx_rkvres_xg_into_cuda(torch::Tensor x, torch::Tensor r, torch::Tensor k, torch::Tensor v, torch::Tensor r_k, torch::Tensor weight, torch::Tensor bias, torch::Tensor g, torch::Tensor out, int64_t H);
void rwkv7_mega_row1_linear_exact4_into_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor y);
void rwkv7_mega_add_ln_cmix_mix_into_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias, torch::Tensor x_k, torch::Tensor x_out, torch::Tensor mixed, double eps, int64_t threads);
void rwkv7_mega_row1_linear_exact4_vec4_threads_tile_into_cuda(torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t threads, int64_t out_tile);
void rwkv7_mega_cmix_sparse_down_relu_one_vtile_hfma2_split2_into_cuda(torch::Tensor preact, torch::Tensor value_weight, torch::Tensor out);
void rwkv7_mega_add_last_layer_norm_f16_into_cuda(torch::Tensor x, torch::Tensor residual, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, double eps);

TORCH_LIBRARY(rwkv7_mega_ops_260602, m) {
    m.def("emb_ln0_bf16_to_f16(Tensor emb, Tensor weight, Tensor bias, float eps) -> Tensor");
    m.def("emb_lookup_f16_into(Tensor emb, Tensor tokens, Tensor(a!) out) -> ()");
    m.def("ln_mix6_into(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_r, Tensor x_w, Tensor x_k, Tensor x_v, Tensor x_a, Tensor x_g, Tensor(b!) x_out, Tensor(c!) out_r, Tensor(d!) out_w, Tensor(e!) out_k, Tensor(f!) out_v, Tensor(g!) out_a, Tensor(h!) out_g, float eps, int threads) -> ()");
    m.def("rkv_lowrank_pre_executor_into(Tensor xr, Tensor xk, Tensor xv, Tensor wr, Tensor wk, Tensor wv, Tensor(a!) yr, Tensor(b!) yk, Tensor(c!) yv, Tensor xw, Tensor xa, Tensor xg, Tensor xlr_v, Tensor w1_t, Tensor a1_t, Tensor g1_t, Tensor v1_t, Tensor w2_t, Tensor g2_t, Tensor(d!) w1, Tensor(e!) a1, Tensor(f!) g1, Tensor(g!) v1, Tensor(h!) w, Tensor(i!) g, int blocks, int threads, int lowrank_worker_budget, int rkv_out_tile, int role_order) -> ()");
    m.def("lowrank_rank_out4_kk_lanes_into(Tensor w1, Tensor a1, Tensor g1, Tensor v1, Tensor w2_t, Tensor a2_t, Tensor g2_t, Tensor v2_t, Tensor v, Tensor v_first, Tensor v0, Tensor k_raw, Tensor k_k, Tensor a0, Tensor k_a, Tensor(a!) w, Tensor(b!) a, Tensor(c!) g, Tensor(d!) v_out, Tensor(e!) new_k, Tensor(f!) neg_kk, Tensor(g!) kka) -> ()");
    m.def("lnx_rkvres_xg_into(Tensor x, Tensor r, Tensor k, Tensor v, Tensor r_k, Tensor weight, Tensor bias, Tensor g, Tensor(a!) out, int H) -> ()");
    m.def("row1_linear_exact4_into(Tensor x, Tensor w, Tensor(a!) y) -> ()");
    m.def("add_ln_cmix_mix_into(Tensor x, Tensor residual, Tensor(a!) shift_state, Tensor weight, Tensor bias, Tensor x_k, Tensor(b!) x_out, Tensor(c!) mixed, float eps, int threads) -> ()");
    m.def("row1_linear_exact4_vec4_threads_tile_into(Tensor x, Tensor w, Tensor(a!) y, int threads, int out_tile) -> ()");
    m.def("cmix_sparse_down_relu_one_vtile_hfma2_split2_into(Tensor preact, Tensor value_weight, Tensor(a!) out) -> ()");
    m.def("add_last_layer_norm_f16_into(Tensor x, Tensor residual, Tensor weight, Tensor bias, Tensor(a!) out, float eps) -> ()");
}

TORCH_LIBRARY_IMPL(rwkv7_mega_ops_260602, CUDA, m) {
    m.impl("emb_ln0_bf16_to_f16", [](torch::Tensor emb, torch::Tensor weight, torch::Tensor bias, double eps) {
        return rwkv7_mega_emb_ln0_bf16_to_f16_cuda(emb, weight, bias, eps);
    });
    m.impl("emb_lookup_f16_into", [](torch::Tensor emb, torch::Tensor tokens, torch::Tensor out) {
        rwkv7_mega_emb_lookup_f16_into_cuda(emb, tokens, out);
    });
    m.impl("ln_mix6_into", [](torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias, torch::Tensor x_r, torch::Tensor x_w, torch::Tensor x_k, torch::Tensor x_v, torch::Tensor x_a, torch::Tensor x_g, torch::Tensor x_out, torch::Tensor out_r, torch::Tensor out_w, torch::Tensor out_k, torch::Tensor out_v, torch::Tensor out_a, torch::Tensor out_g, double eps, int64_t threads) {
        rwkv7_mega_ln_mix6_into_cuda(x, residual, shift_state, weight, bias, x_r, x_w, x_k, x_v, x_a, x_g, x_out, out_r, out_w, out_k, out_v, out_a, out_g, eps, threads);
    });
    m.impl("rkv_lowrank_pre_executor_into", [](torch::Tensor xr, torch::Tensor xk, torch::Tensor xv, torch::Tensor wr, torch::Tensor wk, torch::Tensor wv, torch::Tensor yr, torch::Tensor yk, torch::Tensor yv, torch::Tensor xw, torch::Tensor xa, torch::Tensor xg, torch::Tensor xlr_v, torch::Tensor w1_t, torch::Tensor a1_t, torch::Tensor g1_t, torch::Tensor v1_t, torch::Tensor w2_t, torch::Tensor g2_t, torch::Tensor w1, torch::Tensor a1, torch::Tensor g1, torch::Tensor v1, torch::Tensor w, torch::Tensor g, int64_t blocks, int64_t threads, int64_t lowrank_worker_budget, int64_t rkv_out_tile, int64_t role_order) {
        rwkv7_mega_rkv_lowrank_pre_executor_into_cuda(xr, xk, xv, wr, wk, wv, yr, yk, yv, xw, xa, xg, xlr_v, w1_t, a1_t, g1_t, v1_t, w2_t, g2_t, w1, a1, g1, v1, w, g, blocks, threads, lowrank_worker_budget, rkv_out_tile, role_order);
    });
    m.impl("lowrank_rank_out4_kk_lanes_into", [](torch::Tensor w1, torch::Tensor a1, torch::Tensor g1, torch::Tensor v1, torch::Tensor w2_t, torch::Tensor a2_t, torch::Tensor g2_t, torch::Tensor v2_t, torch::Tensor v, torch::Tensor v_first, torch::Tensor v0, torch::Tensor k_raw, torch::Tensor k_k, torch::Tensor a0, torch::Tensor k_a, torch::Tensor w, torch::Tensor a, torch::Tensor g, torch::Tensor v_out, torch::Tensor new_k, torch::Tensor neg_kk, torch::Tensor kka) {
        rwkv7_mega_lowrank_rank_out4_kk_lanes_into_cuda(w1, a1, g1, v1, w2_t, a2_t, g2_t, v2_t, v, v_first, v0, k_raw, k_k, a0, k_a, w, a, g, v_out, new_k, neg_kk, kka);
    });
    m.impl("lnx_rkvres_xg_into", [](torch::Tensor x, torch::Tensor r, torch::Tensor k, torch::Tensor v, torch::Tensor r_k, torch::Tensor weight, torch::Tensor bias, torch::Tensor g, torch::Tensor out, int64_t H) {
        rwkv7_mega_lnx_rkvres_xg_into_cuda(x, r, k, v, r_k, weight, bias, g, out, H);
    });
    m.impl("row1_linear_exact4_into", [](torch::Tensor x, torch::Tensor w, torch::Tensor y) {
        rwkv7_mega_row1_linear_exact4_into_cuda(x, w, y);
    });
    m.impl("add_ln_cmix_mix_into", [](torch::Tensor x, torch::Tensor residual, torch::Tensor shift_state, torch::Tensor weight, torch::Tensor bias, torch::Tensor x_k, torch::Tensor x_out, torch::Tensor mixed, double eps, int64_t threads) {
        rwkv7_mega_add_ln_cmix_mix_into_cuda(x, residual, shift_state, weight, bias, x_k, x_out, mixed, eps, threads);
    });
    m.impl("row1_linear_exact4_vec4_threads_tile_into", [](torch::Tensor x, torch::Tensor w, torch::Tensor y, int64_t threads, int64_t out_tile) {
        rwkv7_mega_row1_linear_exact4_vec4_threads_tile_into_cuda(x, w, y, threads, out_tile);
    });
    m.impl("cmix_sparse_down_relu_one_vtile_hfma2_split2_into", [](torch::Tensor preact, torch::Tensor value_weight, torch::Tensor out) {
        rwkv7_mega_cmix_sparse_down_relu_one_vtile_hfma2_split2_into_cuda(preact, value_weight, out);
    });
    m.impl("add_last_layer_norm_f16_into", [](torch::Tensor x, torch::Tensor residual, torch::Tensor weight, torch::Tensor bias, torch::Tensor out, double eps) {
        rwkv7_mega_add_last_layer_norm_f16_into_cuda(x, residual, weight, bias, out, eps);
    });
}
