#!/usr/bin/env python3
"""Clean fixed-path RWKV7 v3b B1T1 runner for the 2026-06-02 best path."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


ROOT = Path(__file__).resolve().parent
MODEL_PATH = "/dev/shm/rwkv7-g1f-7.2b-20260414-ctx8192.pth"
EVAL_JSON = ROOT.parents[0] / "rwkv7-fast-v2-standalone" / "eval_src2.json"
LN_EPS = 1e-5
HEAD_SIZE = 64
DTYPE = torch.float16
C_SIZE = 4096
RKV_EXECUTOR_BLOCKS = 7040
RKV_EXECUTOR_THREADS = 128
RKV_EXECUTOR_WORKERS = 880
RKV_OUT_TILE = 2
RKV_ROLE_ORDER_LRINT8 = 2
CMIX_KEY_THREADS = 128
CMIX_KEY_OUT_TILE = 2
LOWRANK_SUFFIXES = ("att.w1", "att.w2", "att.a1", "att.a2", "att.g1", "att.g2", "att.v1", "att.v2")


def cuda_flags(*extra: str) -> list[str]:
    flags = ["-O3", "--use_fast_math", *extra]
    if os.environ.get("RWKV7_CUDA_LINEINFO") == "1":
        flags.append("-lineinfo")
    return flags


def load_ops() -> None:
    load(
        name="rwkv7_mega_ops_260602",
        sources=[str(ROOT / "cuda/rwkv7_mega_ops_260602.cpp"), str(ROOT / "cuda/rwkv7_mega_ops_260602.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=cuda_flags(),
        is_python_module=False,
        verbose=True,
    )
    load(
        name="rwkv7_wkv_fp32io16_w0",
        sources=[str(ROOT / "cuda/rwkv7_wkv_fp32io16_w0.cpp"), str(ROOT / "cuda/rwkv7_wkv_fp32io16_w0.cu")],
        extra_cflags=["-O3", "-D_IO_FP16_"],
        extra_cuda_cflags=cuda_flags("-D_IO_FP16_"),
        is_python_module=False,
        verbose=True,
    )


def select_token(args: argparse.Namespace, vocab_size: int) -> tuple[int, str]:
    if args.token is not None:
        return int(args.token) % vocab_size, "arg"
    if args.token_source == "v3a-synthetic":
        token = (int(args.token_index) * 1103515245 + 12345) % vocab_size
        return token, f"v3a_synthetic:index={args.token_index}"
    with open(args.eval_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    tokens = obj["tokens"] if isinstance(obj, dict) else obj
    return int(tokens[args.token_index]) % vocab_size, f"eval_json:{Path(args.eval_json).name}:index={args.token_index}"


def tile_cmix_value_weight(w: torch.Tensor) -> torch.Tensor:
    ffn, channels = w.shape
    if (ffn % 128) != 0 or (channels % 256) != 0:
        raise ValueError(f"vtile requires F%128==0 and C%256==0, got {tuple(w.shape)}")
    return w.reshape(ffn // 128, 128, channels // 256, 256).permute(0, 2, 1, 3).contiguous().view(ffn, channels)


def make_layer_work(z: dict[str, torch.Tensor], layer: int, C: int, ffn: int, H: int) -> dict[str, torch.Tensor]:
    p = f"blocks.{layer}."
    work = {
        name: torch.empty((1, 1, C), device="cuda", dtype=DTYPE)
        for name in ("x_base", "xr", "xw", "xk", "xv", "xa", "xg", "new_k", "neg_kk", "kka", "wkv_y", "tail_out", "cmix_x", "cmix_mixed")
    }
    work.update({
        "r": torch.empty(C, device="cuda", dtype=DTYPE),
        "k_raw": torch.empty(C, device="cuda", dtype=DTYPE),
        "v_base": torch.empty(C, device="cuda", dtype=DTYPE),
        "att": torch.empty(C, device="cuda", dtype=DTYPE),
        "cmix_hid": torch.empty(ffn, device="cuda", dtype=DTYPE),
        "cmix_out": torch.empty(C, device="cuda", dtype=DTYPE),
        "lr_w1": torch.empty(1, z[p + "att.w1.t"].size(0), device="cuda", dtype=DTYPE),
        "lr_a1": torch.empty(1, z[p + "att.a1.t"].size(0), device="cuda", dtype=DTYPE),
        "lr_g1": torch.empty(1, z[p + "att.g1.t"].size(0), device="cuda", dtype=DTYPE),
        "lr_v1": torch.empty(1, z[p + "att.v1.t"].size(0), device="cuda", dtype=DTYPE),
        "gate_w": torch.empty(1, C, device="cuda", dtype=DTYPE),
        "gate_a": torch.empty(1, C, device="cuda", dtype=DTYPE),
        "gate_g": torch.empty(1, C, device="cuda", dtype=DTYPE),
        "gate_v": torch.empty(1, C, device="cuda", dtype=DTYPE),
    })
    return work


class RWKV7FastB1T1260602:
    def __init__(self, model_path: str) -> None:
        t0 = time.perf_counter()
        src = torch.load(model_path, map_location="cpu", mmap=True)
        self.H, self.N = [int(x) for x in src["blocks.0.att.r_k"].shape]
        self.C = self.H * self.N
        self.V = int(src["emb.weight"].shape[0])
        self.L = max(int(k.split(".")[1]) for k in src if k.startswith("blocks.")) + 1
        if self.N != HEAD_SIZE or self.C != C_SIZE:
            raise ValueError(f"expected HxN={self.H}x{HEAD_SIZE} and C={C_SIZE}, got H={self.H} N={self.N} C={self.C}")
        self.z: dict[str, torch.Tensor] = {}
        self.load_weights(src)
        ffn = int(self.z["blocks.0.ffn.key.weight"].size(0))
        self.input_work = torch.empty((1, 1, self.C), device="cuda", dtype=DTYPE)
        self.work = [make_layer_work(self.z, layer, self.C, ffn, self.H) for layer in range(self.L)]
        self.ln_out = torch.empty((1, self.C), device="cuda", dtype=DTYPE)
        self.logits = torch.empty(self.V, device="cuda", dtype=DTYPE)
        self.initial_residual = torch.zeros((1, 1, self.C), device="cuda", dtype=DTYPE)
        torch.cuda.synchronize()
        print(
            f"[260602] ready L={self.L} C={self.C} H={self.H} V={self.V} "
            f"rkv_executor=k2pipe_body21 blocks={RKV_EXECUTOR_BLOCKS} workers={RKV_EXECUTOR_WORKERS} "
            f"cmix=nofc_f32acc_key_vec4_vtile_hfma2_split2 tail=split load_s={time.perf_counter() - t0:.3f}",
            flush=True,
        )

    def load_weights(self, src: dict[str, torch.Tensor]) -> None:
        ops = torch.ops.rwkv7_mega_ops_260602
        emb = src["emb.weight"].squeeze().to(device="cuda").contiguous()
        ln0_w = src["blocks.0.ln0.weight"].squeeze().to(device="cuda").contiguous()
        ln0_b = src["blocks.0.ln0.bias"].squeeze().to(device="cuda").contiguous()
        self.z["emb.weight"] = ops.emb_ln0_bf16_to_f16(emb, ln0_w, ln0_b, LN_EPS)
        for key, tensor in src.items():
            if key == "emb.weight" or key.startswith("blocks.0.ln0."):
                continue
            value = tensor.squeeze()
            if key.endswith(LOWRANK_SUFFIXES):
                self.z[key + ".t"] = value.to(device="cuda", dtype=DTYPE).t().contiguous()
                continue
            value = value.to(device="cuda", dtype=DTYPE).contiguous()
            if key.endswith("att.r_k"):
                value = value.flatten().contiguous()
            if key.endswith("ffn.value.weight"):
                self.z[key + ".sparse"] = tile_cmix_value_weight(value.t().contiguous())
            else:
                self.z[key] = value
        p0 = "blocks.0.att."
        if p0 + "v1.t" not in self.z:
            self.z[p0 + "v1.t"] = torch.zeros((1, self.C), device="cuda", dtype=DTYPE)
            self.z[p0 + "v2.t"] = torch.zeros((self.C, 1), device="cuda", dtype=DTYPE)
            self.z[p0 + "v0"] = torch.zeros((self.C,), device="cuda", dtype=DTYPE)

    def zero_state(self) -> list[torch.Tensor]:
        return [
            torch.zeros((self.L, 2, 1, self.C), device="cuda", dtype=DTYPE),
            torch.zeros((self.L, 1, self.H, self.N, self.N), device="cuda", dtype=torch.float32),
        ]

    def forward(self, tokens: torch.Tensor, state: list[torch.Tensor]) -> torch.Tensor:
        ops = torch.ops.rwkv7_mega_ops_260602
        ops.emb_lookup_f16_into(self.z["emb.weight"], tokens.contiguous(), self.input_work)
        x = self.input_work
        residual = self.initial_residual
        v_first = None
        for layer in range(self.L):
            x, residual, v = self.layer(layer, x, residual, state[0][layer], state[1][layer], v_first)
            if layer == 0:
                v_first = v
        ops.add_last_layer_norm_f16_into(x, residual, self.z["ln_out.weight"], self.z["ln_out.bias"], self.ln_out, LN_EPS)
        ops.row1_linear_exact4_into(self.ln_out.view(self.C), self.z["head.weight"], self.logits)
        return self.logits

    def layer(self, layer: int, x: torch.Tensor, residual: torch.Tensor, shift: torch.Tensor, wkv_state: torch.Tensor, v_first: torch.Tensor | None):
        ops = torch.ops.rwkv7_mega_ops_260602
        wkv = torch.ops.rwkv7_wkv_fp32io16_w0
        z = self.z
        p = f"blocks.{layer}."
        work = self.work[layer]
        ops.ln_mix6_into(
            x, residual, shift[0].view(1, 1, self.C), z[p + "ln1.weight"], z[p + "ln1.bias"],
            z[p + "att.x_r"], z[p + "att.x_w"], z[p + "att.x_k"], z[p + "att.x_v"], z[p + "att.x_a"], z[p + "att.x_g"],
            work["x_base"], work["xr"], work["xw"], work["xk"], work["xv"], work["xa"], work["xg"], LN_EPS, 1024,
        )
        scratch = {
            "w1": work["lr_w1"], "a1": work["lr_a1"], "g1": work["lr_g1"], "v1": work["lr_v1"],
            "w": work["gate_w"], "a": work["gate_a"], "g": work["gate_g"], "v_out": work["gate_v"],
        }
        ops.rkv_lowrank_pre_executor_into(
            work["xr"].view(-1).contiguous(), work["xk"].view(-1).contiguous(), work["xv"].view(-1).contiguous(),
            z[p + "att.receptance.weight"], z[p + "att.key.weight"], z[p + "att.value.weight"],
            work["r"], work["k_raw"], work["v_base"],
            work["xw"].view(1, self.C), work["xa"].view(1, self.C), work["xg"].view(1, self.C), work["xv"].view(1, self.C),
            z[p + "att.w1.t"], z[p + "att.a1.t"], z[p + "att.g1.t"], z[p + "att.v1.t"], z[p + "att.w2.t"], z[p + "att.g2.t"],
            scratch["w1"], scratch["a1"], scratch["g1"], scratch["v1"], scratch["w"], scratch["g"],
            RKV_EXECUTOR_BLOCKS, RKV_EXECUTOR_THREADS, RKV_EXECUTOR_WORKERS, RKV_OUT_TILE,
            RKV_ROLE_ORDER_LRINT8,
        )
        r = work["r"].view(1, 1, self.C)
        k_raw = work["k_raw"].view_as(r)
        v_base = work["v_base"].view_as(r)
        v_ref = v_base if layer == 0 else v_first
        ops.lowrank_rank_out4_kk_lanes_into(
            scratch["w1"], scratch["a1"], scratch["g1"], scratch["v1"],
            z[p + "att.w2.t"], z[p + "att.a2.t"], z[p + "att.g2.t"], z[p + "att.v2.t"],
            v_base.view(1, self.C), v_ref.view(1, self.C), z[p + "att.v0"],
            k_raw.view(1, self.C), z[p + "att.k_k"], z[p + "att.a0"], z[p + "att.k_a"],
            scratch["w"], scratch["a"], scratch["g"], scratch["v_out"],
            work["new_k"].view(1, self.C), work["neg_kk"].view(1, self.C), work["kka"].view(1, self.C),
        )
        w = scratch["w"].view(1, 1, self.C).contiguous()
        a = scratch["a"].view_as(w).contiguous()
        g = scratch["g"].view_as(w).contiguous()
        v = scratch["v_out"].view_as(w).contiguous()
        wkv.forward_w0(1, 1, self.C, self.H, wkv_state, r, w, z[p + "att.w0"], work["new_k"], v, work["neg_kk"], work["kka"], work["wkv_y"])
        ops.lnx_rkvres_xg_into(work["wkv_y"], r, work["new_k"], v, z[p + "att.r_k"], z[p + "att.ln_x.weight"], z[p + "att.ln_x.bias"], g, work["tail_out"], self.H)
        ops.row1_linear_exact4_into(work["tail_out"].view(-1), z[p + "att.output.weight"], work["att"])
        ops.add_ln_cmix_mix_into(
            work["x_base"], work["att"].view(1, 1, self.C), shift[1], z[p + "ln2.weight"], z[p + "ln2.bias"], z[p + "ffn.x_k"],
            work["cmix_x"], work["cmix_mixed"], LN_EPS, 1024,
        )
        ops.row1_linear_exact4_vec4_threads_tile_into(
            work["cmix_mixed"].view(-1).contiguous(), z[p + "ffn.key.weight"], work["cmix_hid"], CMIX_KEY_THREADS, CMIX_KEY_OUT_TILE
        )
        ops.cmix_sparse_down_relu_one_vtile_hfma2_split2_into(work["cmix_hid"], z[p + "ffn.value.weight.sparse"], work["cmix_out"])
        return work["cmix_x"], work["cmix_out"].view(1, 1, self.C), v


def bench(model: RWKV7FastB1T1260602, token: torch.Tensor, warmup: int, iters: int, graph: bool) -> list[float]:
    state = model.zero_state()
    for _ in range(warmup):
        model.forward(token, state)
    torch.cuda.synchronize()
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            model.forward(token, state)
        torch.cuda.synchronize()
        times = []
        for _ in range(iters):
            t0 = time.perf_counter()
            g.replay()
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
        return times
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.forward(token, state)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return times


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--eval-json", default=str(EVAL_JSON))
    p.add_argument("--token-source", choices=("v3a-synthetic", "eval-json"), default="v3a-synthetic")
    p.add_argument("--token-index", type=int, default=0)
    p.add_argument("--token", type=int, default=None)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    load_ops()
    model = RWKV7FastB1T1260602(args.model)
    token_id, source = select_token(args, model.V)
    token = torch.tensor([[token_id]], device="cuda", dtype=torch.long)
    logits = model.forward(token, model.zero_state())
    torch.cuda.synchronize()
    top = torch.topk(logits.float(), 5)
    print(f"CONFIG token={token_id} source={source} graph={int(args.graph)} warmup={args.warmup} iters={args.iters}", flush=True)
    print(f"SMOKE top_ids={top.indices.tolist()} top_vals={[round(x, 4) for x in top.values.tolist()]}", flush=True)
    times = bench(model, token, args.warmup, args.iters, args.graph)
    p50 = statistics.median(times)
    p10 = sorted(times)[max(0, int(len(times) * 0.10) - 1)]
    p90 = sorted(times)[min(len(times) - 1, int(len(times) * 0.90))]
    print(f"RESULT p10_ms={p10:.6f} p50_ms={p50:.6f} p90_ms={p90:.6f} tok_s_p50={1000.0 / p50:.2f}", flush=True)


if __name__ == "__main__":
    main()
