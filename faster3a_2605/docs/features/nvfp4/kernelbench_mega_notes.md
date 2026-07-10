# KernelBench-Mega 学习笔记 — 超级内核优化思路

## 来源
- [KernelBench-Mega](https://kernelbench.com/mega)
- [Fable 5 完整优化记录](https://huggingface.co/datasets/Infatoshi/kernelbench-mega-traces/blob/main/20260701_172615_claude_claude-fable-5_02_kimi_linear_decode.jsonl)
- 测试题目: `02_kimi_linear_decode` (Kimi-Linear W4A16, 4-bit 权重 + bf16 激活)
- 最佳成绩: Fable 5 达到 **18.71x** 加速 (RTX PRO 6000 Blackwell)

---

## Fable 5 核心技术

### 1. 单次 Kernel Launch — 超级内核
将整个推理流程压进**一个 CUDA kernel**，中间不切换：
- int4 解量化 → 卷积/矩阵乘 → SiLU → 门控 → 注意力 → MoE 路由 → RMSNorm → KV Cache 写入
- 全部塞进一次 launch，靠 **14 道 grid barrier** 分阶段执行
- 对比：其他模型需要 4-14 次独立 kernel launch

### 2. Grid Barrier 替代 Kernel Launch
- 使用 `cuda::cooperative_groups::grid_sync()` 或 `__grid_sync()` 在单个 kernel 内同步
- 避免每次 kernel launch 的固定开销 (~5-10 μs/launch)
- 上下文越长，摊薄效果越显著 (2K: 17.8x → 16K: 19.5x)

### 3. Int4 解量化 "几乎免费"
- 将 int4 反量化与计算**交织**进行，利用内存延迟隐藏
- 解量化不再是一个独立步骤，而是在 K 循环累加时即时完成
- 这正是我们 `e2m1_decode_f()` 的思路 — 位运算解码已经很快了

### 4. Roofline 分析先行
- Fable 5 用 **64% 的时间**做测量和推导
- 先建立 baseline (5.47/6.05/6.74 ms/tok @ 2K/8K/16K)
- 微基准 grid barrier 开销
- 推导出 "每 token 约 29 倍字节" 的 roofline 上限
- 然后才一口气写出完整 kernel

---

## 对 RWKV-7 NVFP4 推理的启发

### 当前瓶颈分析

我们的 NVFP4 实现：
- T=1 decode: 179 tok/s → **5.58 ms/tok**
- 已优化: 内核带宽 350 GB/s (e2m1_decode_f 位运算)
- 瓶颈: **Python dispatch 开销 + 多次 kernel launch**

每个 RWKV-7 层的 kernel launch 数：
```
ln1 + tmix_fuse (linear ×6 + wkv + group_norm) + ln2 + ffn (linear ×2 + cmix)
≈ 10-15 次 kernel launch per layer × 32 layers = 320-480 次/token
```

### 可应用的优化

#### Phase 1: 减少 Kernel Launch (短期可行)
- **融合 linear + activation**: 将 SiLU/ReLU² 与前一个 GEMV 内核融合
- **融合 LayerNorm + linear**: ln1 输出直接传入 tmix 的第一个 linear
- **融合 cmix_sparse**: ReLU² + dequant + matmul 已经融合了，这个已经做了

#### Phase 2: 单层 Megakernel (中期)
- 将整个 RWKV-7 block (ln1 + tmix + ln2 + ffn) 压进一个 kernel
- 使用 `__grid_sync()` 在 kernel 内分阶段执行
- 预期效果: 消除 ~15 次 launch × 5μs = 75μs/layer 的固定开销
- 32 层 × 75μs = **2.4ms 节省** → 可将 5.58ms 降至 ~3.2ms → ~312 tok/s

#### Phase 3: 多层 Megakernel (长期)
- 将多个连续层压进一个 kernel
- 需要处理层间依赖 (上一层的输出是下一层的输入)
- 使用 shared memory 或 global memory + grid barrier

### 关键约束
- RWKV-7 的 **广义 delta 规则** 不能修改 — 这限制了 WKV 部分的融合灵活性
- **ffn.value.weight 必须保持 FP16** — cmix 内核硬编码了 fp16 权重访问
- **sm_120 (Blackwell)**: 支持 cooperative launch, `cuda::cooperative_groups`

### 具体实施路线

```
当前: [ln1 kernel] [linear_r] [linear_k] [linear_v] [linear_a] [linear_g] [wkv] [group_norm] [ln2] [linear_key] [cmix] [linear_value]
      ↑ 12-15 launches × 32 layers = 384-480 launches/token

Phase 1: [ln1+linear_r fused] [linear_k+v fused] [linear_a+g fused] [wkv+gnorm fused] [ln2+linear_key fused] [cmix+linear_value fused]
         ↑ 6 launches × 32 layers = 192 launches/token (50% reduction)

Phase 2: [full layer megakernel with grid barriers]
         ↑ 1 launch × 32 layers = 32 launches/token (93% reduction)
```

### 预期性能提升

| 优化阶段 | Launches/Token | 预期延迟 | 预期 tok/s |
|----------|---------------|---------|-----------|
| 当前 (已优化内核) | ~384 | 5.58 ms | 179 |
| Phase 1 (融合) | ~192 | ~3.5 ms | ~286 |
| Phase 2 (单层 mega) | ~32 | ~2.5 ms | ~400 |
| Phase 3 (多层 mega) | ~4 | ~2.0 ms | ~500 |

> Phase 2+ 需要 cooperative launch 和 grid_sync, 在 sm_120 上可行

---

*学习日期: 2026-07-10*
