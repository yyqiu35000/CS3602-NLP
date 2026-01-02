# 量化与流式生成基准测试报告 (Quantization & Streaming Benchmark Report)

本报告基于 `Pythia-2.8B` 模型，对比了 Baseline (FP16)、Streaming (FP16) 以及结合 Int8/NF4 量化技术后的 Streaming 性能表现。测试涵盖了模型精度 (PPL)、生成速度 (Throughput) 和显存占用 (Memory) 三个维度。

## 1. 结果概览
**设置**：`ppl_tokens=1000`, `prompt_len=500`, `gen_len=500`

| Configuration | Wiki PPL | PG19 PPL | Throughput (tok/s) | TPOT (ms) | Avg Attn (ms) | Peak Mem (GB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (FP16)** | 6.98 | 8.54 | 27.27 | 36.52 | 119.56 | 5.49 |
| **Baseline + NF4** | 7.22 | 8.97 | 25.45 | 39.08 | 112.98 | 2.00 |
| **Streaming (FP16)** | 9.63 | 8.49 | 29.73 | 33.43 | 84.90 | 5.31 |
| **Streaming + NF4** | 10.19 | 8.93 | 26.80 | 37.08 | 85.12 | **1.85** |

> **注**: Streaming 配置参数为 `sink=4`, `window=256`。

---

## 2. 详细分析

### 2.1 显存占用 (Memory Efficiency)
*   **NF4 量化优势巨大**: `Streaming + NF4` 仅需 **1.85 GB** 显存，相比 Baseline (5.49 GB) 节省了约 **66%**。这意味着在低显存设备（如 4GB/6GB 显存的入门级显卡）上也能流畅运行 3B 级别的模型。
*   **Baseline + NF4**: 即使不使用 Streaming，仅使用 NF4 量化也能将显存降至 **2.00 GB**。这说明量化是降低显存的主要贡献者，而 Streaming 进一步将 KV Cache 限制住，节省了额外的 150MB。
*   **Streaming 机制**: 即使不量化，Streaming (FP16) 也能通过限制 KV Cache 大小（Window=256）略微降低显存（5.31 vs 5.49 GB）。

### 2.2 生成速度 (Inference Speed)
*   **Streaming (FP16) 速度最佳**: 达到了 **29.73 tok/s**。得益于滑动窗口机制，其 Attention 计算耗时 (Avg Attn) 仅为 **84.90 ms**，显著低于 Baseline 的 119.56 ms。
*   **NF4 的反量化开销**: 无论是 Baseline 还是 Streaming，引入 NF4 后速度均有小幅下降：
    *   `Baseline (27.27)` -> `Baseline + NF4 (25.45)`: 速度下降约 7%。
    *   `Streaming (29.73)` -> `Streaming + NF4 (26.80)`: 速度下降约 10%。
    *   **原因**: 模型的大部分参数位于 MLP (线性) 层。Int4 虽然减少了显存带宽压力，但在计算前需要将 Int4 权重**反量化** (Dequantize) 回 FP16。这种频繁的数据搬运和格式转换开销非常大，抵消了 Attention 部分带来的带宽红利。
*   **Streaming 在 NF4 下依然有效**: 对比 `Baseline + NF4` (25.45 tok/s, 112.98 ms Attn) 和 `Streaming + NF4` (26.80 tok/s, 85.12 ms Attn)，可以看出 **Streaming 机制成功降低了 Attention 耗时 (113ms -> 85ms)**，并提升了整体吞吐量。这证明流式算法在量化模型上依然能发挥加速作用，尽管被反量化的底噪掩盖了部分效果。

### 2.3 模型精度 (Model Quality / PPL)
*   **Baseline 精度最高**: 在 Wikitext 上 PPL 为 6.98。
*   **量化的影响**: `Baseline + NF4` 的 PPL 为 7.22，相比 FP16 仅有微小损失，说明 4-bit Normal Float (NF4) 量化在保留模型知识方面非常出色。
*   **Streaming 的影响**: 由于窗口限制 (256 tokens)，模型无法看到更早的上下文，导致 Wikitext PPL 上升至 9.63 (FP16) / 10.19 (NF4)。但在 PG19 数据集上，Streaming (FP16) 的 PPL (8.49) 与 Baseline (8.54) 几乎持平，说明对于某些长文本任务，局部上下文（Local Context）依然能提供很好的预测能力。
---

