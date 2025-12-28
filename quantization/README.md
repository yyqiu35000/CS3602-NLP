# 量化与流式生成基准测试报告 (Quantization & Streaming Benchmark Report)

本报告基于 `Pythia-2.8B` 模型，对比了 Baseline (FP16)、Streaming (FP16) 以及结合 Int8/NF4 量化技术后的 Streaming 性能表现。测试涵盖了模型精度 (PPL)、生成速度 (Throughput) 和显存占用 (Memory) 三个维度。

## 1. 结果概览

| Configuration | Wiki PPL | PG19 PPL | Throughput (tok/s) | TPOT (ms) | Avg Attn (ms) | Peak Mem (GB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline (FP16)** | **6.98** | 8.54 | 23.57 | 42.25 | 138.52 | 5.49 |
| **Streaming (FP16)** | 9.66 | **8.49** | **23.95** | **41.57** | 122.03 | 5.31 |
| **Streaming + Int8** | 9.72 | 8.54 | 9.21 | 108.51 | 120.95 | 2.98 |
| **Streaming + NF4** | 10.24 | 8.93 | 18.88 | 52.73 | **98.53** | **1.85** |

> **注**: Streaming 配置参数为 `sink=4`, `window=256`。

---

## 2. 详细分析

### 2.1 显存占用 (Memory Efficiency)
*   **NF4 量化优势巨大**: `Streaming + NF4` 仅需 **1.85 GB** 显存，相比 Baseline (5.49 GB) 节省了约 **66%**。这意味着在低显存设备（如 4GB/6GB 显存的入门级显卡）上也能流畅运行 3B 级别的模型。
*   **Int8 表现尚可**: `Streaming + Int8` 占用 2.98 GB，节省约 45% 显存。
*   **Streaming 机制**: 即使不量化，Streaming (FP16) 也能通过限制 KV Cache 大小（Window=256）略微降低显存（5.31 vs 5.49 GB），且能防止长文本生成时的显存爆炸。

### 2.2 生成速度 (Inference Speed)
*   **Streaming (FP16) 速度最佳**: 达到了 **23.95 tok/s**，略快于 Baseline。这得益于滑动窗口机制将 Attention 计算复杂度从 $O(N^2)$ 降低到了 $O(N \times Window)$，从 `Avg Attn` 指标可以看出，Streaming 的 Attention 耗时 (122ms) 明显低于 Baseline (138ms)。
*   **Int8 速度瓶颈**: `Streaming + Int8` 的速度大幅下降至 **9.21 tok/s**。这是由于 `bitsandbytes` 的 Int8 实现主要针对显存优化，其解量化 (Dequantization) 过程引入了较大的计算开销，导致在小 Batch 推理时速度反而变慢。
*   **NF4 性价比极高**: `Streaming + NF4` 保持了 **18.88 tok/s** 的较高速度，仅比 FP16 慢约 20%，但换来了巨大的显存优势。

### 2.3 模型精度 (Model Quality / PPL)
*   **Baseline 精度最高**: 在 Wikitext 上 PPL 为 6.98。
*   **Streaming 的影响**: 由于窗口限制 (256 tokens)，模型无法看到更早的上下文，导致 Wikitext PPL 上升至 9.66。但在 PG19 数据集上，Streaming (FP16) 的 PPL (8.49) 与 Baseline (8.54) 几乎持平，说明对于某些长文本任务，局部上下文（Local Context）依然能提供很好的预测能力。
*   **量化的影响**:
    *   **Int8**: 几乎无损 (Wiki PPL 9.72 vs Streaming FP16 9.66)。
    *   **NF4**: 略有损失 (Wiki PPL 10.24)，但在可接受范围内，考虑到其带来的显存和速度收益，这是一个非常划算的交换。

---

