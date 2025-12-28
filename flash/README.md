# FlashAttention (SDPA) 与 StreamingLLM 基准测试报告

本报告基于 `Pythia-2.8B` 模型，深入对比了在 **Eager** (PyTorch 原生) 和 **SDPA** (Scaled Dot Product Attention, 即 Flash Attention) 两种 Attention 实现下，**Baseline** 与 **StreamingLLM** 的性能表现。

测试环境使用 `BF16` (BFloat16) 精度以确保数值稳定性，特别是针对 Eager 模式下的溢出问题。

## 1. 结果概览

| Configuration | Attn Impl | Wiki PPL | Throughput (tok/s) | Avg Attn (ms) | Total Time (s) | Peak Mem (GB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Eager | 7.23 | 18.49 | 356.66 | 27.05 | 5.48 |
| **StreamingLLM** | Eager | 12.11 | 18.59 | 331.74 | 26.90 | 5.35 |
| **Baseline** | **SDPA** | **7.03** | 25.57 | 125.72 | 19.55 | 5.48 |
| **StreamingLLM** | **SDPA** | 11.57 | **30.77** | **70.29** | **16.25** | **5.31** |

> **注**: Streaming 配置参数为 `sink=8`, `window=256`。

---

## 2. 详细分析

### 2.1 Flash Attention (SDPA) 的加速效果
*   **显著的吞吐量提升**: 
    *   对于 **Baseline**，SDPA 将吞吐量从 18.49 tok/s 提升至 25.57 tok/s (**+38%**)。
    *   对于 **StreamingLLM**，SDPA 的提升更为夸张，从 18.59 tok/s 飙升至 30.77 tok/s (**+65%**)。
*   **Attention 计算耗时骤降**: 
    *   在 SDPA 模式下，StreamingLLM 的平均 Attention 计算时间仅为 **70.29 ms**，相比 Eager 模式的 331.74 ms 减少了约 **79%**。这直接证明了 Flash Attention 在处理注意力计算时的极高效率。

### 2.2 StreamingLLM 与 SDPA 的协同效应
*   **最快组合**: `StreamingLLM + SDPA` 是所有配置中速度最快的 (30.77 tok/s)。
    *   这是因为 StreamingLLM 限制了 KV Cache 的大小 (Window=256)，使得 Attention 矩阵的计算量保持恒定且较小。
    *   配合 SDPA 的底层优化，使得这一组合在长文本生成中具有极高的效率，甚至超过了全量计算的 Baseline (SDPA)。
*   **Eager 模式下的瓶颈**: 在 Eager 模式下，StreamingLLM (18.59 tok/s) 与 Baseline (18.49 tok/s) 速度几乎持平。这是因为 Python层面的 Cache 驱逐 (Eviction) 操作带来了额外的 CPU 开销，抵消了减少 Attention 计算量带来的优势。而在 SDPA 模式下，Attention 计算极快，使得整体生成速度不再受限于计算瓶颈，从而体现出 Streaming 的优势。

### 2.3 模型精度 (PPL) 与 稳定性
*   **数值稳定性**: 测试中发现，在 FP16 模式下 Eager Attention 容易发生数值溢出 (PPL=inf)。切换到 **BF16** 后，Eager 模式恢复正常。SDPA 本身内部通常使用高精度累加，因此稳定性更好。
*   **PPL 差异**: 
    *   Baseline (SDPA) 的 PPL (7.03) 略优于 Baseline (Eager) (7.23)，再次印证了 SDPA 更好的数值特性。
    *   StreamingLLM 的 PPL (11.57) 高于 Baseline，这是符合预期的，因为它只能看到有限的窗口上下文。但在流式长文本场景下，这种局部注意力的精度通常足以维持流畅的生成。
---

## 3. 遇到的问题与解决方案总结

### 3.1 Eager 模式下的数值溢出问题 (FP16 Overflow)

**问题现象**:
*   在初始测试中，当 `attn_implementation="eager"` 且使用 `torch.float16` (FP16) 时，模型出现严重故障。
*   **Perplexity (PPL)** 显示为 `inf` (无穷大)。
*   **生成异常**: 本应生成 500 个 token，实际只生成了 1 个 token 即停止。

**原因分析**:
*   `FP16` (半精度浮点数) 的动态范围较小（最大值约为 65504）。
*   在 Attention 计算过程中（尤其是 Softmax 之前的 Score 计算），中间结果可能会超过 FP16 的表示范围，导致数值溢出（Overflow）。
*   `SDPA` (Flash Attention) 内部实现通常采用了数值稳定性优化（如在更高精度下累加），或者其计算路径规避了部分溢出风险，因此在 FP16 下表现正常。
*   `Eager` 模式直接使用 PyTorch 的标准算子按顺序计算，缺乏针对低精度大数值的特定优化，更容易触发 FP16 溢出。

**解决方案**:
*   **切换精度至 BF16**: 将数据类型从 `torch.float16` 改为 `torch.bfloat16`。
    *   `BF16` 牺牲了部分尾数精度，但保留了与 FP32 相同的指数位宽，因此具有与 FP32 相同的动态范围，极大地减少了溢出风险。
*   **结果**: 切换到 BF16 后，Eager 模式下的 PPL 恢复正常 (7.23)，生成长度也恢复符合预期 (500 tokens)。
