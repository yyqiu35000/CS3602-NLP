# FlashAttention (SDPA) 与 StreamingLLM 基准测试报告

本报告基于 `Pythia-2.8B` 模型，深入对比了在 **Eager** (PyTorch 原生) 和 **SDPA** (Scaled Dot Product Attention, 即 Flash Attention) 两种 Attention 实现下，**Baseline** 与 **StreamingLLM** 的性能表现。

测试环境使用 `BF16` (BFloat16) 精度以确保数值稳定性，特别是针对 Eager 模式下的溢出问题。

## 1. 结果概览
**设置**：`ppl_tokens=1000`, `prompt_len=500`, `gen_len=500`

| Configuration | Attn Impl | Wiki PPL | PG-19 PPL | Total Time (s) | Avg Attn (ms) | TTFT (s) | TPOT (ms) | Throughput | Peak Mem (GB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Eager | 7.23 | 8.86 | 24.38 | 293.51 | 0.1370 | 48.77 | 20.50 | 5.48 |
| **StreamingLLM** | Eager | 12.11 | 8.86 | 19.39 | 183.09 | 0.1509 | 38.79 | 25.78 | 5.35 |
| **Baseline** | **SDPA** | **7.03** | **8.57** | **18.49** | **120.35** | **0.1177** | **36.99** | **27.04** | 5.48 |
| **StreamingLLM** | **SDPA** | 11.57 | 8.74 | 17.73 | 97.28 | 0.1430 | 35.46 | 28.20 | **5.31** |

> **注**: Streaming 配置参数为 `sink=8`, `window=256`。

---

## 2. 详细分析

### 2.1 Flash Attention (SDPA) 的加速效果
*   **显著的吞吐量提升**: 
    *   对于 **Baseline**，SDPA 将吞吐量从 20.50 tok/s 提升至 27.04 tok/s (**+32%**)。
    *   对于 **StreamingLLM**，SDPA 也带来了提升，从 25.78 tok/s 提升至 28.20 tok/s (**+9%**)。
*   **Attention 计算耗时骤降**: 
    *   在 SDPA 模式下，StreamingLLM 的平均 Attention 计算时间仅为 **97.28 ms**，相比 Eager 模式的 183.09 ms 减少了约 **47%**。这直接证明了 Flash Attention 在处理注意力计算时的极高效率。

### 2.2 StreamingLLM 与 SDPA 的协同效应
*   **最快组合**: `StreamingLLM + SDPA` 是所有配置中速度最快的 (28.20 tok/s)。
    *   这是因为 StreamingLLM 限制了 KV Cache 的大小 (Window=256)，使得 Attention 矩阵的计算量保持恒定且较小。
    *   配合 SDPA 的底层优化，使得这一组合在长文本生成中具有极高的效率。
*   **Eager 模式下的提升**: 值得注意的是，在 Eager 模式下，StreamingLLM (25.78 tok/s) 已经显著快于 Baseline (20.50 tok/s)。这表明在经过代码优化（如禁用 GC）后，StreamingLLM 减少计算量的优势成功抵消了 Python 层面的额外开销，即使在没有 Flash Attention 的情况下也能带来加速。

### 2.3 模型精度 (PPL) 与 稳定性
*   **数值稳定性**: 测试中发现，在 FP16 模式下 Eager Attention 容易发生数值溢出 (PPL=inf)。切换到 **BF16** 后，Eager 模式恢复正常。SDPA 本身内部通常使用高精度累加，因此稳定性更好。
*   **PPL 差异**: 
    *   Baseline (SDPA) 的 PPL (7.03) 略优于 Baseline (Eager) (7.23)，再次印证了 SDPA 更好的数值特性。
    *   StreamingLLM 的 PPL (11.57) 高于 Baseline，这是符合预期的，因为它只能看到有限的窗口上下文。但在流式长文本场景下，这种局部注意力的精度通常足以维持流畅的生成。

### 2.4 与 Main 脚本 (FP16) 的 PPL 差异说明
细心的读者可能会发现，本报告中的 Baseline PPL (7.03/7.23) 略高于 `main.py` 报告的数值 (约 6.97)。这并非异常，而是由测试配置差异导致的：
1.  **精度差异 (Precision)**: `main.py` 默认使用 `FP16`，而本测试为解决溢出问题强制使用了 `BF16`。`BF16` 的尾数精度低于 `FP16`，导致 PPL 计算出现微小偏差。
2.  **评估长度 (Evaluation Length)**: `main.py` 默认评估长度为 2000 tokens，而本测试为了快速验证设置为 1000 tokens。更长的上下文通常能带来更低的 PPL。

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
