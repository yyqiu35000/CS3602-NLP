# StreamingLLM 创新方法 (Innovations) 基准测试报告

本报告基于 `Pythia-2.8B` 模型，对比了 **Baseline**、**StreamingLLM (Original)** 以及多种改进策略（**POS Aware**、**Semantic Block**）在 Wikitext 数据集上的性能表现。

## 1. 创新方法简介

为了在有限的 KV Cache 预算下提升长文本生成的质量，我们在原始 StreamingLLM (Sliding Window + Attention Sink) 的基础上引入了两种新的 Token 选择策略：

### 1.1 POS Aware (词性感知选择)
*   **核心思路**: 并非所有 Token 对理解上下文都同等重要。停用词 (Stop Words，如 "the", "is", "at") 虽然在语法上必要，但往往不承载核心语义。
*   **实现方式**: 在 KV Cache 驱逐 (Eviction) 阶段，优先保留**非停用词** (名词、动词等实词)。我们维护一个停用词表，同时将StreamingLLM的窗口改良为8+192+64，中间部分作为位置缓冲区，当缓存满时，优先移除最早进入的停用词，从而让有限的窗口容纳更多有意义的内容。

### 1.2 Semantic Block (语义块选择)
*   **核心思路**: 上下文的相关性并不完全取决于距离 (Recency)。某些早期的文本块可能包含对当前生成至关重要的信息，但在滑动窗口机制下会被过早丢弃。
*   **实现方式**: 
    同样我们将Streaming窗口改良为8（sink）+192（extract）+64（window），同时在kv中维护一块位置缓冲区。而针对缓冲区，这次我们做出如下几种改动
    1、我们针对超出window的token，我们不直接丢弃，而是放入缓冲区，而缓冲区token积累到一定数量后，进行固定压缩或动态压缩或不压缩，
    2、固定压缩：我们将缓冲区的token按照固定大小的块进行压缩，每个块的大小为16个token，每个块的压缩方法为Mean Pooling，即将块中的所有token的embedding取平均值，得到一个新的embedding，作为该块的表示。
    3、动态压缩：我们将缓冲区的token按照动态大小的块进行压缩，每个块的大小根据当前缓冲区的token数量以及前后的k相似关系动态调整，压缩方法同上，作为该块的表示。
    4、不压缩：我们仍按照固定大作对KV进行分块，而块内我们仅对K压缩取均值代表某块，不对V进行压缩。
    然后extract区域则是从缓冲区进行提取。提取方式我们采取top-k策略。
*   **优势与劣势**:
    *   **优势**: 能够有效保留上下文的语义信息，避免因滑动窗口机制而丢失重要的历史信息。
    *   **劣势**: 压缩率较高时，可能会引入一定的信息损失。（取均值手段还可能混淆信息），同时Streaming原有的无限生成优势被打破，固定压缩和动态压缩只是延缓了KV增长，而不压缩方法更是只节约了K的空间，而V的空间仍然是线性增长的。  
---

## 2. 结果概览

测试环境统一使用 `SDPA` (Flash Attention) 加速，并控制总 KV Cache 大小进行公平对比。

| Configuration | Type | Wiki PPL | Throughput (tok/s) | Avg Attn (ms) | Peak Mem (GB) | KV Size (MB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | Full Attention | **6.97** | 28.27 | 109.96 | 5.48 | 0.00 |
| **StreamingLLM (Original)** | Sliding Window | 11.49 | 30.19 | 81.29 | 5.31 | 82.50 |
| **Streaming+Pos** | POS Filtered | 11.49 | 30.35 | 81.05 | 5.31 | 82.50 |
| **Streaming+Semantic (Fixed)** | Compressed (16x) | 12.12 | **30.62** | **70.53** | **5.28** | **60.00** |
| **Streaming+Semantic (Dynamic)** | Compressed (Dyn) | 12.12 | 30.49 | 74.70 | **5.28** | **60.00** |
| **Streaming+Semantic (No Comp)** | Selection Only | **10.61** | 30.10 | 76.23 | 5.31 | 82.50 |

> **注**: 所有流式配置的总 Token 窗口限制在 ~264 tokens 左右。

---

## 3. 详细分析

### 3.1 Semantic Block (No Comp) vs. StreamingLLM
*   **PPL 提升**: 在相同 KV Cache 大小下，`Streaming+Semantic (No Comp)` 的 PPL 为 **10.61**，优于原始 StreamingLLM 的 **11.49**。
*   **分析**: 相比于 StreamingLLM 仅基于时间滑动窗口保留最近的 Token，Semantic Block 通过计算余弦相似度保留与当前查询最相关的历史块。结果表明，基于语义相关性的选择策略能更有效地保留关键上下文信息，从而提升模型困惑度表现。

### 3.2 Semantic Block (Fixed/Dynamic Compression) vs. StreamingLLM
*   **计算效率**: `Streaming+Semantic (Fixed)` 的平均 Attention 计算时间为 **70.53 ms**，相比原始 StreamingLLM (81.29 ms) 降低了约 **13%**。
*   **显存占用**: 通过对非选中块进行 16x 压缩，KV Cache 大小从 82.50 MB 降低至 **60.00 MB**。
*   **PPL 权衡**: 压缩策略导致 PPL 上升至 **12.12** (相比原始 StreamingLLM 增加 0.63)。这表明虽然压缩能显著提升速度并降低显存，但会对生成质量产生一定负面影响。

### 3.3 POS Aware vs. StreamingLLM
*   **性能表现**: `Streaming+Pos` 的 PPL (11.49) 和吞吐量 (30.35 tok/s) 与原始 StreamingLLM 基本一致。
*   **分析**: 在本测试的窗口限制 (256 tokens) 和数据集下，单纯过滤停用词并未带来额外的 PPL 收益。这可能是因为保留的有效信息与滑动窗口重叠度较高，或者是该策略在当前参数设置下未能显著区分关键信息。

### 3.4 总体对比
*   **吞吐量**: 所有流式方法 (Innovations) 的吞吐量均稳定在 **30 tok/s** 左右，优于 Baseline 的 28.27 tok/s。
*   **Attention 耗时**: 流式方法将 Attention 计算时间控制在常数级 (70-80ms)，而 Baseline 随序列长度增加平均耗时为 109.96ms。

---

## 4. 结论

1.  **Semantic Block (No Comp)**: 在不增加显存开销的前提下，通过语义选择策略取得了比原始 StreamingLLM 更好的 PPL 表现。
2.  **Semantic Block (Compression)**: 提供了更低的显存占用和更快的推理速度，但牺牲了部分 PPL 精度。
3.  **POS Aware**: 在当前测试设置下，表现与原始 StreamingLLM 持平，未展现出明显优势。

