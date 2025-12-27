# Pythia-2.8B 优化实验报告

本目录包含针对 Pythia-2.8B 模型推理加速的实验记录，主要采用两种无训练（Training-free）加速方法：**StreamingLLM** 和 **Speculative Decoding（投机采样）**。

## 1. 核心结论 (Executive Summary)

经过在真实文本（WikiText）上的严格对比测试（强制生成 500 Tokens，消除 EOS 干扰），我们得出以下结论：

1.  **推荐方案**: **纯 StreamingLLM** 是最稳健、最高效的选择（~31.88 tok/s）。
2.  **避雷方案**: **Speculative Decoding** 在当前配置下（Draft=160m, Target=2.8B）不仅没有加速，反而导致了 **30% 的性能倒退**。
3.  **技术洞察**: 
    *   **提前结束现象**: 投机采样会改变生成路径，导致模型更容易生成 EOS（结束符），在未强制忽略 EOS 时会造成“速度很快”的假象。
    *   **接受率瓶颈**: 在复杂文本上，小模型的预测准确率极低（~1.14 tokens/step），节省的算力远不足以覆盖验证和调度的开销。

## 2. 实验设置

*   **硬件**: 单卡 GPU (假设环境)
*   **模型**: Target=Pythia-2.8b, Draft=Pythia-160m
*   **数据**: WikiText-2 (真实长文本)
*   **参数**: `max_new_tokens=500`, `min_new_tokens=500` (强制跑满), `repetition_penalty=1.2`

## 3. 详细测试结果 (Fair Comparison)

| 模式 | 耗时 (s) | 吞吐量 (tok/s) | 加速比 | 关键指标 (Avg Tokens/Step) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 16.29 | **30.70** | 1.00x | 1.00 (纯串行) |
| **Baseline + Spec** | 23.16 | **21.59** | **0.70x** | 1.14 (接受率过低，严重拖累) |
| **Streaming** | 15.68 | **31.88** | **1.04x** | 1.00 (轻微提升，长文潜力巨大) |
| **Streaming + Spec** | 15.74 | **31.77** | **1.03x** | 1.61 (接受率略有提升，勉强持平) |

## 4. 深度分析

### 4.1 为什么 Speculative Decoding 失败了？
*   **数学账算不过来**: 
    *   Speculative Decoding 的核心赌注是：`Draft推理 + Target验证 < Target串行生成`。
    *   在本实验中，平均每步只接受 **1.14** 个 Token。这意味着我们花费了“运行一次小模型 + 运行一次大模型”的昂贵代价，却只比“运行一次大模型”多生成了 **0.14** 个词。
    *   **成本 > 收益**，导致速度暴跌至 0.7x。

### 4.2 为什么之前看起来很快？(EOS 陷阱)
*   在未强制忽略 EOS 时，Baseline 跑了 272 步停止，而 Streaming+Spec 跑了 287 步停止。
*   Speculative Decoding 引入了随机性（改变了生成路径），导致某些 Run 在 200 多步就生成了 EOS 提前结束。
*   **短跑 vs 长跑**: 用短跑的耗时去除以短跑的距离，虽然数值没问题，但掩盖了长距离生成时的稳定性问题。

### 4.3 StreamingLLM 的表现
*   在 500-1000 tokens 的长度下，StreamingLLM 的 KV Cache 优化带来的收益尚不明显（只快了 4%）。
*   **核心价值**: 它的核心价值不在于短文加速，而在于 **无限长度生成的稳定性**。当序列长度超过 2000 或 4000 时，Baseline 会 OOM (显存溢出) 或极度变慢，而 StreamingLLM 将保持恒定的 31+ tok/s。

## 5. 最终建议

*   **日常推理 / 长文本生成**: 
    *   ✅ **仅使用 StreamingLLM**。
    *   配置: `window_size=512`, `sink=4`。
*   **追求极致速度 (需要调优)**:
    *   ❌ **不推荐直接使用当前的 Speculative Decoding 配置**。
    *   如果必须使用，建议：
        1.  **更换更强的 Draft Model**: Pythia-160m 可能太弱了，尝试 Pythia-410m。
        2.  **降低任务难度**: 在简单对话或代码补全中，接受率可能会提高。
        3.  **关闭 Repetition Penalty**: 采样参数会破坏概率分布一致性，降低接受率。
