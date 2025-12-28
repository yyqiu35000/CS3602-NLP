# StreamingLLM Implementation on Pythia-2.8B

本项目是对 **StreamingLLM** 算法在 **Pythia-2.8B** 模型上的复现与实验。项目采用侵入式 Patch 方案，通过自定义 `StreamingDynamicCache` 接管 KV Cache 管理，实现了 Attention Sink + Sliding Window 机制。

##  项目结构

为了保持根目录整洁，项目核心代码与实验性代码进行了分离：

### 1. 核心运行文件 (根目录，个人作业)

| 文件名 | 说明 |
| :--- | :--- |
| **`main.py`** | **项目主入口**。负责加载模型、数据集，并调度评估任务（PPL测试、速度测试）及调试模式。 |
| **`pythia_streaming_patch.py`** | **核心实现**。包含 `StreamingDynamicCache` 类（实现 Sink+Window 驱逐策略）和 Monkey-Patching 逻辑。 |
| **`requirements.txt`** | 项目运行所需的 Python 依赖库。 |

### 2. 实验与探索 (子目录，小组作业)

| 目录 | 说明 |
| :--- | :--- |
| **`flash/`** | **性能优化 (FlashAttention)**。对比了 Eager 与 SDPA (FlashAttention) 模式，验证了 SDPA 对 StreamingLLM 的加速效果及 BF16 对数值稳定性的重要性。 |
| **`quantization/`** | **量化推理 (Quantization)**。探索了 Int8 和 NF4 量化在流式场景下的表现，实现了单卡 2GB 显存运行 3B 模型的高效推理。 |
| **`compress/`** | **创新压缩 (Innovations)**。基于 POS (词性) 和 Semantic Block (语义块) 的 KV Cache 压缩策略，旨在比简单滑动窗口保留更多语义信息。 |
| **`h2o/`** | **混合策略 (H2O + Streaming)**。结合了 H2O (Heavy Hitter Oracle) 的重要性筛选机制，显著提升了长文本 PPL 并突破了模型训练长度限制。 |
| **`debug_streaming/`** | **底层验证**。包含用于验证 Attention Mask 结构、RoPE 维度冲突等底层逻辑的独立测试脚本 (如 `verify_mask_logic.py`)。 |
| **`note/`** | **早期验证代码**。包含最初的非侵入式实现版本，仅作为原理参考。 |

## 个人完成部分说明

### 使用说明

#### 环境准备

您可以选择以下任意一种方式配置环境：

**方式一：使用 requirements.txt (推荐)**

```bash
pip install -r requirements.txt
```

**方式二：手动安装核心依赖**

如果requirements安装失败，可以尝试手动安装各个包：

```bash
# 安装 PyTorch (根据您的 CUDA 版本选择，这里以 CUDA 12.8 为例)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# 安装 Transformers 及其他 NLP 相关库
pip install transformers datasets accelerate huggingface_hub
```

注：datasets版本应为2.x，最新版本可能会导致PG-19数据集加载失败。

#### 1. 标准评估模式
运行完整的 PPL (困惑度) 测试和生成速度测试，对比 Baseline 与 StreamingLLM 的性能。

```bash
python main.py
```
**输出内容**：
- 自动运行 Baseline, Streaming (Window=256), Streaming (Window=512) 等多组配置。
- 在 **Wikitext** 和 **PG-19** 数据集上评估 PPL。
- 测试 TTFT (首字延迟)、TPOT (生成耗时)、Throughput (吞吐量) 和 Peak Memory。
- 最终生成 Markdown 格式的对比表格。

#### 2. 调试模式
启动调试模式，打印每100token KV Cache 的形状变化，用于验证 Sink 和 Window 机制是否正常工作。
```bash
python main.py test
```


**输出内容**：
- 运行单次 Streaming 生成任务。
- **实时日志**：每 100 步或发生驱逐 (Eviction) 时，打印当前 KV Cache 的形状。
- **验证点**：你可以清晰地看到 Cache 大小在达到 `Limit + Buffer` 后被压缩回 `Limit`，证明 Sink 机制和滑动窗口正在工作。

#### 3. 参数调整

在main.py全局变量中

- `model_id`:使用的模型
- `ppl_tokens`: 预测ppl使用的token数
- `speed_tokens`: 预测生成对应数量的token
- `Pre_tokens`:测速中预先给出的token数
- `config`:streamingLLM参数设置


### 早期探索：非侵入式实现 (test.ipynb)

在最终采用 Monkey-Patching 方案之前，我们首先在 note下的`test.ipynb`/`test.py + utils.py` 中尝试了一种**非侵入式**的实现方案。

- **实现方式**：不修改模型源码，而是编写一个独立的 `StreamingLLM` 类，手动接管 `generate` 循环。在每一步生成后，显式调用 `compress_cache` 函数对 `past_key_values` 进行裁剪。
- **压缩策略**：最初使用了 PyTorch 的 `index_select` 方法来提取 Sink 和 Window 区域的 Token。
- **局限性**：
  - **生态兼容性差**：难以直接复用 HuggingFace 现有的复杂解码策略（如 Beam Search, Top-k/p Sampling 等），需要重写整个生成逻辑。
  - **效率瓶颈**：Python层面的手动循环和频繁的 `index_select` 显存拷贝导致了较大的额外开销。
- **演进**：这一阶段的探索帮助我们理清了 StreamingLLM 的核心逻辑，促使我们最终转向了更高效、兼容性更好的侵入式 Patch 方案 (`pythia_streaming_patch.py`)，即直接修改底层 Attention 的 Cache 更新行为。

---

###  实现原理与优化

#### 1. 核心算法：Sink + Sliding Window
StreamingLLM 的核心在于保留序列开头的初始 Tokens (Sink) 作为“注意力锚点”，同时仅保留最近的窗口 (Window)。

#### 2. Lazy Eviction & Buffer 机制
在 Python/PyTorch 中，如果严格按照StreamingLLM的设计，频繁的张量拼接 (`torch.cat`) 和显存分配开销巨大。为了解决这个问题，我们采用了 **Lazy Eviction** 策略：
- **Buffer 区**：我们不强制每一步都维持严格的 `window_size`。而是允许 Cache 作为一个 "Buffer" 增长。
- **批量驱逐**：只有当 Cache 大小超过 `Limit + 64` (Buffer 大小) 时，才触发一次驱逐操作，将大小重置为 `Limit`。
- **优势**：将昂贵的 Tensor 操作频率降低了几十倍，显著减少了 Python 带来的额外开销。

#### 3. 纯 Attention 计时 (Attention Timing Isolation)
为了在 Python 层面准确验证 StreamingLLM 的 O(1) 复杂度优势（排除 Python 循环和 KV 搬运的干扰），我们在 `pythia_streaming_patch.py` 中实现了针对 Attention 层的**独立计时器**。
- 通过 `torch.cuda.synchronize()` 确保计时准确。
- 仅统计 `Forward` 中 Attention 计算的核心耗时，证明了随着序列变长，StreamingLLM 的计算耗时保持恒定。

---

### 遇到的 Bug 与修复思路

在复现过程中，我遇到并解决了以下问题：

#### Bug 1: Cache 已压缩但 PPL 异常爆炸
**现象**：启用了 Streaming，窗口也限制住了（我检查了chunk间的kv大小，在StreamingLLM中都是符合预期的，也与baseline的前sink+后windows一致），但 PPL 高达 300+（正常应为 40-70）。
**原因**：**Eager Eviction (急切驱逐)**。最初的实现中，我们在 `update` 时先将新 Token 加入，然后**立即驱逐**旧 Token，并返回驱逐后的 Cache 给当前层计算。这意味着当前步生成的 Token 无法“看到”它刚刚生成的最近几个 Token（因为被切掉了），导致上下文断裂。
**修复**：改为 **Lazy Eviction**。`update` 函数返回**完整**的 Cache 供当前步计算 Attention，计算完成后（或在准备下一步时）再更新底层的存储状态。这确保了当前步的计算拥有完整的局部上下文。

#### Bug 2: ppl计算中，调试日志显示 Cache 尺寸远超 Window Size
**现象**：设置 Window=256，但日志显示 Cache 增长到了 700 多才变小。
**原因**：这是 PPL 评估时的 **Chunk-wise Processing** 导致的。为了加速 PPL 计算，我们一次性输入 `chunk_size=512` 个 Token。在处理这个 Chunk 时，Cache 会暂时容纳这些 Token 直到 Chunk 结束才触发驱逐。
**修复/验证**：
- 这是一个 Feature 而非 Bug，是为了计算效率的权衡。
- 在 `debug_test_mechanics` (调试模式) 中，我们将 PPL 测试的 `chunk_size` 调小为 64，从而在日志中清晰地展示出平滑、稳定的驱逐行为 (e.g., `329 -> 264`)，验证了机制的正确性。

#### Bug 3: Streaming 速度反而比 Baseline 慢
**现象**：理论上 Streaming 是 O(1)，但实测 Python 实现的 Streaming 速度没有显著提升，甚至略慢。
**原因**：Python 层面的 `torch.cat` 和切片操作带来的 CPU 开销掩盖了 GPU 上的计算节省（特别是在序列长度还不够长，如 < 4k 时）。
**解决**：
1. 引入 **Buffer (64 tokens)** 减少 `cat` 频率。
2. 重点关注 **Avg Attention Time** 指标而非端到端时间。实测显示 Streaming 模式下 Attention 计算时间稳定在 ~0.05ms，而 Baseline 会随长度线性/二次增长，这验证了算法理论上的成功。

#### Bug 4: RoPE 绝对位置与物理 Cache 维度的核心冲突
**现象**：出现ppl爆炸
| Configuration        | Wikitext PPL | PG-19 PPL  | Total Time (s) | Avg Attn (ms)  | TTFT (s)   | TPOT (ms)  | Throughput (tok/s) | Peak Mem (GB) |
| :---                 | :---         | :---       | :---           | :---           | :---       | :---       | :---               | :---          |
| **baseline**         | 7.97         | 8.63       | 77.87          | 164.49         | 0.1092     | 38.93      | 25.69              | 5.94          |
| **streaming_8_256**  | 71.67        | 47.12      | 63.56          | 73.19          | 0.1216     | 31.78      | 31.47              | 5.31          |
| **streaming_8_512**  | 41.28        | 35.22      | 66.71          | 94.64          | 0.1183     | 33.36      | 29.98              | 5.36          |

在引入 `chunk_size > 1` (如 Chunked Prefill) 时，ppl计算速度大幅增加，但ppl突然增大，其原因涉及我们侵入式的底层原理：我们直接对kv cache进行了切片，这时候原来判断cache长度的函数就会失效，即`get_seq_length`返回的是物理长度，而不是逻辑长度。这会导致RoPE计算错误，从而导致ppl增大。而我们第一版实现中修改了`get_seq_length`函数，使其返回逻辑长度。但这也引入了第二个问题：`attn_mask`错误,因为模型误把逻辑长度理解成了现有缓存长度，而这远大于当前所有的kv长度（缓存+新输入的），所以所有token都变成了可见

例：chunksize为10，window=10，sink=10时，理论上我们处理第31个token时他应该只能看到0~10以及21~30这20个token，可由于`get_seq_length`已知现在在第三十个token，而加上新输入的10个token后总的kv也才30（因为有10个被裁掉了），所以模型会错误地认为当前token可以看到所有token（即他看到了0~10以及21~30这30个token）。

因此系统面临两难困境：
1.  **保 RoPE**：如果让 `get_seq_length` 返回真实的逻辑长度 (e.g., 1000) 以确保 RoPE 计算正确的旋转角度，HuggingFace 的默认逻辑会生成一个 `[1, 1, 10, 1000]` 的巨大 Mask，与此时仅有 `24` (Sink+Window) 长度的物理 KV Cache 发生维度不匹配报错。
2.  **保 Mask**：如果让 `get_seq_length` 返回物理长度 (e.g., 24) 以适配矩阵乘法，RoPE 会错误地将当前 Token 当作序列开头的 Token (位置 0-23)，彻底破坏长文本的位置信息。

显然后者严重性低一些，因为他只影响批量验证（对于逐token生成，因为每次都只有一个token，所以mask正常情况下也都是True，不会有问题）

**排查方式**：
- 建立了独立的 `debug_streaming` 环境，复刻了 `main.py` 和 `patch` 逻辑。
- 在 `Attention.forward` 中植入探针，打印 Query, Key, Mask 的 Shape 以及 `get_seq_length` 的返回值。
- 观察到当 `Chunk Size=10` 时，逻辑长度 (30) 与物理 Key 长度 (24) 的脱节导致了 Mask 维度的各种异常。

**解决方案**：**“逻辑与物理分离”**
1.  **RoPE 层面**：坚持让 `get_seq_length` 返回**逻辑长度** (Logical Length)，确保 RoPE 拿到绝对位置索引，保证旋转正确。
2.  **Mask 层面**：**弃用 Transformers 默认 Mask 生成逻辑**。
3.  **手动构建 Mask**：在 Patch 代码中，基于物理 Cache 的结构 (Sink + Window + Current Chunk) 手动构建一个形状为 `[Batch, 1, Chunk_Len, Physical_Cache_Len]` 的 Mask。
    - 确保当前 Token 只能看到 Sink 区、Window 区以及 Chunk 内部它之前的 Token。
    - 这种方案完美调和了 RoPE 对“绝对位置”的需求和 Attention 计算对“物理维度”的限制。

---

### 评估函数原理

#### 1. Chunk-wise Stateful PPL
为了高效评估长文本 PPL，我们实现了 `evaluate_ppl_unified`：
- 将长文本切分为多个 Chunk (128)。
- 顺序处理每个 Chunk，并传递 `past_key_values`。
- **动态窗口**：在处理一个 Chunk 期间，Cache 大小会从 `Window` 增长到 `Window + Chunk`。这意味着模型看到的平均上下文长度实际上略大于设定的 `Window Size`，这有助于获得更好且更快的 PPL 结果。

#### 2. Speed Benchmark
- **TTFT (Time to First Token)**: 首字生成延迟。
- **TPOT (Time per Output Token)**: 生成阶段每 Token 耗时。
- **Peak Memory**: 显存峰值。StreamingLLM 应能显著降低此指标（在超长文本下）。

---

### 实验结果分析

以下结果基于 **Pythia-2.8B** 在 **RTX 4060 Ti (16GB)** 上的实测数据。考虑到模型训练长度限制，评估控制在 2048 tokens 以内。

#### 实验 1: 短序列评估
**设置**：`ppl_tokens=1000`, `prompt_len=500`, `gen_len=500`

| Configuration | Wikitext PPL | PG-19 PPL | Total Time (s) | Avg Attn (ms) | TTFT (s) | TPOT (ms) | Throughput (tok/s) | Peak Mem (GB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 6.97 | 8.57 | 18.06 | 0.1159 | 0.1158 | 36.12 | 27.69 | 5.48 |
| **Streaming (Win=256)** | 11.49 | 8.70 | 17.03 | 0.0847 | 0.1191 | 34.07 | 29.35 | 5.31 |
| **Streaming (Win=512)** | 7.72 | 8.52 | 17.88 | **0.0899** | 0.1225 | 35.76 | **29.18** | 5.37 |

*(注：Streaming 模式的 Avg Attn 已经过最新优化，通过跳过 Decoding 阶段的 Mask 构建，实现了 ~90us 的极速 Attention)*

#### 实验 2: 长序列评估
**设置**：`ppl_tokens=2000`, `prompt_len=500`, `gen_len=1500`

| Configuration | Wikitext PPL | PG-19 PPL | Total Time (s) | Avg Attn (ms) | TTFT (s) | TPOT (ms) | Throughput (tok/s) | Peak Mem (GB) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** | 7.96 | 8.66 | 57.20 | 0.1508 | 0.1126 | 38.13 | 26.22 | 5.79 |
| **Streaming (Win=256)** | 12.39 | 8.87 | 61.31 | 0.0993 | 0.1278 | 40.88 | 24.46 | 5.31 |
| **Streaming (Win=512)** | 9.46 | 8.70 | 56.22 | 0.1128 | 0.1272 | 37.48 | 26.68 | 5.37 |

### 结果深度分析

#### 1. 速度与计算开销 (Speed & Computation)
*   **真正的 O(1) Attention**：
    *   最新的优化（Decoding 阶段跳过 Mask 构建）解决了早期实现中 Avg Attn 偏高的问题。
    *   在短序列实验中，Streaming (Win=512) 的 **Avg Attn Time (0.0899 ms)** 显著低于 Baseline (**0.1159 ms**)，降幅达 **22%**。
    *   这证明了 StreamingLLM 不仅理论上是 O(1)，在工程上也成功实现了比 Eager 模式更快的单步计算速度。
*   **端到端性能 (Total Time)**：
    *   得益于更快的 Attention 和轻量的 Cache 管理，StreamingLLM 在短序列下的 **Throughput (29.18 tok/s)** 也成功反超了 Baseline (27.69 tok/s)。
    *   在长序列下，虽然 Python 层面的 KV 搬运开销仍有一定影响，但随着长度进一步增加，Baseline 的 O(N^2) 瓶颈将导致其速度急剧下降，而 StreamingLLM 将保持恒定高速。

#### 2. 模型质量 (PPL / Quality)
*   **Window Size 的权衡**：
    *   **Window=256**：PPL 明显恶化 (Wikitext 6.97 -> 11.49)，说明 256 窗口对于 Pythia-2.8B 过小，丢失了关键上下文。
    *   **Window=512**：PPL 表现优异。在短序列上 (6.97 vs 7.72) 差距很小，在 PG-19 数据集上甚至略优 (8.57 vs 8.52)。这有力地证明了 "Sink + Sliding Window" 策略能够有效捕获长文本的核心语义。

#### 3. 显存占用 (Memory Usage)
*   StreamingLLM 成功将显存锁定。在 2000 tokens 生成中，显存稳定在 **5.3 GB** 左右，而 Baseline 已经增长到 **5.79 GB**。对于无限长度生成，Baseline 必然 OOM，而 StreamingLLM 将永远稳定运行。


## 团队完成部分说明

本部分简要总结了在优化、压缩、量化及混合策略方面的探索成果。

### 1. 性能优化 (FlashAttention) - `flash/`
*   **SDPA 加速**: 验证了 PyTorch 2.0+ `SDPA` (Scaled Dot Product Attention) 对流式生成的巨大提升。测试显示，`StreamingLLM + SDPA` 的吞吐量提升了 **65%** (30.77 tok/s)，Attention 计算耗时降低了 **79%** (70ms)。
*   **数值稳定性**: 解决了 Eager 模式下 FP16 的数值溢出问题，确认了使用 `BF16` (BFloat16) 对于保证大模型推理稳定性的必要性。

### 2. 量化推理 (Quantization) - `quantization/`
*   **极致显存优化**: 引入 `NF4` (4-bit Normal Float) 量化，将 2.8B 模型的推理显存占用从 5.5GB 压缩至 **1.85GB**，降幅达 **66%**，使得在消费级显卡上运行大模型成为可能。
*   **速度与质量平衡**: NF4 量化在大幅降低显存的同时，保持了较高的生成速度 (19 tok/s) 和可接受的 PPL 损失。

### 3. 创新压缩策略 (Innovations) - `compress/`
*   **语义感知缓存**: 提出了 **Semantic Block** 策略，利用语义相似度（Cosine Similarity）而非单纯的时间顺序来保留 KV Block。实验表明，在相同 KV 大小下，该策略的 PPL (10.61) 优于原始 StreamingLLM (11.49)。
*   **词性过滤**: 尝试了 **POS Aware** 策略，优先保留实词（名词、动词）而驱逐停用词，探索了基于语言学特征的压缩方向。

### 4. H2O 混合机制 (H2O + Streaming) - `h2o/`
*   **Heavy Hitter Oracle**: 实现了 H2O 算法，通过累积 Attention Score 动态识别并保留“重关注”Token (Heavy Hitters)。
*   **长文本突破**: H2O 机制不仅显著降低了 PPL (接近 Baseline)，更成功实现了突破模型训练长度限制的生成 (稳定生成 10000+ tokens)，证明了动态稀疏注意力的强大潜力。
