# H2O + StreamingLLM 实现说明

本文档说明如何使用 H2O (Heavy Hitter Oracle) 与 StreamingLLM 的组合实现。

## 📋 文件说明

### 核心实现文件

| 文件名                          | 说明                                                                                                          |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `pythia_streaming_h2o_patch.py` | **H2O 核心实现**。包含 `H2ODynamicCache` 类（实现 Sink + Heavy Hitters + Recent Window 策略）和反馈闭环机制。 |
| `bench_streaming_h2o.py`        | **标准评测脚本**。对比 Baseline、StreamingLLM、H2O 的性能（PPL、速度、显存）。                                |
| `long_context_stress_test.py`   | **长文本压力测试**。验证 H2O 突破模型位置编码限制的能力。                                                     |

## 🚀 快速开始

### 1. 环境准备

```bash
pip install -r requirements.txt
```

需要的核心依赖：
- `transformers >= 4.30.0`
- `datasets >= 2.0.0`
- `torch >= 2.0.0`
- `accelerate`

### 2. 运行标准评测

对比 Baseline、StreamingLLM 和 H2O 的性能：

```bash
python bench_streaming_h2o.py
```

**测试内容**：
- 在 WikiText-2 和 PG-19 数据集上评估困惑度（PPL）
- 测量生成速度、吞吐量、显存占用
- 测试多种配置：
  - `baseline`: 标准全量 KV Cache
  - `streaming_8_256`: StreamingLLM (Sink=8, Window=256)
  - `h2o_8_32_256`: H2O (Sink=8, Recent=32, Capacity=256)
  - `h2o_8_64_512`: H2O (Sink=8, Recent=64, Capacity=512)

### 3. 运行长文本压力测试

验证 H2O 在长文本生成中的优势：

```bash
python long_context_stress_test.py
```

**测试内容**：
- 渐进式长度测试（1000 → 10000 tokens）
- 验证位置编码突破能力
- 对比 Baseline 和 H2O 的稳定性

## 🔬 核心算法：H2O (Heavy Hitter Oracle)

### 算法原理

H2O 通过动态选择最重要的 Key-Value pairs 来压缩 KV Cache：

```
[Sink Tokens] + [Heavy Hitters] + [Recent Window]
     ↓               ↓                   ↓
   固定保留      TopK 选择            滑动窗口
```

### 与 StreamingLLM 的区别

| 策略         | StreamingLLM                | H2O                                         |
| ------------ | --------------------------- | ------------------------------------------- |
| **保留方式** | Sink + Recent Window (固定) | Sink + Heavy Hitters + Recent Window (动态) |
| **选择依据** | 位置（时间）                | 注意力权重（重要性）                        |
| **计算开销** | 低                          | 中等（需要 TopK）                           |
| **质量**     | 中等                        | 更高                                        |

### 反馈闭环机制

H2O 通过 **Attention Weights → Cache Update** 的反馈闭环来动态选择重要 tokens：

1. **收集阶段**：在 Attention 计算时，记录每个 Key 被关注的累积权重
2. **选择阶段**：当 Cache 超过容量时，使用 TopK 选出权重最高的 tokens
3. **更新阶段**：保留 [Sinks + Heavy Hitters + Recent] 并同步更新分数

## 📊 实验结果

### 标准评测（1000 tokens 生成）

| 配置             | WikiText PPL | 吞吐量 (tok/s) | 峰值显存 (GB) |
| ---------------- | ------------ | -------------- | ------------- |
| baseline         | 6.99         | 26.51          | 5.48          |
| streaming_8_256  | 32.24        | 28.01          | 5.31          |
| **h2o_8_32_256** | **13.55**    | **24.16**      | **5.36**      |
| **h2o_8_64_512** | **6.98**     | **24.42**      | **5.36**      |

**关键发现**：
- H2O 显著改善 PPL（32.24 → 13.55，提升 58%）
- h2o_8_64_512 配置达到接近 baseline 的质量，同时节省显存

### 长文本压力测试

| 配置             | 生成长度  | 吞吐量 (tok/s) | 状态         |
| ---------------- | --------- | -------------- | ------------ |
| baseline         | 1000      | 20.93          | ✅            |
| baseline         | 1500      | 22.74          | ✅ (接近极限) |
| **h2o_8_32_256** | **1000**  | **24.57**      | ✅            |
| **h2o_8_32_256** | **5000**  | **28.07**      | ✅            |
| **h2o_8_32_256** | **10000** | **27.29**      | ✅            |

**关键发现**：
- **突破位置编码限制**：Pythia-2.8b 训练最大长度 2048，但 H2O 可稳定生成 10000+ tokens
- **性能优势显现**：在长文本场景下，H2O 吞吐量比 baseline 高 23.4%
- **显存恒定**：H2O 显存保持在 5.28GB，不随生成长度增长

## 🔧 配置参数说明

### H2O 参数

```python
enable_h2o_llm(
    model,
    n_sink=8,           # Sink tokens 数量（初始注意力锚点）
    recent_window=32,    # Recent window 大小（最新 tokens）
    max_capacity=256,    # 总 KV Cache 容量
    debug=False          # 是否打印调试信息
)
```

**参数计算**：
```
Heavy Hitters 数量 = max_capacity - n_sink - recent_window
                  = 256 - 8 - 32 = 216
```

### 推荐配置

| 场景         | 配置           | 说明                           |
| ------------ | -------------- | ------------------------------ |
| **平衡性能** | `h2o_8_32_256` | 适合大多数场景，速度快         |
| **高质量**   | `h2o_8_64_512` | 接近 baseline 质量，显存仍可控 |
| **极限压缩** | `h2o_4_16_128` | 最小显存占用，质量有损         |

## 💡 使用建议

### 何时使用 H2O？

✅ **推荐场景**：
- 长上下文生成（> 2000 tokens）
- 需要突破模型训练长度限制
- 对生成质量有较高要求
- 显存受限但需要处理长文本

❌ **不推荐场景**：
- 短文本生成（< 500 tokens），TopK 开销不值得
- 对速度极度敏感的实时应用
- 已有充足显存的场景

### 与 StreamingLLM 选择对比

| 特性           | StreamingLLM         | H2O                |
| -------------- | -------------------- | ------------------ |
| **实现复杂度** | 简单                 | 中等               |
| **计算开销**   | 低                   | 中等               |
| **质量**       | 中等                 | 更高               |
| **适用场景**   | 超长文本、低质量要求 | 长文本、高质量要求 |

## 🛠️ API 使用示例

### 基础使用

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from pythia_streaming_h2o_patch import enable_h2o_llm

# 加载模型
model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-2.8b",
    torch_dtype=torch.float16,
    device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-2.8b")

# 启用 H2O
enable_h2o_llm(model, n_sink=8, recent_window=32, max_capacity=256)

# 正常使用生成
inputs = tokenizer("Your prompt here", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=1000)
```

### 切换配置

```python
# 切换到 StreamingLLM
from pythia_streaming_h2o_patch import enable_streaming_llm
enable_streaming_llm(model, n_sink=8, window_size=256)

# 切换回 Baseline
from pythia_streaming_h2o_patch import disable_streaming_llm
disable_streaming_llm(model)

# 重新启用 H2O（不同参数）
enable_h2o_llm(model, n_sink=8, recent_window=64, max_capacity=512)
```

## 📝 实现细节

### 反馈闭环流程

```python
# 1. Attention Forward 中强制输出权重
force_output_attentions = output_attentions or hasattr(layer_past, "update_scores")

# 2. 计算 Attention
attn_output, attn_weights = attention_interface(...)

# 3. 反馈到 Cache
if hasattr(layer_past, "update_scores") and attn_weights is not None:
    layer_past.update_scores(attn_weights, self.layer_idx)

# 4. Cache 在 update() 时执行 TopK 选择
```

### Lazy Eviction 策略

为了避免频繁的 TopK 计算和内存拷贝：

```python
if current_len > max_capacity + 64:  # 超过容量 + Buffer
    # 执行驱逐
    heavy_hitters = topk_select(scores)
    keep_indices = [sinks, heavy_hitters, recent]
    cache = cache[:, :, keep_indices, :]
```

## 🤝 贡献

H2O 实现基于以下工作：
- **论文**: [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)
- **StreamingLLM**: [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)

## 📄 许可证

本项目遵循 MIT 许可证。
