import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

# ================= 配置 =================
MODEL_PATH = "./models/pythia-70m"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_CAPACITY = 32  # 窗口大小
N_SINK = 4  # Sink 大小
GEN_LEN = 100  # 生成长度
# =======================================

print(f"正在加载模型... 设备: {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
model.eval()


def crop_past_key_values(past_key_values, capacity, n_sink):
    """
    终极版 StreamingLLM 裁剪: 原地修改策略
    直接修改 DynamicCache 对象内部的 tensor，保留对象本身的方法特性
    """
    if past_key_values is None:
        return None

    # === 策略 A: 针对新版 Transformers (DynamicCache 对象) ===
    # 只要它有 key_cache 属性，它就是 DynamicCache (或类似对象)
    if hasattr(past_key_values, "key_cache"):
        # 遍历每一层
        # 注意: key_cache 是一个列表，列表里存的是 Tensor
        for i in range(len(past_key_values.key_cache)):
            k = past_key_values.key_cache[i]
            v = past_key_values.value_cache[i]

            # k shape: [batch, heads, seq_len, dim]
            current_seq_len = k.shape[2]

            # 如果没超标，跳过
            if current_seq_len <= capacity:
                continue

            # === 剪枝逻辑 ===
            window_size = capacity - n_sink

            k_sink = k[:, :, :n_sink, :]
            v_sink = v[:, :, :n_sink, :]

            k_window = k[:, :, -window_size:, :]
            v_window = v[:, :, -window_size:, :]

            k_new = torch.cat([k_sink, k_window], dim=2)
            v_new = torch.cat([v_sink, v_window], dim=2)

            # === 核心操作: 原地替换 ===
            # 我们不创建新对象，而是修改列表中引用的 Tensor
            past_key_values.key_cache[i] = k_new
            past_key_values.value_cache[i] = v_new

        # 返回原始对象 (它现在包含了被剪短的数据)
        return past_key_values

    # === 策略 B: 针对旧版 (Tuple) ===
    else:
        # 获取长度 (假设是 tuple of tuple)
        try:
            current_seq_len = past_key_values[0][0].shape[2]
        except:
            return past_key_values

        if current_seq_len <= capacity:
            return past_key_values

        window_size = capacity - n_sink
        new_past = []
        for layer_idx, (key, value) in enumerate(past_key_values):
            k_sink = key[:, :, :n_sink, :]
            v_sink = value[:, :, :n_sink, :]
            k_window = key[:, :, -window_size:, :]
            v_window = value[:, :, -window_size:, :]
            k_new = torch.cat([k_sink, k_window], dim=2)
            v_new = torch.cat([v_sink, v_window], dim=2)
            new_past.append((k_new, v_new))

        return tuple(new_past)


def run_manual_generation(prompt_text, mode="baseline"):
    print(f"\n>>> 开始测试: [{mode.upper()}]")
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids

    generated_tokens = []
    len_history = []
    start_time = time.time()

    with torch.no_grad():
        # 1. Prefill
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)

        # 2. Decoding Loop
        for i in range(GEN_LEN):

            # === StreamingLLM 介入 ===
            if mode == "streaming":
                # 直接传入 past_key_values 对象进行原地手术
                past_key_values = crop_past_key_values(
                    past_key_values, MAX_CAPACITY, N_SINK
                )

            # === 记录长度 ===
            # 无论什么版本，我们都得想办法读长度
            if hasattr(past_key_values, "get_seq_length"):
                # 新版 DynamicCache 的标准方法
                # 注意：有些版本 get_seq_length 可能返回 seen_tokens 而不是物理长度
                # 所以最稳妥的是直接看 tensor
                if hasattr(past_key_values, "key_cache"):
                    curr_len = past_key_values.key_cache[0].shape[2]
                else:
                    curr_len = past_key_values.get_seq_length()
            else:
                curr_len = past_key_values[0][0].shape[2]

            len_history.append(curr_len)

            # Forward
            # 此时传入的 past_key_values 是原装正版的对象 (只是内部 tensor 变了)
            # 所以 get_mask_sizes 等方法都能正常工作
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)

            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(0)
            generated_tokens.append(next_token.item())

    end_time = time.time()
    throughput = len(generated_tokens) / (end_time - start_time)

    print(f"[{mode}] 最终 Cache 长度: {len_history[-1]}")
    print(f"[{mode}] 吞吐量: {throughput:.2f} t/s")

    return len_history


# ================= 执行 =================
prompt = "Testing the in-place modification strategy for StreamingLLM. " * 5

history_base = run_manual_generation(prompt, mode="baseline")
history_stream = run_manual_generation(prompt, mode="streaming")

# ================= 结果 =================
print("\n" + "=" * 50)
print(f"{'Step':<10} | {'Baseline Len':<15} | {'Streaming Len':<15}")
print("-" * 50)

max_len = max(len(history_base), len(history_stream))
start = max(0, max_len - 20)
for i in range(start, max_len):
    vb = history_base[i] if i < len(history_base) else "-"
    vs = history_stream[i] if i < len(history_stream) else "-"
    print(f"{i:<10} | {vb:<15} | {vs:<15}")
print("=" * 50)

if history_stream[-1] <= MAX_CAPACITY + 1:
    print("\n✅ 成功！StreamingLLM 正常工作！")
else:
    print("\n❌ 失败。")
