import os
import time
import torch
import copy
from tqdm import tqdm

# from calflops import calculate_flops  # 暂时注释掉，加快测试

# 1. 设置镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
current_dir = os.getcwd()
os.environ["HF_HOME"] = os.path.join(current_dir, "hf_cache")

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from datasets import load_dataset

# 引入自己实现的 StreamingLLM Compressor
from pythia_press import PythiaStreamingLLMPress

# ================= 配置区域 =================
MODEL_PATH = "./models/pythia-2.8b"
# MODEL_PATH = "EleutherAI/pythia-70m" # 如果本地没有，可以用这个在线拉取
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 2048
COMPRESSION_RATIO = 0.7  # 压缩率：丢弃 70% 的中间 tokens
N_SINK = 4  # Attention Sink 数量
MAX_CAPACITY = None  # KV Cache 最大容量，None 表示自动计算（512 * 0.3 = 154 tokens）
# ===========================================

print(f"检测到的设备: {DEVICE}")
print(f"正在加载模型: {MODEL_PATH}...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
model.eval()

# -------- 准备数据 --------
print("准备测试数据...")
# 设置为离线模式，使用本地缓存
import datasets

datasets.config.HF_DATASETS_OFFLINE = True

# 1. WikiText
wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
wiki_text = "\n\n".join(wiki_data["text"])

# 2. PG-19 (取样)
pg19_stream = load_dataset("pg19", split="test", streaming=True, trust_remote_code=True)
book_sample = next(iter(pg19_stream))
book_text = book_sample["text"]
short_book_text = book_text[:10000]  # 用于 PPL
prompt_text = book_text[:2000]  # 加长 Prompt（从 200 增加到 2000）


# -------- 定义辅助类 --------
class SpeedTestStreamer(TextStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.reset()

    def on_finalized_text(self, text: str, stream_end: bool = False):
        now = time.time()
        if self.token_count == 0:
            self.first_token_time = now
        self.token_count += 1

    def reset(self):
        self.start_time = 0
        self.first_token_time = 0
        self.token_count = 0


# -------- 核心测试逻辑封装 --------
def calculate_ppl(text, stride=512, use_kv_cache=False):
    """
    计算困惑度 (PPL)

    Args:
        text: 输入文本
        stride: 滑动窗口步长
        use_kv_cache: 是否使用 KV Cache（用于测试 StreamingLLM 的真实影响）
    """
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0

    # 限制一下最大测试长度，避免太慢
    # 使用 KV Cache 模式时大幅减少长度（逐 token 计算很慢）
    if use_kv_cache:
        max_test_len = min(seq_len, 512)  # KV Cache 模式：只测试 512 tokens
    else:
        max_test_len = min(seq_len, 4096)  # 快速模式：测试 4096 tokens

    if not use_kv_cache:
        # 原始方法：不使用 KV Cache（快速但不反映压缩影响）
        for begin_loc in range(0, max_test_len, stride):
            end_loc = min(begin_loc + MAX_LENGTH, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == max_test_len:
                break
    else:
        # 新方法：生成式 PPL 计算（逐 token，累积 past_key_values）
        # 这样 StreamingLLM 的压缩会真正影响后续预测
        print(f"   (生成式计算 {max_test_len} tokens，预计需要 1-2 分钟...)")

        input_ids = encodings.input_ids[:, :max_test_len].to(DEVICE)
        past_key_values = None

        # 逐 token 预测：用 token[0:i] 预测 token[i]
        for i in tqdm(
            range(1, input_ids.size(1)), desc="   计算 PPL", ncols=80, leave=False
        ):
            with torch.no_grad():
                # 输入当前 token[i-1]（配合之前的 past_kv）
                current_input = input_ids[:, i - 1 : i]

                outputs = model(
                    current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

                # 预测 token[i]
                logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
                target = input_ids[:, i]  # [batch]

                # 计算 loss
                loss = torch.nn.functional.cross_entropy(logits, target)
                nlls.append(loss)

                # 更新 past_key_values（会被 StreamingLLM 压缩！）
                past_key_values = outputs.past_key_values

        prev_end_loc = input_ids.size(1) - 1

    if not nlls:
        return 0.0
    ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc)
    return ppl.item()


def test_speed(input_text, generate_len=100):
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    streamer = SpeedTestStreamer(tokenizer, skip_prompt=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    streamer.reset()
    streamer.start_time = time.time()

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=generate_len,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            use_cache=True,
        )

    end_time = time.time()
    peak_memory_bytes = torch.cuda.max_memory_allocated(DEVICE)
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    ttft = streamer.first_token_time - streamer.start_time
    tpot = (
        (end_time - streamer.first_token_time) / (streamer.token_count - 1)
        if streamer.token_count > 1
        else 0
    )
    throughput = streamer.token_count / (end_time - streamer.start_time)

    return {
        "peak_memory_mb": peak_memory_mb,
        "ttft": ttft,
        "tpot_ms": tpot * 1000,
        "throughput": throughput,
    }


# -------- 统一运行函数 --------
def run_benchmark_suite(suite_name, use_kv_cache_for_ppl=False):
    print(f"\n{'#'*20} 开始测试: {suite_name} {'#'*20}")

    # 1. 测试 PPL
    if use_kv_cache_for_ppl:
        print(">>> 计算 WikiText PPL (使用 KV Cache，反映压缩影响)...")
    else:
        print(">>> 计算 WikiText PPL (快速模式)...")
    ppl = calculate_ppl(wiki_text, use_kv_cache=use_kv_cache_for_ppl)
    print(f"[{suite_name}] WikiText PPL: {ppl:.2f}")

    # 2. 测试速度与显存
    print(f">>> 测试生成速度 (Prompt: {len(prompt_text)} chars)...")
    metrics = test_speed(
        prompt_text, generate_len=2000
    )  # 加长生成长度（从 500 增加到 2000）

    print(f"[{suite_name}] 结果:")
    print(f"  - 显存峰值: {metrics['peak_memory_mb']:.2f} MB")
    print(f"  - TTFT: {metrics['ttft']:.4f} s")
    print(f"  - Throughput: {metrics['throughput']:.2f} tokens/s")
    print(f"  - TPOT: {metrics['tpot_ms']:.2f} ms")

    return {"ppl": ppl, **metrics}


# ================= 主程序执行 =================

results = {}

# 1. 运行 Baseline (无压缩)
print("\n正在运行 Baseline...")
results["Baseline"] = run_benchmark_suite("Baseline", use_kv_cache_for_ppl=True)

# 2. 运行 StreamingLLM
print(f"\n正在运行 StreamingLLM (压缩率 {COMPRESSION_RATIO})...")
# 初始化自定义的 PythiaStreamingLLMPress
press = PythiaStreamingLLMPress(
    compression_ratio=COMPRESSION_RATIO, n_sink=N_SINK, max_capacity=MAX_CAPACITY
)

# 使用 context manager: with press(model):
# 在这个块内，模型所有的 forward 都会自动应用 StreamingLLM 策略
with press(model):
    results["StreamingLLM"] = run_benchmark_suite(
        "StreamingLLM", use_kv_cache_for_ppl=True
    )

# ================= 最终对比报表 =================
print("\n" + "=" * 40)

# 如果只有 StreamingLLM 结果，直接打印而不做对比
if "StreamingLLM" in results and "Baseline" not in results:
    print(f"StreamingLLM (压缩率 {COMPRESSION_RATIO}) 测试结果:")
    print("-" * 40)
    metrics = results["StreamingLLM"]
    print(f"  PPL: {metrics['ppl']:.2f}")
    print(f"  显存峰值: {metrics['peak_memory_mb']:.2f} MB")
    print(f"  TTFT: {metrics['ttft']:.4f} s")
    print(f"  Throughput: {metrics['throughput']:.2f} tokens/s")
    print(f"  TPOT: {metrics['tpot_ms']:.2f} ms")
    print("=" * 40)
else:
    # 如果有 Baseline，进行对比
    print(f"{'指标':<15} | {'Baseline':<12} | {'StreamingLLM':<12} | {'变化':<10}")
    print("-" * 55)

    keys_to_compare = [
        ("ppl", "PPL (Lower is better)", "{:.2f}"),
        ("peak_memory_mb", "Memory (MB)", "{:.2f}"),
        ("throughput", "Throughput (t/s)", "{:.2f}"),
        ("ttft", "TTFT (s)", "{:.4}"),
        ("tpot_ms", "TPOT (ms)", "{:.2f}"),
    ]

    for key, label, fmt in keys_to_compare:
        base_val = results["Baseline"][key]
        stream_val = results["StreamingLLM"][key]

        # 计算变化率
        if key == "ppl" or key == "tpot_ms" or key == "peak_memory_mb" or key == "ttft":
            # 越低越好
            delta = (stream_val - base_val) / base_val * 100
            change_str = f"{delta:+.1f}%"
        else:
            # 越高越好
            delta = (stream_val - base_val) / base_val * 100
            change_str = f"{delta:+.1f}%"

        print(
            f"{label:<15} | {fmt.format(base_val):<12} | {fmt.format(stream_val):<12} | {change_str:<10}"
        )

    print("=" * 40)
