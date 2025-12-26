import os
import time
import torch
import math
from tqdm import tqdm
from calflops import calculate_flops

# 1. 依然要设置镜像，否则下载数据集会报错
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

current_dir = os.getcwd()
os.environ["HF_HOME"] = os.path.join(current_dir, "hf_cache")

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from datasets import load_dataset

# ================= 配置区域 =================
MODEL_PATH = "./models/pythia-2.8b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 2048  # Pythia 的最大上下文窗口
# ===========================================

print(f"检测到的设备: {DEVICE}")
print(f"正在加载模型: {MODEL_PATH} 到 {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"  # 使用半精度加速
)
model.eval()  # 开启评估模式


# -------- 定义计算 PPL 的函数 --------
def calculate_ppl(text, stride=512):
    """
    计算长文本的 PPL (Perplexity)。
    由于文本可能超过模型最大长度，需要用滑动窗口(stride)切分。
    """
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0

    print(f"文本总长度: {seq_len} tokens, 开始计算 PPL...")

    # 这是一个标准的滑动窗口计算 PPL 的循环
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + MAX_LENGTH, seq_len)
        trg_len = end_loc - prev_end_loc  # 这里的逻辑是为了处理重叠部分

        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        target_ids = input_ids.clone()

        # 我们不计算重叠部分的 loss，只计算新部分的
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # outputs.loss 是平均 log-likelihood
            # 我们需要乘回去得到总的 loss
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # 最终计算 PPL = exp(总loss / 总长度)
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.item()


def measure_model_flops(model, tokenizer, device):
    print("\n" + "=" * 20 + " FLOPs 分析 " + "=" * 20)

    # 模拟一个典型的输入场景
    batch_size = 1
    prompt_length = 512

    # 使用 calflops 计算
    # 直接传递 kwargs 而不是让 calflops 自动生成，避免 tokenizer 兼容性问题
    # 生成虚拟输入
    input_ids = torch.ones((batch_size, prompt_length), dtype=torch.long)
    attention_mask = torch.ones((batch_size, prompt_length), dtype=torch.long)

    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}

    flops, macs, params = calculate_flops(
        model=model,
        kwargs=kwargs,
        print_results=False,  # 我们手动打印更清晰
        output_as_string=True,
    )

    print(f"模型参数量: {params}")
    print(f"处理 {prompt_length} tokens 的 Prefill 阶段总计算量: {flops}")

    # 估算平均每个 Token 的 FLOPs
    # 注意：生成阶段因为有 KV Cache，计算量通常显著低于 Prefill 阶段
    avg_token_flops = float(flops.split()[0]) / prompt_length
    print(f"平均每个 Token 的近似计算量: {avg_token_flops:.2f} GFLOPS")

    return flops


# -------- 定义一个用于速度测试的 Streamer --------
class SpeedTestStreamer(TextStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.start_time = 0
        self.first_token_time = 0
        self.token_count = 0

    def on_finalized_text(self, text: str, stream_end: bool = False):
        # 这个方法会在每个 token 生成后被调用
        now = time.time()
        if self.token_count == 0:
            # 这是第一个 token
            self.first_token_time = now
        self.token_count += 1
        # 可以在这里打印生成的 token，如果不想看可以注释掉
        # print(text, end="", flush=True)

    def reset(self):
        self.start_time = 0
        self.first_token_time = 0
        self.token_count = 0


# -------- 【最终版】定义测试生成速度和显存的函数 --------
def test_speed(input_text, generate_len=50):
    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    streamer = SpeedTestStreamer(tokenizer, skip_prompt=True)

    print(
        f"\n开始速度与显存测试 (Prompt: {inputs['input_ids'].shape[1]} tokens, 生成: {generate_len} tokens)..."
    )

    # ---- 显存测量准备 ----
    torch.cuda.empty_cache()  # 清理缓存
    torch.cuda.reset_peak_memory_stats(DEVICE)  # 重置峰值统计

    # 记录调用 generate 前的时间
    streamer.reset()
    streamer.start_time = time.time()

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=generate_len,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
        )

    # 记录 generate 结束后的时间
    end_time = time.time()

    # ---- 获取峰值显存 ----
    peak_memory_bytes = torch.cuda.max_memory_allocated(DEVICE)
    peak_memory_mb = peak_memory_bytes / (1024 * 1024)

    # ---- 计算时间指标 ----
    ttft = streamer.first_token_time - streamer.start_time
    # 避免 token_count 小于等于 1 时的除零错误
    if streamer.token_count > 1:
        tpot = (end_time - streamer.first_token_time) / (streamer.token_count - 1)
    else:
        tpot = float("inf")  # 如果只生成一个 token，则 TPOT 无意义

    throughput = streamer.token_count / (end_time - streamer.start_time)

    print("\n--- 性能指标 ---")
    print(f"峰值显存占用: {peak_memory_mb:.2f} MB")
    print(f"TTFT (Time To First Token): {ttft:.4f}s")
    print(f"TPOT (Time Per Output Token): {tpot * 1000:.2f} ms/token")
    print(f"Throughput (吞吐量): {throughput:.2f} tokens/s")
    print("----------------")
    return ttft, tpot, throughput, peak_memory_mb


# ================= 任务 1: WikiText 测试 =================
print("\n" + "=" * 20 + " 测试 WikiText-2 " + "=" * 20)
# 加载 wikitext-2 测试集
wiki_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
# 为了快速演示，我们把所有测试数据拼成一个长字符串（标准做法之一）
wiki_text = "\n\n".join(wiki_data["text"])

print(f"WikiText 字符数: {len(wiki_text)}")
ppl = calculate_ppl(wiki_text)
print(f"WikiText-2 PPL: {ppl:.2f}")

# 用 wikitext 的一句话来测速
test_speed(wiki_text[:100], generate_len=100)


# ================= 任务 2: PG-19 测试 (单一样本) =================
print("\n" + "=" * 20 + " 测试 PG-19 (单本) " + "=" * 20)
# streaming=True 表示不下载整个数据集（几百G），而是像看视频一样边下边看
pg19_stream = load_dataset(
    "pg19",
    split="test",
    streaming=True,
    trust_remote_code=True,
)

# 取出第一本书作为 sample
book_sample = next(iter(pg19_stream))
book_text = book_sample["text"]

# PG-19 的书可能极长，为了测试不跑太久，我们只取前 10000 个字符做 PPL 测试
# 如果你想测整本书，把 [:10000] 去掉即可
short_book_text = book_text[:10000]

print(f"书名: {book_sample.get('short_book_title', 'Unknown')}")
print(f"截取长度: {len(short_book_text)} 字符 (原书超长)")

# 1. 测 PPL
pg_ppl = calculate_ppl(short_book_text)
print(f"PG-19 Sample PPL: {pg_ppl:.2f}")

# 2. 测速度 (用开头的一句话作为 prompt)
prompt = book_text[:100]
test_speed(prompt, generate_len=100)

# 计算 FLOPs
measure_model_flops(model, tokenizer, DEVICE)

print("\nBaseline 测试完成！")
