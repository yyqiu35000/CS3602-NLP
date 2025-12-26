import os
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from pythia_press_fix_capacity import PythiaStreamingLLMPress

MODEL_PATH = "./models/pythia-70m"
DEVICE = "cuda"

# === 关键设置 ===
# 1. 窗口设得很小，强制模型频繁丢弃 Cache
MAX_CAPACITY = 256
N_SINK = 4

# 2. 生成长度设得很大，远超上面的 256
GEN_LEN = 2000
# ================

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
model.eval()


class SpeedTestStreamer(TextStreamer):
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer, **kwargs)
        self.reset()

    def on_finalized_text(self, text: str, stream_end: bool = False):
        if self.token_count == 0:
            self.first_token_time = time.time()
        self.token_count += 1

    def reset(self):
        self.start_time = 0
        self.first_token_time = 0
        self.token_count = 0


def run_benchmark(name, input_text, gen_len):
    print(f"\n--- 测试: {name} (生成 {gen_len} tokens) ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(DEVICE)

    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)
    streamer = SpeedTestStreamer(tokenizer, skip_prompt=True)
    streamer.start_time = time.time()

    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=gen_len,
            pad_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            use_cache=True,
        )

    end_time = time.time()
    peak_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
    throughput = streamer.token_count / (end_time - streamer.start_time)

    print(f"[{name}] 显存峰值: {peak_mem:.2f} MB")
    print(f"[{name}] 吞吐量: {throughput:.2f} tokens/s")
    return peak_mem, throughput


# 构造长 Prompt
prompt = "This is a story about machine learning. " * 10  # 约 70 tokens

# 1. Baseline
mem_base, speed_base = run_benchmark("Baseline", prompt, GEN_LEN)

# 2. StreamingLLM
press = PythiaStreamingLLMPress(max_capacity=MAX_CAPACITY, n_sink=N_SINK)
print(f"\n>>> 激活 StreamingLLM (Capacity={MAX_CAPACITY})...")
with press(model):
    mem_stream, speed_stream = run_benchmark("StreamingLLM", prompt, GEN_LEN)

print("\n" + "=" * 40)
print(f"{'指标':<10} | {'Baseline':<10} | {'Streaming':<10} | {'节省'}")
print("-" * 50)
print(
    f"Memory     | {mem_base:<10.2f} | {mem_stream:<10.2f} | {mem_base - mem_stream:.2f} MB"
)
print("=" * 40)
