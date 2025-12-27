import os
import torch
import time
import math
import gc
import sys
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from pythia_streaming_h2o_patch import (
    enable_streaming_llm,
    disable_streaming_llm,
    enable_h2o_llm,
    reset_attention_timing,
    enable_attention_timing_collection,
    disable_attention_timing_collection,
    get_attention_stats,
    patch_attention_layers,
    get_raw_attention_times,
)

# HuggingFace 配置（可选：设置镜像加速）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

configs = [
    {"name": "baseline", "type": "baseline"},
    {"name": "streaming_8_256", "type": "streaming", "sink": 8, "window": 256},
    {"name": "streaming_8_512", "type": "streaming", "sink": 8, "window": 512},
    # H2O 配置：max_capacity=256, 其中 4 个 Sink + 32 个 Recent + 220 个 Heavy Hitters
    {"name": "h2o_4_32_256", "type": "h2o", "sink": 4, "recent": 32, "capacity": 256},
    {"name": "h2o_8_32_256", "type": "h2o", "sink": 8, "recent": 32, "capacity": 256},
    # 更大容量的 H2O 配置
    {"name": "h2o_8_64_512", "type": "h2o", "sink": 8, "recent": 64, "capacity": 512},
]
# 使用 HuggingFace 模型
model_id = "EleutherAI/pythia-2.8b"
device = "cuda"
ppl_tokens = 1000
speed_tokens = 1000
Pre_tokens = 500


def load_long_text(dataset_name="wikitext", split="test", limit_chars=50000):
    """从 HuggingFace 加载数据集"""
    print(f"正在加载 {dataset_name}...")
    try:
        if dataset_name == "wikitext":
            # 从 HuggingFace 加载 wikitext-2-raw-v1
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            text = "\n\n".join(ds["text"])
            print(f"  -> 成功加载 {len(ds)} 条记录")

        elif dataset_name == "pg19":
            # 从 HuggingFace 加载 PG-19
            ds = load_dataset("deepmind/pg19", split=split, streaming=True)
            # 取前几个样本拼接
            texts = []
            for i, example in enumerate(ds):
                texts.append(example["text"])
                if i >= 2:  # 取前3个样本
                    break
            text = "\n\n".join(texts)
            print(f"  -> 成功加载 PG-19 样本")
        else:
            text = "Long text placeholder. " * 1000
    except Exception as e:
        print(f"警告: 加载 {dataset_name} 失败 ({e}), 使用占位文本。")
        text = "This is a dummy text for fallback. " * 1000

    if len(text) < limit_chars:
        text = text * math.ceil(limit_chars / len(text))
    return text[:limit_chars]


def evaluate_ppl_unified(
    model, tokenizer, text: str, max_tokens: int = 2000, chunk_size: int = 512
):
    """
    统一的 PPL 评估函数，支持 Baseline 和 StreamingLLM。
    使用分块处理 (chunk-wise) 来模拟流式输入或滑动窗口。
    """
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[:, :max_tokens].to(model.device)
    seq_len = input_ids.size(1)

    nlls = []
    token_counts = []
    past_key_values = None

    # print(f"  > PPL 评估: {seq_len} tokens, chunk_size={chunk_size}")

    for i in range(0, seq_len, chunk_size):
        chunk_input_ids = input_ids[:, i : i + chunk_size]
        chunk_target = chunk_input_ids.clone()

        position_ids = torch.arange(
            i,
            i + chunk_input_ids.size(1),
            dtype=torch.long,
            device=chunk_input_ids.device,
        )
        position_ids = position_ids.unsqueeze(0)

        with torch.no_grad():
            outputs = model(
                chunk_input_ids,
                labels=chunk_target,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
            )

            loss = outputs.loss
            past_key_values = outputs.past_key_values

            # StreamingLLM: 手动触发驱逐以在 PPL 评估期间模拟窗口限制
            if hasattr(past_key_values, "evict_all_layers"):
                past_key_values.evict_all_layers()

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            nlls.append(loss)
            token_counts.append(chunk_input_ids.size(1))

    if not nlls:
        return float("inf")

    total_loss = sum(l * c for l, c in zip(nlls, token_counts))
    total_tokens = sum(token_counts)
    return torch.exp(total_loss / total_tokens).item()


def benchmark_generation_speed(model, tokenizer, prompt, num_tokens=500, verbose=False):
    """基准测试生成速度和 Attention 计算时间"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    if verbose:
        print(f"   Prompt tokens: {input_len}, Requesting {num_tokens} tokens")

    # 预热 (Warmup)
    model.generate(
        **inputs, max_new_tokens=5, use_cache=True, pad_token_id=tokenizer.eos_token_id
    )
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # 开启计时
    reset_attention_timing()
    enable_attention_timing_collection()

    # 测量 TTFT (首个 Token 时间)
    start_time = time.time()
    model.generate(
        **inputs, max_new_tokens=1, use_cache=True, pad_token_id=tokenizer.eos_token_id
    )
    torch.cuda.synchronize()
    ttft = time.time() - start_time

    # 测量完整生成
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=num_tokens,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    torch.cuda.synchronize()
    total_time = time.time() - start_time

    disable_attention_timing_collection()

    # 统计数据
    actual_tokens = outputs.shape[1] - input_len
    raw_times = get_raw_attention_times()
    stats_mean, stats_std = get_attention_stats()

    avg_attn = sum(raw_times) / len(raw_times) if raw_times else 0
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)  # GB

    throughput = actual_tokens / total_time if total_time > 0 else 0
    tpot = (total_time / actual_tokens * 1000) if actual_tokens > 0 else 0

    if verbose:
        print(
            f"   Avg Attention Time: {stats_mean*1000:.4f} ms +/- {stats_std*1000:.4f}"
        )
        if len(raw_times) >= 200:
            first_100 = sum(raw_times[:100]) / 100 * 1000
            last_100 = sum(raw_times[-100:]) / 100 * 1000
            print(f"   Attn Time (First 100): {first_100:.4f} ms")
            print(f"   Attn Time (Last 100):  {last_100:.4f} ms")
        print(
            f"   Actual generated: {actual_tokens} tokens, Peak Memory: {peak_mem:.4f} GB"
        )

    return {
        "TTFT (s)": ttft,
        "Total Time (s)": total_time,
        "TPOT (ms)": tpot,
        "Throughput (tok/s)": throughput,
        "Avg Attn (ms)": avg_attn * 1000,
        "Peak Mem (GB)": peak_mem,
    }


def print_results_table(results):
    """打印 Markdown 格式的结果表格"""
    print("\n" + "=" * 160)
    print(
        f"| {'Configuration':<20} | {'Wikitext PPL':<12} | {'PG-19 PPL':<10} | {'Total Time (s)':<14} | {'Avg Attn (ms)':<14} | {'TTFT (s)':<10} | {'TPOT (ms)':<10} | {'Throughput (tok/s)':<18} | {'Peak Mem (GB)':<13} |"
    )
    print(
        f"| {':---':<20} | {':---':<12} | {':---':<10} | {':---':<14} | {':---':<14} | {':---':<10} | {':---':<10} | {':---':<18} | {':---':<13} |"
    )

    for row in results:
        print(
            f"| {row['name']:<20} | "
            f"{row['wiki_ppl']:<12.2f} | "
            f"{row['pg19_ppl']:<10.2f} | "
            f"{row['total_time']:<14.4f} | "
            f"{row['avg_attn']:<14.4f} | "
            f"{row['ttft']:<10.4f} | "
            f"{row['tpot']:<10.2f} | "
            f"{row['throughput']:<18.2f} | "
            f"{row['peak_mem']:<13.2f} |"
        )
    print("=" * 160 + "\n")


def run_standard_benchmark():
    """执行标准测试流程：计算 PPL 和 Speed，并展示对比表格"""
    print("开始标准评估流程...")

    # 初始化模型
    print(f"正在加载模型: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map=device
    )

    # 加载数据
    wiki_text = load_long_text("wikitext", limit_chars=50000)
    pg19_text = load_long_text("pg19", limit_chars=50000)

    results = []

    for config in configs:
        print(f"\n正在运行配置: {config['name']} ...")

        # 1. 应用配置
        if config["type"] == "baseline":
            disable_streaming_llm(model)  # 确保移除之前的 patch
            patch_attention_layers(model)  # 仅 patch 计时器
        elif config["type"] == "streaming":
            enable_streaming_llm(
                model, n_sink=config["sink"], window_size=config["window"], debug=False
            )
        elif config["type"] == "h2o":
            enable_h2o_llm(
                model,
                n_sink=config["sink"],
                recent_window=config["recent"],
                max_capacity=config["capacity"],
                debug=False,
            )

        torch.cuda.empty_cache()
        gc.collect()

        # 2. PPL 测试
        # 对于 H2O，使用 capacity 作为 chunk_size
        if config["type"] == "h2o":
            chunk_size = config.get("capacity", 256)
        else:
            chunk_size = config.get("window", 1024)  # Baseline 默认 chunk 1024
        wiki_ppl = evaluate_ppl_unified(
            model, tokenizer, wiki_text, max_tokens=ppl_tokens, chunk_size=chunk_size
        )
        pg19_ppl = evaluate_ppl_unified(
            model, tokenizer, pg19_text, max_tokens=ppl_tokens, chunk_size=chunk_size
        )

        # 3. 速度测试
        prompt = wiki_text[: Pre_tokens * 4]
        speed_metrics = benchmark_generation_speed(
            model, tokenizer, prompt, num_tokens=speed_tokens
        )

        # 收集结果
        results.append(
            {
                "name": config["name"],
                "wiki_ppl": wiki_ppl,
                "pg19_ppl": pg19_ppl,
                "total_time": speed_metrics["Total Time (s)"],
                "avg_attn": speed_metrics["Avg Attn (ms)"],
                "ttft": speed_metrics["TTFT (s)"],
                "tpot": speed_metrics["TPOT (ms)"],
                "throughput": speed_metrics["Throughput (tok/s)"],
                "peak_mem": speed_metrics["Peak Mem (GB)"],
            }
        )

        print(
            f"  -> 完成。 Wiki PPL: {wiki_ppl:.2f}, Throughput: {speed_metrics['Throughput (tok/s)']:.2f} tok/s"
        )

    # 打印最终表格
    print_results_table(results)


def debug_test_mechanics():
    """调试函数：运行一次生成，开启 debug 模式以查看 KV Cache 内部状态"""
    print("=" * 60)
    print(" Running Configuration: streaming_8_256")
    print("=" * 60)
    print("   [Mode] StreamingLLM (Sink=8, Window=256)")

    # 使用 HuggingFace 模型
    model_id = "EleutherAI/pythia-2.8b"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="cuda"
    )

    print(" Enabling StreamingLLM: sink=8, window=256")
    # 启用 StreamingLLM 并开启 debug=True
    enable_streaming_llm(model, n_sink=8, window_size=256, debug=True)
    print(" Attention layers patched for timing.")
    print(" StreamingLLM enabled successfully.")

    # 1. Wikitext PPL (简略版，只跑少量 tokens 展示机制)
    print("   [Eval] Wikitext PPL (Limit 2000 tokens)...")
    # 加载一小段文本用于演示
    wiki_text = load_long_text("wikitext", limit_chars=5000)
    # 为了演示 PPL 计算中的 evict，我们调用 evaluate_ppl_unified
    # 注意：为了让 debug 信息打印出来，我们需要确保 evaluate_ppl_unified 内部会触发 update
    # 使用较小的 chunk_size (64) 来展示更频繁的 eviction 和更稳定的缓存大小，避免出现 752 -> 264 的大跳变
    ppl = evaluate_ppl_unified(
        model, tokenizer, wiki_text, max_tokens=1000, chunk_size=64
    )
    print(f"     -> PPL: {ppl:.4f}")

    # 2. PG19 PPL
    print("   [Eval] PG19 PPL (Limit 2000 tokens)...")
    pg19_text = load_long_text("pg19", limit_chars=5000)
    ppl = evaluate_ppl_unified(
        model, tokenizer, pg19_text, max_tokens=1000, chunk_size=64
    )
    print(f"    -> PPL: {ppl:.4f}")

    # 3. Speed Test
    print("   [Bench] Speed Test (Prompt=500, Gen=500)...")
    # 传递 verbose=True 给 benchmark_generation_speed 以打印详细的 attention 统计
    prompt = wiki_text[: Pre_tokens * 4]
    speed_metrics = benchmark_generation_speed(
        model, tokenizer, prompt, num_tokens=500, verbose=True
    )


if __name__ == "__main__":
    import argparse

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        print("运行调试模式")
        debug_test_mechanics()
    else:
        # 默认运行 Main
        print("运行标准评测")
        run_standard_benchmark()
