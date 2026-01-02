import os
import torch
import time
import math
import gc
import sys
import subprocess
from datasets import load_dataset
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
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
    # FP16 Baseline
    {"name": "baseline_fp16", "type": "baseline", "quant": None},
    {
        "name": "streaming_8_256_fp16",
        "type": "streaming",
        "sink": 8,
        "window": 256,
        "quant": None,
    },
    {
        "name": "h2o_8_32_264_fp16",
        "type": "h2o",
        "sink": 8,
        "recent": 32,
        "capacity": 264,
        "quant": None,
    },
    # 8-bit Quantization
    {"name": "baseline_int8", "type": "baseline", "quant": "8bit"},
    {
        "name": "streaming_8_256_int8",
        "type": "streaming",
        "sink": 8,
        "window": 256,
        "quant": "8bit",
    },
    {
        "name": "h2o_8_32_264_int8",
        "type": "h2o",
        "sink": 8,
        "recent": 32,
        "capacity": 264,
        "quant": "8bit",
    },
    # 4-bit Quantization
    {"name": "baseline_int4", "type": "baseline", "quant": "4bit"},
    {
        "name": "streaming_8_256_int4",
        "type": "streaming",
        "sink": 8,
        "window": 256,
        "quant": "4bit",
    },
    {
        "name": "h2o_8_32_264_int4",
        "type": "h2o",
        "sink": 8,
        "recent": 32,
        "capacity": 264,
        "quant": "4bit",
    },
]
# 使用 HuggingFace 模型
model_id = "EleutherAI/pythia-2.8b"
device = "cuda"
ppl_tokens = 1000
speed_tokens = 1000
Pre_tokens = 500


def load_model(quantization_type=None):
    """加载模型，支持可选的量化配置"""
    print(f"正在加载模型 {model_id}，量化类型={quantization_type}...")

    bnb_config = None
    try:
        if quantization_type == "8bit":
            import bitsandbytes

            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            print("  -> 使用 8-bit 量化")
        elif quantization_type == "4bit":
            import bitsandbytes

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            print("  -> 使用 4-bit 量化 (NF4 + Double Quantization)")
    except ImportError:
        print("错误: 未找到 'bitsandbytes' 库。量化已跳过。")
        return None, None
    except Exception as e:
        print(f"检查 bitsandbytes 时出错: {e}")
        return None, None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        if bnb_config:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, dtype=torch.float16, device_map=device, trust_remote_code=True
            )

        print(f"  -> 模型加载成功")
        return model, tokenizer
    except Exception as e:
        print(f"加载模型时出错: {e}")
        return None, None


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
            ds = load_dataset(
                "deepmind/pg19", split=split, streaming=True, trust_remote_code=True
            )
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


def calculate_ppl(
    model, tokenizer, text, max_tokens=1000, use_kv_cache=True, debug=False
):
    """
    计算困惑度 (PPL)

    Args:
        model: 模型
        tokenizer: 分词器
        text: 输入文本
        max_tokens: 最大测试长度
        use_kv_cache: 是否使用 KV Cache（True 时逐 token 生成，能真实反映压缩影响）
        debug: 是否打印调试信息
    """
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0

    # 限制最大测试长度
    max_test_len = min(seq_len, max_tokens)

    if debug:
        print(f"    [DEBUG] 原始序列长度: {seq_len}, 测试长度: {max_test_len}")

    if not use_kv_cache:
        # 快速模式：滑动窗口方式（不反映压缩影响）
        stride = 512
        MAX_LENGTH = 2048

        for begin_loc in range(0, max_test_len, stride):
            end_loc = min(begin_loc + MAX_LENGTH, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
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
        # 生成式 PPL 计算（逐 token，累积 past_key_values）
        # 这样 StreamingLLM/H2O 的压缩会真正影响后续预测
        input_ids = encodings.input_ids[:, :max_test_len].to(device)
        past_key_values = None

        eviction_count = 0
        cache_sizes = []
        cache_type_detected = False

        # 逐 token 预测：用 token[0:i] 预测 token[i]
        for i in range(1, input_ids.size(1)):
            with torch.no_grad():
                # 输入当前 token[i-1]（配合之前的 past_kv）
                current_input = input_ids[:, i - 1 : i]

                outputs = model(
                    current_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

                # 诊断：检测 cache 类型
                if not cache_type_detected and outputs.past_key_values is not None:
                    cache_class = type(outputs.past_key_values).__name__
                    if debug:
                        print(f"    [DEBUG] Cache 类型: {cache_class}")
                    cache_type_detected = True

                # 预测 token[i]
                logits = outputs.logits[:, -1, :]  # [batch, vocab_size]
                target = input_ids[:, i]  # [batch]

                # 计算 loss
                loss = torch.nn.functional.cross_entropy(logits, target)
                nlls.append(loss)

                # 更新 past_key_values（会被 StreamingLLM/H2O 压缩！）
                past_key_values = outputs.past_key_values

                # 记录实际 cache 大小（不是累计 token 数）
                if past_key_values is not None and hasattr(past_key_values, "layers"):
                    if len(past_key_values.layers) > 0 and hasattr(
                        past_key_values.layers[0], "keys"
                    ):
                        actual_cache_size = past_key_values.layers[0].keys.shape[-2]
                        cache_sizes.append(actual_cache_size)

                # H2O/StreamingLLM: 手动触发驱逐
                if hasattr(past_key_values, "evict_all_layers"):
                    past_key_values.evict_all_layers()
                    eviction_count += 1

        if debug:
            print(f"    [DEBUG] 触发驱逐次数: {eviction_count}")
            if cache_sizes:
                print(
                    f"    [DEBUG] Cache 大小范围: {min(cache_sizes)} ~ {max(cache_sizes)}, 平均: {sum(cache_sizes)/len(cache_sizes):.1f}"
                )

        prev_end_loc = input_ids.size(1) - 1

    if not nlls:
        return float("inf")

    ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc)
    return ppl.item()


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
        f"| {'Configuration':<20} | {'Wikitext PPL':<14} | {'PG-19 PPL':<12} | {'Total Time (s)':<14} | {'Avg Attn (ms)':<14} | {'TTFT (s)':<10} | {'TPOT (ms)':<10} | {'Throughput (tok/s)':<18} | {'Peak Mem (GB)':<13} |"
    )
    print(
        f"| {':---':<20} | {':---':<14} | {':---':<12} | {':---':<14} | {':---':<14} | {':---':<10} | {':---':<10} | {':---':<18} | {':---':<13} |"
    )

    for row in results:
        print(
            f"| {row['name']:<20} | "
            f"{row['wiki_ppl']:<14.4f} | "
            f"{row['pg19_ppl']:<12.4f} | "
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
    print("开始标准评估流程（包含量化配置）...")

    # 加载数据（提前加载，所有配置共用）
    wiki_text = load_long_text("wikitext", limit_chars=50000)
    pg19_text = load_long_text("pg19", limit_chars=50000)

    results = []

    # 模型缓存：按量化类型分组加载
    current_model = None
    current_tokenizer = None
    current_quant = "UNSET"

    for config in configs:
        print(f"\n正在运行配置: {config['name']} ...")

        # 检查是否需要重新加载模型（量化类型变化时）
        if config["quant"] != current_quant:
            if current_model is not None:
                print("  -> 清理旧模型...")
                del current_model
                del current_tokenizer
                torch.cuda.empty_cache()
                gc.collect()

            current_model, current_tokenizer = load_model(config["quant"])
            current_quant = config["quant"]

            if current_model is None:
                print(f"  -> 跳过配置 {config['name']}（模型加载失败）")
                continue

        # 1. 应用配置 - 先完全清理之前的配置！
        disable_streaming_llm(current_model)  # 清理所有之前的 patch 和配置

        if config["type"] == "baseline":
            patch_attention_layers(current_model)  # 仅 patch 计时器
        elif config["type"] == "streaming":
            enable_streaming_llm(
                current_model,
                n_sink=config["sink"],
                window_size=config["window"],
                debug=False,
            )
        elif config["type"] == "h2o":
            enable_h2o_llm(
                current_model,
                n_sink=config["sink"],
                recent_window=config["recent"],
                max_capacity=config["capacity"],
                debug=False,
            )

        # 强制清理 GPU 缓存和内存
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

        # 获取 GPU 温度和功率（用于诊断性能波动）
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=temperature.gpu,power.draw,clocks.gr",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.returncode == 0:
                temp, power, clock = result.stdout.strip().split(", ")
                print(f"    [GPU状态] 温度: {temp}°C, 功率: {power}W, 频率: {clock}MHz")
        except:
            pass

        # 2. PPL 测试（使用生成式计算，准确反映压缩影响）
        print(f"    正在计算 WikiText PPL...")
        wiki_ppl = calculate_ppl(
            current_model,
            current_tokenizer,
            wiki_text,
            max_tokens=ppl_tokens,
            use_kv_cache=True,
            debug=True,
        )
        print(f"    正在计算 PG-19 PPL...")
        pg19_ppl = calculate_ppl(
            current_model,
            current_tokenizer,
            pg19_text,
            max_tokens=ppl_tokens,
            use_kv_cache=True,
            debug=True,
        )

        # 3. 速度测试
        prompt = wiki_text[: Pre_tokens * 4]
        speed_metrics = benchmark_generation_speed(
            current_model, current_tokenizer, prompt, num_tokens=speed_tokens
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
            f"  -> 完成。 Wiki PPL: {wiki_ppl:.4f}, Throughput: {speed_metrics['Throughput (tok/s)']:.2f} tok/s"
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
    print("   [Eval] Wikitext PPL (Limit 1000 tokens)...")
    # 加载一小段文本用于演示
    wiki_text = load_long_text("wikitext", limit_chars=5000)
    ppl = calculate_ppl(model, tokenizer, wiki_text, max_tokens=1000, use_kv_cache=True)
    print(f"     -> PPL: {ppl:.4f}")

    # 2. PG19 PPL
    print("   [Eval] PG19 PPL (Limit 1000 tokens)...")
    pg19_text = load_long_text("pg19", limit_chars=5000)
    ppl = calculate_ppl(model, tokenizer, pg19_text, max_tokens=1000, use_kv_cache=True)
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
