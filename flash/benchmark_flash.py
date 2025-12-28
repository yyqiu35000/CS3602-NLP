import torch
import time
import math
import gc
import sys
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add current directory to sys.path to ensure we import the local patch
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pythia_streaming_patch import (
    enable_streaming_llm, 
    disable_streaming_llm, 
    reset_attention_timing, 
    enable_attention_timing_collection, 
    disable_attention_timing_collection, 
    get_attention_stats,
    patch_attention_layers,
    get_raw_attention_times
)

# Configuration
configs = [
    {"name": "Baseline", "type": "baseline"},
    {"name": "StreamingLLM", "type": "streaming", "sink": 8, "window": 256},
]

# Attention implementations to test
# Note: "sdpa" requires PyTorch 2.0+ and compatible hardware
attn_impls = ["eager", "sdpa"]

model_id = "EleutherAI/pythia-2.8b" 
device = "cuda"
ppl_tokens = 1000
speed_tokens = 500
Pre_tokens = 500

def load_long_text(dataset_name="wikitext", split="test", limit_chars=50000):
    """加载wiki/pg19文本用于测试"""
    print(f"正在加载 {dataset_name}...")
    try:
        if dataset_name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            text = "\n\n".join(ds["text"])
        elif dataset_name == "pg19":
            ds = load_dataset("pg19", split=split, streaming=True)
            sample = next(iter(ds))
            text = sample["text"]
        else:
            text = "Long text placeholder. " * 1000
    except Exception as e:
        print(f"警告: 加载 {dataset_name} 失败 ({e}), 使用占位文本。")
        text = "This is a dummy text for fallback. " * 1000
    
    if len(text) < limit_chars:
        text = text * math.ceil(limit_chars / len(text))
    return text[:limit_chars]

def evaluate_ppl_unified(model, tokenizer, text: str, max_tokens: int = 2000, chunk_size: int = 512):
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
    
    for i in range(0, seq_len, chunk_size):
        chunk_input_ids = input_ids[:, i : i + chunk_size]
        chunk_target = chunk_input_ids.clone()
        
        position_ids = torch.arange(i, i + chunk_input_ids.size(1), dtype=torch.long, device=chunk_input_ids.device)
        position_ids = position_ids.unsqueeze(0) 

        with torch.no_grad():
            outputs = model(
                chunk_input_ids, 
                labels=chunk_target, 
                past_key_values=past_key_values,
                position_ids=position_ids, 
                use_cache=True
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
        
    if not nlls: return float("inf")
    
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
    model.generate(**inputs, max_new_tokens=5, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # 开启计时
    reset_attention_timing()
    enable_attention_timing_collection()
    
    # 测量 TTFT (首个 Token 时间)
    start_time = time.time()
    model.generate(**inputs, max_new_tokens=1, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    ttft = time.time() - start_time

    # 测量完整生成
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=num_tokens,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id
    )
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    disable_attention_timing_collection()
    
    # 统计数据
    actual_tokens = outputs.shape[1] - input_len
    raw_times = get_raw_attention_times()
    stats_mean, stats_std = get_attention_stats()
    
    avg_attn = sum(raw_times) / len(raw_times) if raw_times else 0
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) # GB
    
    throughput = actual_tokens / total_time if total_time > 0 else 0
    tpot = (total_time / actual_tokens * 1000) if actual_tokens > 0 else 0
    
    if verbose:
         print(f"   Avg Attention Time: {stats_mean*1000:.4f} ms +/- {stats_std*1000:.4f}")
         print(f"   Actual generated: {actual_tokens} tokens, Peak Memory: {peak_mem:.4f} GB")

    return {
        "TTFT (s)": ttft,
        "Total Time (s)": total_time,
        "TPOT (ms)": tpot,
        "Throughput (tok/s)": throughput,
        "Avg Attn (ms)": avg_attn * 1000,
        "Peak Mem (GB)": peak_mem
    }

def print_results_table(results):
    """打印 Markdown 格式的结果表格"""
    print("\n" + "="*180)
    print(f"| {'Configuration':<30} | {'Attn Impl':<10} | {'Wikitext PPL':<12} | {'PG-19 PPL':<10} | {'Total Time (s)':<14} | {'Avg Attn (ms)':<14} | {'TTFT (s)':<10} | {'TPOT (ms)':<10} | {'Throughput':<12} | {'Peak Mem (GB)':<13} |")
    print(f"| {':---':<30} | {':---':<10} | {':---':<12} | {':---':<10} | {':---':<14} | {':---':<14} | {':---':<10} | {':---':<10} | {':---':<12} | {':---':<13} |")
    
    for row in results:
        print(f"| {row['name']:<30} | "
              f"{row['attn']:<10} | "
              f"{row['wiki_ppl']:<12.2f} | "
              f"{row['pg19_ppl']:<10.2f} | "
              f"{row['total_time']:<14.4f} | "
              f"{row['avg_attn']:<14.4f} | "
              f"{row['ttft']:<10.4f} | "
              f"{row['tpot']:<10.2f} | "
              f"{row['throughput']:<12.2f} | "
              f"{row['peak_mem']:<13.2f} |")
    print("="*180 + "\n")

def run_benchmark():
    print(f"Starting Flash Benchmark for model: {model_id}")
    
    # Pre-load data
    wiki_text = load_long_text("wikitext", limit_chars=50000)
    pg19_text = load_long_text("pg19", limit_chars=50000)
    
    results = []
    
    for attn_impl in attn_impls:
        print(f"\n{'='*40}")
        print(f" Testing with Attention Implementation: {attn_impl.upper()}")
        print(f"{'='*40}")
        
        print(f"Loading model with {attn_impl}...")
        try:
            # Use bfloat16 for stability in eager mode, especially for 2.8B model
            # FP16 eager mode often overflows, causing PPL=inf
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16, 
                device_map=device,
                attn_implementation=attn_impl
            )
        except Exception as e:
            print(f"Failed to load model with {attn_impl}: {e}")
            continue

        # Check if SDPA is actually used
        print(f"Model config attn_implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")

        for config in configs:
            config_name = f"{config['name']}"
            print(f"\nRunning configuration: {config_name} ({attn_impl})")
            
            # Reset/Patch
            disable_streaming_llm(model)
            
            if config["type"] == "baseline":
                patch_attention_layers(model) # Ensure timing is patched for Baseline
            elif config["type"] == "streaming":
                enable_streaming_llm(model, n_sink=config["sink"], window_size=config["window"], debug=False)
            
            torch.cuda.empty_cache()
            gc.collect()
            
            # PPL
            chunk_size = 128
            wiki_ppl = evaluate_ppl_unified(model, tokenizer, wiki_text, max_tokens=ppl_tokens, chunk_size=chunk_size)
            pg19_ppl = evaluate_ppl_unified(model, tokenizer, pg19_text, max_tokens=ppl_tokens, chunk_size=chunk_size)
            
            
            # Speed
            prompt = wiki_text[:Pre_tokens*4]
            speed_metrics = benchmark_generation_speed(model, tokenizer, prompt, num_tokens=speed_tokens)
            
            results.append({
                "name": config_name,
                "attn": attn_impl,
                "wiki_ppl": wiki_ppl,
                "pg19_ppl": pg19_ppl,
                "total_time": speed_metrics["Total Time (s)"],
                "avg_attn": speed_metrics["Avg Attn (ms)"],
                "ttft": speed_metrics["TTFT (s)"],
                "tpot": speed_metrics["TPOT (ms)"],
                "throughput": speed_metrics["Throughput (tok/s)"],
                "peak_mem": speed_metrics["Peak Mem (GB)"]
            })
            
            print(f"  -> Done. Wiki PPL: {wiki_ppl:.2f}, Throughput: {speed_metrics['Throughput (tok/s)']:.2f} tok/s")
            
        # Clean up model to free memory for next iteration
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    print_results_table(results)

if __name__ == "__main__":
    run_benchmark()
