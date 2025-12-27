import torch
import time
import math
import gc
import sys
import copy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForCausalLM
import torch.nn.functional as F

# Add project root to sys.path to import modules
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import existing streaming patch
from pythia_streaming_patch import (
    enable_streaming_llm, 
    disable_streaming_llm, 
    reset_attention_timing, 
    enable_attention_timing_collection, 
    disable_attention_timing_collection, 
    get_attention_stats,
    patch_attention_layers,
    get_raw_attention_times,
    StreamingDynamicCache as PyramidStreamingCache
)
from pythia_streaming_flash import enable_streaming_flash_attn

# ==========================================
# Benchmark Logic
# ==========================================

configs = [
        # {"name": "Baseline (Eager)", "type": "baseline"},
        {"name": "StreamingLLM", "type": "streaming", "sink": 8, "window": 512},
        {"name": "StreamingLLM + FlashAttn", "type": "streaming_flash_attn", "sink": 8, "window": 512},
    ]

model_id = "EleutherAI/pythia-2.8b" 
device = "cuda"
ppl_tokens = 1000
speed_tokens = 1000
pre_tokens = 500

def print_flush(msg, end="\n"):
    print(msg, end=end, flush=True)

def load_long_text(dataset_name="wikitext", split="test", limit_chars=50000):
    """加载wiki/pg19文本用于测试"""
    print_flush(f"正在加载 {dataset_name}...")
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
    
    print_flush(f"  > PPL 评估: {seq_len} tokens, chunk_size={chunk_size}")
    
    for i in range(0, seq_len, chunk_size):
        if i % (chunk_size * 5) == 0:
            print_flush(".", end="")
        
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
         print_flush(f"   Avg Attention Time: {stats_mean*1000:.4f} us +/- {stats_std*1000:.4f} us")
         print_flush(f"   Actual generated: {actual_tokens} tokens, Peak Memory: {peak_mem:.4f} GB")

    return {
        "TTFT (s)": ttft,
        "Total Time (s)": total_time,
        "TPOT (ms)": tpot,
        "Throughput (tok/s)": throughput,
        "Avg Attn (ms)": avg_attn,
        "Peak Mem (GB)": peak_mem
    }

def print_results_table(results):
    """打印 Markdown 格式的结果表格"""
    print("\n" + "="*160)
    print(f"| {'Configuration':<24} | {'Wikitext PPL':<12} | {'PG-19 PPL':<10} | {'Total Time (s)':<14} | {'Avg Attn (ms)':<14} | {'TTFT (s)':<10} | {'TPOT (ms)':<10} | {'Throughput (tok/s)':<18} | {'Peak Mem (GB)':<13} |")
    print(f"| {':---':<24} | {':---':<12} | {':---':<10} | {':---':<14} | {':---':<14} | {':---':<10} | {':---':<10} | {':---':<18} | {':---':<13} |")
    
    for row in results:
        print(f"| {row['name']:<24} | "
              f"{row['wiki_ppl']:<12.2f} | "
              f"{row['pg19_ppl']:<10.2f} | "
              f"{row['total_time']:<14.4f} | "
              f"{row['avg_attn']:<14.4f} | "
              f"{row['ttft']:<10.4f} | "
              f"{row['tpot']:<10.2f} | "
              f"{row['throughput']:<18.2f} | "
              f"{row['peak_mem']:<13.2f} |")
    print("="*160 + "\n")

def run_comprehensive_benchmark():
    print("Starting Comprehensive Benchmark...")
    results = []
    
    # 初始化模型
    print(f"正在加载模型: {model_id}")
    # Important: Flash Attention (SDPA) usually requires float16/bfloat16
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map=device)
    
    # 加载数据
    wiki_text = load_long_text("wikitext", limit_chars=50000)
    pg19_text = load_long_text("pg19", limit_chars=50000)
    
    for config in configs:
        print(f"\n>>> Running Configuration: {config['name']}")
        
        # 1. Setup Environment
        # Reload model to ensure clean state for patching/unpatching
        del model
        torch.cuda.empty_cache()
        gc.collect()
        
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map=device)
        
        if config["type"] == "baseline":
            disable_streaming_llm(model)
            patch_attention_layers(model) # 仅 patch 计时器
            
        elif config["type"] == "streaming":
            enable_streaming_llm(model, n_sink=config["sink"], window_size=config["window"], debug=False)
            
        elif config["type"] == "streaming_flash_attn":
            print("Enabling Real Streaming Flash Attention (SDPA)...")
            # 同样需要 patch 计时器，这部分逻辑在 pythia_streaming_flash.py 中如果没实现，需要补充
            enable_streaming_flash_attn(model, n_sink=config["sink"], window_size=config["window"])
            # Manually set debug=True for the cache to verify truncation
            # We need to iterate over layers to find where the cache is stored or just rely on the next forward pass
            # Since enable_streaming_flash_attn replaces model.forward, we can pass debug=True there if supported,
            # but currently it's hardcoded or uses default. Let's patch it or just wait for the run.
            # Actually, let's force the model to use debug mode in the cache
            pass

        torch.cuda.empty_cache()
        gc.collect()
        
        # 2. Measure PPL (chunk size 128 for consistency with main.py)
        print("Measuring PPL...")
        chunk_size = 128
        wiki_ppl = evaluate_ppl_unified(model, tokenizer, wiki_text, max_tokens=ppl_tokens, chunk_size=chunk_size)
        pg19_ppl = evaluate_ppl_unified(model, tokenizer, pg19_text, max_tokens=ppl_tokens, chunk_size=chunk_size)
        
        # 3. Measure Speed
        print("Measuring Speed...")
        prompt = wiki_text[:pre_tokens*4]
        try:
            speed_metrics = benchmark_generation_speed(model, tokenizer, prompt, num_tokens=speed_tokens, verbose=True)
        except Exception as e:
            print(f"Error in generation: {e}")
            speed_metrics = {k: 0.0 for k in ["TTFT (s)", "Total Time (s)", "TPOT (ms)", "Throughput (tok/s)", "Avg Attn (ms)", "Peak Mem (GB)"]}
            
        print(f"Result: {speed_metrics['Throughput (tok/s)']:.2f} tok/s, Wiki PPL: {wiki_ppl:.2f}")
        
        results.append({
            "name": config["name"],
            "wiki_ppl": wiki_ppl,
            "pg19_ppl": pg19_ppl,
            "total_time": speed_metrics["Total Time (s)"],
            "avg_attn": speed_metrics["Avg Attn (ms)"],
            "ttft": speed_metrics["TTFT (s)"],
            "tpot": speed_metrics["TPOT (ms)"],
            "throughput": speed_metrics["Throughput (tok/s)"],
            "peak_mem": speed_metrics["Peak Mem (GB)"]
        })
        
    # Print Table
    print_results_table(results)

if __name__ == "__main__":
    run_comprehensive_benchmark()
