import torch
import time
import math
import gc
import sys
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Add project root to sys.path to import from root
# Current file: .../CS3602-NLP/integrate/main_quant_semantic.py
# dirname -> integrate
# dirname(dirname) -> CS3602-NLP
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# Import from compress package since sys.path has root
from compress.pythia_streaming_compress import enable_innovation_llm, disable_innovation_llm

# ==========================================
# Configuration & Globals
# ==========================================

model_id = "EleutherAI/pythia-2.8b" 
device = "cuda" if torch.cuda.is_available() else "cpu"

configs = [
    # Baseline
    {"name": "Baseline (FP16)", "type": "baseline", "quant": None},
    
    # Standard StreamingLLM (Sink=4, Window=256)
    {"name": "StreamingLLM (FP16)", "type": "streaming", "quant": None, "sink": 4, "window": 256},
    
    # Combined: Int4 Quantization + Semantic Block (No Comp)
    # Uses quantization for speed/mem, and Semantic Block (No Comp) for better PPL than simple Streaming
    {"name": "Int4 + Semantic (No Comp)", "type": "semantic_block", "quant": "4bit", 
     "sink": 4, "window": 64, "extra": 192, "compress": False} 
     # Total Cache = 4 + 64 + 192 = 260 tokens (approx similar to 256 window + 4 sink)
]

ppl_tokens = 2000       
speed_tokens = 1500      
pre_tokens = 500       

def print_flush(msg, end="\n"):
    print(msg, end=end, flush=True)

# ==========================================
# Helper Functions
# ==========================================

def load_model(quantization_type=None):
    print_flush(f"Loading model {model_id} with quantization={quantization_type}...")
    
    bnb_config = None
    try:
        if quantization_type == "8bit":
            import bitsandbytes
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        elif quantization_type == "4bit":
            import bitsandbytes
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
    except ImportError:
        print_flush("Error: 'bitsandbytes' library not found. Quantization skipped.")
        return None, None
    except Exception as e:
        print_flush(f"Error checking bitsandbytes: {e}")
        return None, None
    
    try:
        if bnb_config:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                quantization_config=bnb_config, 
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, 
                dtype=torch.float16, 
                device_map="auto",
                trust_remote_code=True
            )
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        return model, tokenizer
    except Exception as e:
        print_flush(f"Error loading model: {e}")
        return None, None

def load_long_text(dataset_name="wikitext", limit_chars=50000):
    print_flush(f"Loading {dataset_name}...")
    try:
        if dataset_name == "wikitext":
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            text = "\n\n".join(ds["text"])
        elif dataset_name == "pg19":
            ds = load_dataset("pg19", split="test", streaming=True)
            text = next(iter(ds))["text"]
        else:
            text = "Long text placeholder. " * 1000
    except Exception as e:
        print_flush(f"Warning: Failed to load {dataset_name} ({e}), using placeholder.")
        text = "This is a dummy text for fallback. " * 1000
    
    if len(text) < limit_chars:
        text = text * math.ceil(limit_chars / len(text))
    return text[:limit_chars]

def evaluate_ppl_unified(model, tokenizer, text: str, max_tokens: int = 2000, chunk_size: int = 512):
    """
    Unified PPL evaluation (Chunk-wise)
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
            
            # Manual eviction for StreamingLLM simulation during PPL
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
    """Benchmark generation speed and Attention time"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    
    if verbose:
        print_flush(f"   Prompt tokens: {input_len}, Requesting {num_tokens} tokens")
    
    # Warmup
    model.generate(**inputs, max_new_tokens=5, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    # Timing
    reset_attention_timing()
    enable_attention_timing_collection()
    
    # TTFT
    start_time = time.time()
    model.generate(**inputs, max_new_tokens=1, use_cache=True, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    ttft = time.time() - start_time

    # Full Generation
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
    
    # Stats
    actual_tokens = outputs.shape[1] - input_len
    raw_times = get_raw_attention_times()
    stats_mean, stats_std = get_attention_stats()
    
    avg_attn = sum(raw_times) / len(raw_times) if raw_times else 0
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) # GB
    
    throughput = actual_tokens / total_time if total_time > 0 else 0
    tpot = (total_time / actual_tokens * 1000) if actual_tokens > 0 else 0
    
    return {
        "TTFT (s)": ttft,
        "Total Time (s)": total_time,
        "TPOT (ms)": tpot,
        "Throughput (tok/s)": throughput,
        "Avg Attn (ms)": avg_attn * 1000,
        "Peak Mem (GB)": peak_mem
    }

def print_results_table(results):
    print("\n" + "="*160)
    print(f"| {'Configuration':<25} | {'Wikitext PPL':<12} | {'PG-19 PPL':<10} | {'Total Time (s)':<14} | {'Avg Attn (ms)':<14} | {'TTFT (s)':<10} | {'TPOT (ms)':<10} | {'Throughput (tok/s)':<18} | {'Peak Mem (GB)':<13} |")
    print(f"| {':---':<25} | {':---':<12} | {':---':<10} | {':---':<14} | {':---':<14} | {':---':<10} | {':---':<10} | {':---':<18} | {':---':<13} |")
    
    for row in results:
        print(f"| {row['name']:<25} | "
              f"{row['wiki_ppl']:<12.2f} | "
              f"{row['pg19_ppl']:<10.2f} | "
              f"{row['total_time']:<14.4f} | "
              f"{row['avg_attn']:<14.4f} | "
              f"{row['ttft']:<10.4f} | "
              f"{row['tpot']:<10.2f} | "
              f"{row['throughput']:<18.2f} | "
              f"{row['peak_mem']:<13.2f} |")
    print("="*160 + "\n")

# ==========================================
# Main Execution
# ==========================================

def run_benchmark():
    print_flush("Starting Combined Benchmark (Quantization + Semantic No Comp)...")
    results = []
    
    # Load Data
    wiki_text = load_long_text("wikitext", limit_chars=50000)
    pg19_text = load_long_text("pg19", limit_chars=50000)
    
    # Group configs by quantization type to minimize model reloading
    # Sort configs so we can load model once per quant type if possible
    # But for simplicity, we just reload if quant type changes.
    
    current_model = None
    current_tokenizer = None
    current_quant = "UNSET"
    
    for config in configs:
        print_flush(f"\nRunning Configuration: {config['name']} ...")
        
        # Check if we need to reload model
        if config["quant"] != current_quant:
            if current_model is not None:
                del current_model
                del current_tokenizer
                torch.cuda.empty_cache()
                gc.collect()
            
            current_model, current_tokenizer = load_model(config["quant"])
            current_quant = config["quant"]
            
            if current_model is None:
                print_flush(f"Skipping {config['name']} due to model load failure.")
                continue
        
        # Apply Configuration
        # Reset any existing patches
        disable_streaming_llm(current_model)
        disable_innovation_llm(current_model)
        
        if config["type"] == "baseline":
            patch_attention_layers(current_model) # Only timing patch
        elif config["type"] == "streaming":
            enable_streaming_llm(current_model, n_sink=config["sink"], window_size=config["window"], debug=False)
        elif config["type"] == "semantic_block":
            enable_innovation_llm(
                current_model, 
                tokenizer=current_tokenizer,
                method="semantic_block", 
                n_sink=config["sink"], 
                window_size=config["window"], 
                extra_size=config["extra"], 
                compress=config["compress"],
                debug=False
            )
            
        torch.cuda.empty_cache()
        gc.collect()
        
        # PPL Test
        chunk_size = 128
        wiki_ppl = evaluate_ppl_unified(current_model, current_tokenizer, wiki_text, max_tokens=ppl_tokens, chunk_size=chunk_size)
        pg19_ppl = evaluate_ppl_unified(current_model, current_tokenizer, pg19_text, max_tokens=ppl_tokens, chunk_size=chunk_size)
        
        # Speed Test
        prompt = wiki_text[:pre_tokens*4]
        speed_metrics = benchmark_generation_speed(current_model, current_tokenizer, prompt, num_tokens=speed_tokens)
        
        # Collect Results
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
        
        print_flush(f"  -> Done. Wiki PPL: {wiki_ppl:.2f}, Throughput: {speed_metrics['Throughput (tok/s)']:.2f} tok/s, Peak Mem: {speed_metrics['Peak Mem (GB)']:.2f} GB")

    print_results_table(results)

if __name__ == "__main__":
    run_benchmark()
