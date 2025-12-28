import torch
import time
import math
import gc
import sys
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pythia_streaming_patch import (
    enable_streaming_llm, 
    disable_streaming_llm, 
    reset_attention_timing, 
    enable_attention_timing_collection, 
    disable_attention_timing_collection, 
    get_attention_stats,
    patch_attention_layers
)

# ==========================================
# Configuration & Globals
# ==========================================

model_id = "EleutherAI/pythia-2.8b" 
device = "cuda" if torch.cuda.is_available() else "cpu"

configs = [
    {"name": "Baseline (FP16)", "type": "baseline", "quant": None},
    {"name": "Streaming (FP16)", "type": "streaming", "quant": None, "sink": 4, "window": 256},
    {"name": "Streaming + Int8", "type": "streaming", "quant": "8bit", "sink": 4, "window": 256},
    {"name": "Streaming + NF4", "type": "streaming", "quant": "4bit", "sink": 4, "window": 256},
]

ppl_tokens = 1000       
speed_tokens = 500      
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
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
    except Exception as e:
        print_flush(f"Warning: Failed to load {dataset_name} ({e}), using dummy text.")
        text = "The quick brown fox jumps over the lazy dog. " * 2000
    
    if len(text) < limit_chars:
        text = text * math.ceil(limit_chars / len(text))
    return text[:limit_chars]

def evaluate_ppl(model, tokenizer, text, max_tokens=2048, chunk_size=512):
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids[:, :max_tokens].to(model.device)
    seq_len = input_ids.size(1)
    
    nlls = []
    token_counts = [] 
    past_key_values = None 
    
    print_flush(f"  > Measuring PPL on {seq_len} tokens (chunk={chunk_size})...", end=" ")
    
    for i in range(0, seq_len, chunk_size):
        chunk_input_ids = input_ids[:, i : i + chunk_size]
        chunk_target = chunk_input_ids.clone()
        
        with torch.no_grad():
            outputs = model(
                chunk_input_ids, 
                labels=chunk_target, 
                past_key_values=past_key_values,
                use_cache=True
            )
            
            loss = outputs.loss
            past_key_values = outputs.past_key_values
            
            if hasattr(past_key_values, "evict_all_layers"):
                past_key_values.evict_all_layers()
            
            if not torch.isnan(loss) and not torch.isinf(loss):
                nlls.append(loss)
                token_counts.append(chunk_input_ids.size(1))
        print_flush(".", end="")
        
    print_flush(" Done.")
    
    if not nlls: return float("inf")
    
    total_loss = sum(l * c for l, c in zip(nlls, token_counts))
    total_tokens = sum(token_counts)
    return torch.exp(total_loss / total_tokens).item()

def benchmark_speed(model, tokenizer, prompt_text, gen_tokens=100):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    
    print_flush(f"  > Benchmarking Speed: Prompt={input_len}, Gen={gen_tokens}...")
    
    # Warmup
    model.generate(**inputs, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    
    # Reset stats
    reset_attention_timing()
    enable_attention_timing_collection()
    torch.cuda.reset_peak_memory_stats()
    
    # 1. Measure TTFT
    start_t = time.time()
    model.generate(**inputs, max_new_tokens=1, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    ttft = time.time() - start_t
    
    # 2. Measure Generation
    start_t = time.time()
    outputs = model.generate(**inputs, max_new_tokens=gen_tokens, pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    total_time = time.time() - start_t
    
    disable_attention_timing_collection()
    
    generated_count = outputs.shape[1] - input_len
    
    throughput = generated_count / total_time
    tpot = (total_time - ttft) / (generated_count - 1) * 1000 if generated_count > 1 else 0
    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    
    attn_mean, attn_std = get_attention_stats()
    
    return {
        "TTFT (s)": ttft,
        "Throughput (tok/s)": throughput,
        "TPOT (ms)": tpot,
        "Peak Mem (GB)": peak_mem,
        "Avg Attn (ms)": attn_mean * 1000
    }

# ==========================================
# Main Execution
# ==========================================

def run_benchmark():
    print_flush("=== Pythia 2.8B Quantization & Streaming Benchmark ===")
    
    # Load data
    wiki_text = load_long_text("wikitext", limit_chars=100000)
    pg19_text = load_long_text("pg19", limit_chars=100000)
    
    prompt_text = wiki_text[:pre_tokens*4]
    
    results = []
    
    for config in configs:
        print_flush(f"\n------------------------------------------------")
        print_flush(f"Running Configuration: {config['name']}")
        print_flush(f"------------------------------------------------")
        
        torch.cuda.empty_cache()
        gc.collect()
        
        # Load Model
        model, tokenizer = load_model(config["quant"])
        if model is None:
            print_flush("Skipping configuration due to load error.")
            continue
            
        # Apply Patching
        if config["type"] == "baseline":
            disable_streaming_llm(model)
            patch_attention_layers(model)
        elif config["type"] == "streaming":
            enable_streaming_llm(
                model, 
                n_sink=config["sink"], 
                window_size=config["window"], 
                debug=False
            )
            
        # Run PPL Eval
        print_flush("Evaluating WikiText PPL:")
        wiki_ppl = evaluate_ppl(model, tokenizer, wiki_text, max_tokens=ppl_tokens, chunk_size=512)
        print_flush(f"  > WikiText PPL: {wiki_ppl:.2f}")

        print_flush("Evaluating PG-19 PPL:")
        pg19_ppl = evaluate_ppl(model, tokenizer, pg19_text, max_tokens=ppl_tokens, chunk_size=512)
        print_flush(f"  > PG-19 PPL: {pg19_ppl:.2f}")
        
        # Run Speed Eval
        speed_metrics = benchmark_speed(model, tokenizer, prompt_text, gen_tokens=speed_tokens)
        print_flush(f"  > Speed: {speed_metrics['Throughput (tok/s)']:.2f} tok/s, Mem: {speed_metrics['Peak Mem (GB)']:.2f} GB")
        
        results.append({
            "name": config["name"],
            "wiki_ppl": wiki_ppl,
            "pg19_ppl": pg19_ppl,
            **speed_metrics
        })
        
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        
    # Print Final Table
    print_flush("\n" + "="*140)
    print_flush(f"| {'Configuration':<25} | {'Wiki PPL':<10} | {'PG19 PPL':<10} | {'Throughput':<15} | {'TPOT (ms)':<10} | {'Avg Attn (ms)':<15} | {'Peak Mem (GB)':<15} |")
    print_flush(f"| {':---':<25} | {':---':<10} | {':---':<10} | {':---':<15} | {':---':<10} | {':---':<15} | {':---':<15} |")
    
    for r in results:
        print_flush(f"| {r['name']:<25} | {r['wiki_ppl']:<10.2f} | {r['pg19_ppl']:<10.2f} | {r['Throughput (tok/s)']:<15.2f} | {r['TPOT (ms)']:<10.2f} | {r['Avg Attn (ms)']:<15.4f} | {r['Peak Mem (GB)']:<15.2f} |")
    print_flush("="*140)

if __name__ == "__main__":
    run_benchmark()
