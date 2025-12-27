import torch
import time
import math
import gc
import sys
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from pythia_streaming_patch_debug import (
    enable_streaming_llm, 
    disable_streaming_llm, 
    reset_attention_timing, 
    enable_attention_timing_collection, 
    disable_attention_timing_collection, 
    get_attention_stats,
    patch_attention_layers,
    get_raw_attention_times
)

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
    
    # print(f"  > PPL 评估: {seq_len} tokens, chunk_size={chunk_size}")
    
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

def debug_roef_mask():
    """Debug function to check RoPE and Mask dimensions"""
    print("="*60)
    print(" Debugging RoPE and Mask Mismatch")
    print("="*60)
    
    model_id = "EleutherAI/pythia-160m"    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16, device_map="cuda")
    
    n_sink = 4
    window_size = 10
    chunk_size = 10
    
    print(f" Enabling StreamingLLM: sink={n_sink}, window={window_size}")
    # Enable StreamingLLM with debug=True to see internal logs from the cache if any
    enable_streaming_llm(model, n_sink=n_sink, window_size=window_size, debug=True)
    
    # Generate dummy text
    # 50 tokens is enough to trigger eviction (4 sink + 10 window = 14 capacity)
    text = "The quick brown fox jumps over the lazy dog. " * 20
    
    print(f" Running PPL evaluation with chunk_size={chunk_size}...")
    # evaluate_ppl_unified handles chunking and calls the model
    evaluate_ppl_unified(model, tokenizer, text, max_tokens=50, chunk_size=chunk_size)

if __name__ == "__main__":
    debug_roef_mask()
