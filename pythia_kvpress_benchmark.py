import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM, GPTNeoXAttention, GPTNeoXModel, GPTNeoXLayer
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
import kvpress
from kvpress import SnapKVPress
from kvpress.presses.base_press import BasePress
import kvpress.utils
import time
import logging
import datasets
from tqdm import tqdm
import math

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Patching for GPTNeoX Support (Verified) ---

# 1. Alias 'model' to 'gpt_neox' for BasePress compatibility
if not hasattr(GPTNeoXForCausalLM, "model"):
    GPTNeoXForCausalLM.model = property(lambda self: self.gpt_neox)

# 2. Alias 'rotary_emb' in GPTNeoXModel to the one in the first layer's attention
if not hasattr(GPTNeoXModel, "rotary_emb"):
    def get_rotary_emb(self):
        return self.layers[0].attention.rotary_emb
    GPTNeoXModel.rotary_emb = property(get_rotary_emb)

# 3. Alias 'self_attn' to 'attention' in GPTNeoXLayer for BasePress compatibility
if not hasattr(GPTNeoXLayer, "self_attn"):
    GPTNeoXLayer.self_attn = property(lambda self: self.attention)

# 3b. Alias 'head_dim' to 'head_size' in GPTNeoXAttention
if not hasattr(GPTNeoXAttention, "head_dim"):
    GPTNeoXAttention.head_dim = property(lambda self: self.head_size)

# 3c. Alias 'num_key_value_heads' to 'num_attention_heads' in GPTNeoXConfig
# (Assume MHA if not present)
if not hasattr(GPTNeoXConfig, "num_key_value_heads"):
    GPTNeoXConfig.num_key_value_heads = property(lambda self: self.num_attention_heads)

# 4. Patch BasePress.forward_hook to handle GPTNeoX arguments
# Save the original hook FIRST
original_forward_hook = BasePress.forward_hook

def patched_forward_hook(self, module, input, kwargs, output):
    # GPTNeoX passes hidden_states as the first positional argument
    if "hidden_states" not in kwargs and len(input) > 0:
        kwargs["hidden_states"] = input[0]
    
    # Handle past_key_values (mapped from layer_past)
    if "past_key_values" not in kwargs:
        if "layer_past" in kwargs:
            kwargs["past_key_values"] = kwargs["layer_past"]
        elif len(input) > 4: # layer_past is often pos arg 4 or 5
            # Check if input[4] looks like a cache or tuple
            kwargs["past_key_values"] = input[4]

    # Handle cache_position (required by BasePress)
    if "cache_position" not in kwargs:
        cache = kwargs.get("past_key_values")
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is not None:
            q_len = hidden_states.shape[1]
            if cache is not None and hasattr(cache, "get_seq_length"):
                past_len = cache.get_seq_length()
            else:
                past_len = 0 # Fallback or infer from attention_mask?
            
            device = hidden_states.device
            kwargs["cache_position"] = torch.arange(past_len, past_len + q_len, device=device)

    # Handle position_embeddings (RoPE) - required by SnapKVPress
    if "position_embeddings" not in kwargs:
        hidden_states = kwargs.get("hidden_states")
        if hidden_states is not None and hasattr(module, "rotary_emb"):
            cache_position = kwargs.get("cache_position")
            if cache_position is not None:
                # GPTNeoX rotary_emb needs seq_len to generate enough embeddings
                seq_len = cache_position[-1].item() + 1
                try:
                    # GPTNeoXRotaryEmbedding.forward(x, seq_len=...)
                    # It returns (cos, sin) with shape [seq_len, dim] or [1, seq_len, dim]
                    cos, sin = module.rotary_emb(hidden_states, seq_len=seq_len)
                    
                    # Ensure batch dimension if needed (Llama expects [bsz, seq_len, dim] sometimes, 
                    # but usually broadcastable [1, seq_len, dim] is fine)
                    if cos.dim() == 2:
                        cos = cos.unsqueeze(0)
                        sin = sin.unsqueeze(0)
                        
                    kwargs["position_embeddings"] = (cos, sin)
                except Exception as e:
                    logger.warning(f"Failed to compute position_embeddings: {e}")

    return original_forward_hook(self, module, input, kwargs, output)

# Apply the patch
BasePress.forward_hook = patched_forward_hook

# 5. Patch query/key extraction functions
original_get_q = kvpress.utils.get_prerope_query_states
original_get_k = kvpress.utils.get_prerope_key_states

def patched_get_prerope_query_states(module, hidden_states):
    if isinstance(module, GPTNeoXAttention):
        # GPTNeoX computes QKV in one go: query_key_value(hidden_states)
        qkv = module.query_key_value(hidden_states)
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        # Q is the first part
        query_states = qkv[..., : num_heads * head_dim]
        bsz, q_len, _ = hidden_states.shape
        query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        return query_states
    return original_get_q(module, hidden_states)

def patched_get_prerope_key_states(module, hidden_states):
    if isinstance(module, GPTNeoXAttention):
        qkv = module.query_key_value(hidden_states)
        num_heads = module.config.num_attention_heads
        head_dim = module.head_dim
        # K is the second part
        start = num_heads * head_dim
        end = start + num_heads * head_dim
        key_states = qkv[..., start : end]
        bsz, k_len, _ = hidden_states.shape
        key_states = key_states.view(bsz, k_len, num_heads, head_dim).transpose(1, 2)
        return key_states
    return original_get_k(module, hidden_states)

kvpress.utils.get_prerope_query_states = patched_get_prerope_query_states
kvpress.utils.get_prerope_key_states = patched_get_prerope_key_states

# Patch the imported function in snapkv_press module (since it uses 'from ... import ...')
import kvpress.presses.snapkv_press
kvpress.presses.snapkv_press.get_prerope_query_states = patched_get_prerope_query_states

# Patch SnapKVPress.compute_window_attention to handle partial RoPE (GPTNeoX)
def patched_compute_window_attention(module, hidden_states, keys, window_size, position_embeddings):
    """
    Compute the last window_size queries and associated attention weights for the first q_len - window_size keys.
    Patched to support partial RoPE (GPTNeoX).
    """
    bsz, _, k_len, _ = keys.shape
    
    # Handle num_key_value_heads access
    if hasattr(module.config, "num_key_value_heads"):
        num_key_value_heads = module.config.num_key_value_heads
    else:
        num_key_value_heads = module.config.num_attention_heads
        
    num_heads = module.config.num_attention_heads
    head_dim = module.head_dim
    num_key_value_groups = num_heads // num_key_value_heads

    # Get last window_size queries
    query_states = kvpress.utils.get_prerope_query_states(module, hidden_states[:, -window_size:])

    # Apply RoPE
    cos, sin = position_embeddings
    cos, sin = cos[:, -window_size:], sin[:, -window_size:]
    
    # Check for partial RoPE (GPTNeoX)
    if query_states.shape[-1] != cos.shape[-1]:
        rotary_dim = cos.shape[-1]
        q_rot = query_states[..., :rotary_dim]
        q_pass = query_states[..., rotary_dim:]
        
        # Apply RoPE to q_rot
        q_rot = (q_rot * cos.unsqueeze(1)) + (rotate_half(q_rot) * sin.unsqueeze(1))
        
        # Concatenate back
        query_states = torch.cat([q_rot, q_pass], dim=-1)
    else:
        # Full RoPE (Llama etc)
        query_states = (query_states * cos.unsqueeze(1)) + (rotate_half(query_states) * sin.unsqueeze(1))

    # Compute attention for first q_len - window_size tokens
    key_states = repeat_kv(keys, num_key_value_groups)
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
    attention_mask = torch.ones_like(attn_weights) * float("-inf")
    attention_mask = torch.triu(attention_mask, diagonal=k_len - window_size + 1)
    attn_weights += attention_mask
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = attn_weights[..., :-window_size]

    return attn_weights

SnapKVPress.compute_window_attention = staticmethod(patched_compute_window_attention)

# 6. Add GPTNeoX to supported models
kvpress.presses.base_press.SUPPORTED_MODELS = kvpress.presses.base_press.SUPPORTED_MODELS + (GPTNeoXForCausalLM,)


# --- Benchmark Functions ---

def evaluate_ppl(model, tokenizer, dataset_name="wikitext", split="test", limit_samples=1, press=None):
    logger.info(f"Evaluating PPL on {dataset_name} ({split}) with press: {press}")
    
    try:
        if dataset_name == "wikitext":
            data = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
            # Take a subset of articles for speed
            text = "\n\n".join(data["text"][:limit_samples])
        elif dataset_name == "pg19":
             # PG19 is large, load only a small part if possible or use streaming
             # For this test, we might fallback to dummy if internet/disk is issue, but let's try
             # We will try to load 'deepmind/pg19' but it might require manual download.
             # Alternative: use a local file or skipping if not available.
             try:
                 data = datasets.load_dataset("pg19", split=split, streaming=True)
                 # Take first book
                 text = next(iter(data))["text"]
                 # Limit text length
                 text = text[:100000] 
             except:
                 logger.warning("Could not load pg19. Skipping.")
                 return float('inf')
        else:
            text = "This is a dummy text. " * 500
    except Exception as e:
        logger.warning(f"Failed to load dataset {dataset_name}: {e}. Using dummy text.")
        text = "This is a dummy text for testing purposes. " * 1000

    encodings = tokenizer(text, return_tensors="pt")
    
    # Limit total tokens to avoid OOM or long wait
    max_tokens = 5000 
    if encodings.input_ids.size(1) > max_tokens:
        encodings.input_ids = encodings.input_ids[:, :max_tokens]

    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    ctx = press(model) if press else torch.no_grad()
    if press is None:
        from contextlib import nullcontext
        ctx = nullcontext()

    pbar = tqdm(range(0, seq_len, stride), desc=f"Evaluating PPL ({dataset_name})")
    
    with ctx:
        for begin_loc in pbar:
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            if input_ids.size(1) == 0:
                break

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)
            prev_end_loc = end_loc
            if end_loc == seq_len:
                break

    if not nlls:
        return float('inf')
        
    ppl = torch.exp(torch.stack(nlls).mean())
    logger.info(f"PPL Result ({dataset_name}): {ppl.item()}")
    return ppl.item()

def benchmark_speed(model, tokenizer, press=None, num_tokens=50, batch_size=1):
    logger.info(f"Benchmarking speed (Generation) with Batch Size {batch_size}...")
    
    # Create input text
    # Max pos is 2048. We want to be close to it but leave room for generation.
    # Target: ~1800 context tokens.
    input_text = "The history of natural language processing is " * 280 # ~1900 tokens?
    
    # Tokenize once
    single_input = tokenizer(input_text, return_tensors="pt")
    input_ids = single_input.input_ids
    
    # Trim to fit
    max_context = 1800
    if input_ids.shape[1] > max_context:
        input_ids = input_ids[:, :max_context]
    
    seq_len = input_ids.shape[1]
    logger.info(f"Input sequence length: {seq_len}")
    
    # Replicate for batch
    input_ids = input_ids.repeat(batch_size, 1)
    attention_mask = torch.ones_like(input_ids)
    
    inputs = {
        "input_ids": input_ids.to(model.device),
        "attention_mask": attention_mask.to(model.device)
    }
    
    def get_ctx():
        if press:
            return press(model)
        from contextlib import nullcontext
        return nullcontext()
        
    # Warmup (batch=1 for speed)
    print("Warmup...")
    warmup_inputs = {k: v[:1] for k, v in inputs.items()}
    with get_ctx():
        model.generate(**warmup_inputs, max_new_tokens=5, do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
        
    # Measure
    print(f"Generating {num_tokens} tokens (BS={batch_size})...")
    start_time = time.time()
    with get_ctx():
        model.generate(**inputs, max_new_tokens=num_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id, use_cache=True)
    end_time = time.time()
    
    duration = end_time - start_time
    total_tokens = num_tokens * batch_size
    tokens_per_sec = total_tokens / duration
    logger.info(f"Time taken: {duration:.4f}s ({tokens_per_sec:.2f} tok/s)")
    return tokens_per_sec

if __name__ == "__main__":
    model_id = "EleutherAI/pythia-2.8b"
    # model_id = "EleutherAI/pythia-70m" # Uncomment for debugging
    
    logger.info(f"Loading {model_id}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # Load in half precision to save memory, map to auto device
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="auto", 
            torch_dtype=torch.float16, 
            trust_remote_code=True
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        exit(1)

    results = {}

    # 1. Baseline
    print("\n=== Baseline (No Press) ===")
    speed_base_bs1 = benchmark_speed(model, tokenizer, press=None, num_tokens=30, batch_size=1)
    # speed_base_bs4 = benchmark_speed(model, tokenizer, press=None, num_tokens=30, batch_size=4) # Commented out to save time for default run
    ppl_base_wiki = evaluate_ppl(model, tokenizer, dataset_name="wikitext", limit_samples=5, press=None)
    ppl_base_pg19 = evaluate_ppl(model, tokenizer, dataset_name="pg19", press=None)
    
    results['Baseline (BS=1)'] = {'speed': speed_base_bs1, 'ppl_wiki': ppl_base_wiki, 'ppl_pg19': ppl_base_pg19}

    # 2. SnapKVPress (Compression 0.5)
    print("\n=== SnapKVPress (ratio=0.5) ===")
    try:
        press_snap = SnapKVPress(compression_ratio=0.5)
        speed_snap_bs1 = benchmark_speed(model, tokenizer, press=press_snap, num_tokens=30, batch_size=1)
        # speed_snap_bs4 = benchmark_speed(model, tokenizer, press=press_snap, num_tokens=30, batch_size=4)
        ppl_snap_wiki = evaluate_ppl(model, tokenizer, dataset_name="wikitext", limit_samples=5, press=press_snap)
        ppl_snap_pg19 = evaluate_ppl(model, tokenizer, dataset_name="pg19", press=press_snap)
        
        results['SnapKV (BS=1)'] = {'speed': speed_snap_bs1, 'ppl_wiki': ppl_snap_wiki, 'ppl_pg19': ppl_snap_pg19}
        
    except Exception as e:
        logger.error(f"SnapKV Failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print(f"{'Method':<20} | {'Speed (tok/s)':<15} | {'PPL (Wiki)':<12} | {'PPL (PG19)':<12}")
    print("-" * 70)
    for method, res in results.items():
        print(f"{method:<20} | {res['speed']:<15.2f} | {res.get('ppl_wiki', 'N/A'):<12.2f} | {res.get('ppl_pg19', 'N/A'):<12.2f}")
    print("="*60)
