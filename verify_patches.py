import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM, GPTNeoXAttention, GPTNeoXModel, GPTNeoXLayer
from transformers.models.gpt_neox.configuration_gpt_neox import GPTNeoXConfig
import kvpress
from kvpress import SnapKVPress
from kvpress.presses.base_press import BasePress
import kvpress.utils
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Patching for GPTNeoX Support ---

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
from transformers.models.llama.modeling_llama import repeat_kv, rotate_half
import math

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

def verify():
    model_id = "EleutherAI/pythia-70m"
    logger.info(f"Loading {model_id} for verification...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    
    input_text = "This is a test " * 50
    inputs = tokenizer(input_text, return_tensors="pt")
    
    press = SnapKVPress(compression_ratio=0.5)
    logger.info("Running generation with SnapKVPress...")
    
    # Enable cache for generation
    model.config.use_cache = True
    
    try:
        with press(model):
            outputs = model.generate(**inputs, max_new_tokens=10, use_cache=True)
            
        logger.info("Generation successful!")
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated text sample:", decoded[:100])
        print("Verification Passed.")
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
