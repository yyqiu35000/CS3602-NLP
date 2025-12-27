import torch
import time
import types
from typing import Optional, Callable, Tuple
from transformers.cache_utils import DynamicCache, Cache
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXForCausalLM, 
    apply_rotary_pos_emb
)
import torch.nn.functional as F

# --- 1. 核心流式缓存实现 ---
class StreamingDynamicCache(DynamicCache):
    def __init__(self, config, n_sink=4, window_size=256, debug=False):
        super().__init__()
        self.n_sink = n_sink
        self.window_size = window_size
        self.debug = debug
        self.limit = n_sink + window_size
        
        # Explicitly initialize storage if parent didn't
        if not hasattr(self, "keys"):
            self.keys = []
        if not hasattr(self, "values"):
            self.values = []
        
        self._seen_tokens = 0  # Track total tokens seen
        
    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        # Update seen tokens (only once per step, usually layer 0 counts)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            
        # 1. 获取当前层之前的缓存并更新
        if len(self.keys) <= layer_idx:
            # 初始化层缓存
            k_out = key_states
            v_out = value_states
            self.keys.append(k_out)
            self.values.append(v_out)
            
            # DEBUG: Check if we are initializing with a huge prompt
            if self.debug and layer_idx == 0:
                print(f"[Layer {layer_idx}] Initialized with shape {k_out.shape}. Limit={self.limit}")

        else:
            # 拼接新 token
            k_out = torch.cat([self.keys[layer_idx], key_states], dim=-2)
            v_out = torch.cat([self.values[layer_idx], value_states], dim=-2)
            self.keys[layer_idx] = k_out
            self.values[layer_idx] = v_out
            
        # 2. 统一驱逐逻辑：检查当前物理缓存是否超限
        # 无论是初始化(Prefill)还是增量更新，都需要检查
        current_len = self.keys[layer_idx].shape[-2]
        
        # 只有当超出 limit + 缓冲区时才执行昂贵的切片操作
        # 注意：对于 Prefill 阶段，如果 Prompt 很长，这里会立即触发截断
        if current_len > self.limit + 64:
            if self.debug and layer_idx == 0:
                print(f"[Layer {layer_idx}] Truncating! {current_len} -> {self.n_sink + self.window_size}")

            k_full = self.keys[layer_idx]
            v_full = self.values[layer_idx]
            
            k_sink = k_full[:, :, :self.n_sink, :]
            k_window = k_full[:, :, -self.window_size:, :]
            k_new = torch.cat([k_sink, k_window], dim=-2)
            
            v_sink = v_full[:, :, :self.n_sink, :]
            v_window = v_full[:, :, -self.window_size:, :]
            v_new = torch.cat([v_sink, v_window], dim=-2)

            self.keys[layer_idx] = k_new
            self.values[layer_idx] = v_new

        # CRITICAL FIX for StreamingLLM + FlashAttn:
        # Although we updated the physical cache (self.keys) to be compressed (Sink + Window),
        # the current computation step MUST see the FULL context (k_out, v_out) that was just computed.
        # If we return k_new/v_new here, the current chunk will lose access to its own middle tokens!
        # This matches the behavior of pythia_streaming_patch.py.
        
        # MEMORY LEAK FIX:
        # PyTorch SDPA might hold internal buffers or gradients (even in no_grad) if inputs are large.
        # But crucially, 'k_out' and 'v_out' here ARE the large tensors (e.g. 2000 tokens).
        # We return them to be used in the current step's attention.
        # The Peak Memory WILL reflect this large allocation because it exists right now!
        
        # Why does Standard StreamingLLM (pythia_streaming_patch.py) have lower Peak Memory?
        # In pythia_streaming_patch.py, 'update' ALSO returns the full 'k_out' / 'v_out'.
        # And 'patched_gpt_neox_attention_forward' uses them.
        # So logically they should have the SAME peak memory.
        
        # However, the user reports:
        # Baseline: 5.63 GB (Full Cache)
        # StreamingLLM: 5.37 GB (Truncated Cache)
        # StreamingLLM + Flash: 5.63 GB (Truncated Cache but high peak)
        
        # This means StreamingLLM somehow AVOIDS allocating the full 5.63GB peak.
        # The only way is if it NEVER concatenates the full tensor?
        # But 'update' DOES concatenate: k_out = torch.cat([...])
        
        # Wait! In 'pythia_streaming_patch.py':
        # if current_len > limit + 64:
        #    ...
        #    self.keys[layer_idx] = k_new
        #    return k_out, v_out
        
        # It behaves identically.
        #
        # Difference might be in HOW attention is computed.
        # Standard: Eager Attention (MatMul)
        # Flash: SDPA
        #
        # Maybe SDPA allocates a large temporary buffer for the attention map (even if using Flash)?
        # If we pass a mask (which we do for Prefill), SDPA might fallback to a less memory-efficient path?
        #
        # In Prefill (q_len > 1), we construct a `full_mask`.
        # `full_mask` shape is [1, 1, Query_Len, Key_Len].
        # If Query_Len=2000, Key_Len=2000.
        # Mask Size = 2000*2000 * 4 bytes (float32) = 16MB. Small.
        #
        # But Eager attention computes `attn_weights = Q @ K.T`.
        # Shape: [Batch, Heads, Q_Len, K_Len] = [1, 32, 2000, 2000].
        # 32 * 4M * 2 bytes (fp16) = 256 MB.
        #
        # SDPA should be MORE memory efficient (O(N) memory).
        # So why is it higher?
        #
        # Maybe `torch.cat` is the culprit?
        # When we do `k_out = torch.cat([self.keys[layer_idx], key_states], dim=-2)`, we allocate a new tensor.
        # This tensor is Size(2000).
        #
        # If we could avoid creating `k_out` (Large) entirely?
        # But we need it for the attention of the current chunk!
        #
        # Unless... we split the prefill into smaller chunks?
        # The benchmark calls `model.generate(prompt)`.
        # HuggingFace `generate` processes the prompt in one go (unless we implement chunking in forward).
        #
        # Is it possible that `pythia_streaming_patch.py` somehow triggers garbage collection earlier?
        #
        # Let's try to explicitly `del` intermediate tensors if possible, or force `empty_cache`.
        # But we can't delete `k_out` before returning it.
        #
        # Let's look at the `update` return value usage in `patched_gpt_neox_attention_forward_sdpa`.
        # `key_states, value_states = layer_past.update(...)`
        # `attn_output = F.sdpa(..., key_states, value_states, ...)`
        #
        # If `key_states` (the large one) is kept alive longer than needed?
        #
        # One radical optimization:
        # If we are in Prefill (q_len > 1) and we are truncating...
        # Do we really need to attend to the tokens that we are about to throw away?
        # YES, absolutely. The current tokens depend on the immediate past, even if that past is about to be evicted for *future* tokens.
        # StreamingLLM assumption is that we can evict middle tokens, but usually for *future* generation.
        # For the *current* prefill, standard Self-Attention attends to all previous tokens in the prompt.
        #
        # WAIT. StreamingLLM's key idea is that we ONLY keep Sink + Window.
        # Does that mean we only ATTEND to Sink + Window?
        # If we attend to Sink + Window ONLY, we can construct `k_out` as `cat(Sink, Window)` directly!
        #
        # If we do that, we save memory and compute!
        #
        # In `pythia_streaming_patch.py`, `update` returns `k_out` which is the CONCATENATION (Large).
        # So Standard StreamingLLM attends to EVERYTHING during prefill (standard attention).
        #
        # So why is Flash Memory higher?
        #
        # Maybe it's `k_out` (Large) + `self.keys` (Small) + `k_sink/k_window` (Small).
        # In Standard: `self.keys` (Small) is updated. `k_out` (Large) is returned.
        #
        # Maybe the issue is simply that SDPA kernel reserves some workspace memory that Eager doesn't (unlikely, Eager allocates full Attn Matrix)?
        #
        # Let's try to force Python GC.
        import gc
        if layer_idx == 0 and self.debug:
             gc.collect()
             # torch.cuda.empty_cache() # This is slow, but good for debugging peak mem.

        return k_out, v_out

    def get_seq_length(self, layer_idx: int = 0) -> int:
        # Return total seen tokens so position_ids are correct (monotonic) for RoPE
        if layer_idx == 0:
            return self._seen_tokens
        # For other layers, assume same sync
        return self._seen_tokens

# --- 2. 适配 Flash Attention (SDPA) 的注意力 Patch ---
def patched_gpt_neox_attention_forward_sdpa(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: Optional[torch.FloatTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    layer_past: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    # SDPA expects [batch, heads, seq_len, head_dim]
    # GPTNeoX default: [batch, seq_len, heads, head_dim] -> transpose(1, 2)
    # But qkv above is: view(hidden_shape).transpose(1, 2)
    # hidden_shape = (batch, seq, -1, 3 * head_size) -> This seems wrong in original code if num_heads is not explicitly handled
    
    # Let's follow standard GPTNeoX logic but adapted
    num_heads = self.config.num_attention_heads
    head_dim = self.head_size
    
    # [batch, seq_len, 3 * hidden_size]
    qkv = self.query_key_value(hidden_states)
    
    # [batch, seq_len, num_heads, 3 * head_dim]
    new_shape = qkv.shape[:-1] + (num_heads, 3 * head_dim)
    qkv = qkv.view(new_shape)
    
    # [batch, num_heads, seq_len, 3 * head_dim]
    qkv = qkv.permute(0, 2, 1, 3)
    
    query_states, key_states, value_states = qkv.chunk(3, dim=-1)

    # 2. RoPE 旋转位置编码
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # 3. 更新 Streaming Cache
    if layer_past is not None:
        # 注意：StreamingLLM 下 cache_position 需要映射到物理缓存位置
        key_states, value_states = layer_past.update(key_states, value_states, self.layer_idx)

    # 4. 使用 PyTorch 原生 SDPA (自动触发 Flash Attention 2 / Memory Efficient Attention)
    
    # 准备 timing (植入全局计时逻辑)
    # 优化：避免在热循环中重复查找 sys.modules
    start_t = None
    should_time = False
    
    # 尝试获取全局 timing 状态 (通过 enable_streaming_flash_attn 注入的引用，或者直接查找一次)
    if hasattr(self, "_timing_module"):
        if self._timing_module.TIMING_ENABLED and hidden_states.shape[1] == 1:
            should_time = True
    else:
        # Fallback (slow path, but should only happen once ideally if we patch correctly)
        import sys
        if "pythia_streaming_patch" in sys.modules:
            tm = sys.modules["pythia_streaming_patch"]
            # Cache it for future calls
            self._timing_module = tm
            if tm.TIMING_ENABLED and hidden_states.shape[1] == 1:
                should_time = True

    if should_time:
        torch.cuda.synchronize()
        start_t = time.time()
    
    # Determine is_causal and attention_mask usage
    # We must construct a correct mask for SDPA because we are manually handling the cache.
    # CRITICAL: get_seq_length() now returns LOGICAL length (total seen tokens) for correct RoPE.
    
    query_len = query_states.shape[-2]
    key_len = key_states.shape[-2]
    is_causal = False
    
    # --- Optimization: Fast Path for Generation (Token-by-Token) ---
    if query_len == 1:
        # Generation step
        # Query Len = 1. It can attend to everything in Key (Past + Self).
        # So no Causal masking needed within the query.
        # Optimization: Pass None to SDPA to use optimized kernel
        attention_mask = None
    elif query_len == key_len:
        # Prefill phase (Standard Causal Attention)
        # If query_len == key_len, it means we are attending to the full sequence (Self-Attention).
        # We can use optimized Flash Attention (is_causal=True) if there's no padding.
        
        has_padding = False
        if attention_mask is not None:
            # Check for padding (assuming additive mask where -inf is masking)
            if attention_mask.min() < -100:
                has_padding = True
        
        if not has_padding:
            attention_mask = None
            is_causal = True
        else:
            # Fallback to manual mask for padding
            # But we can still use causal structure in the mask
            pass 
            
    else:
        # Chunk processing (Prefill) or weird shape
        # Ignore provided attention_mask if shape mismatches significantly (logical vs physical)
        if attention_mask is not None:
            mask_last_dim = attention_mask.shape[-1]
            if mask_last_dim != key_len:
                 attention_mask = None
        
        # Construct [Batch, 1, Query_Len, Key_Len]
        # Left part (Past Cache) = 1 (visible)
        # Right part (Current Query) = Lower Triangular
        
        # Calculate how many past tokens exist
        past_len = key_len - query_len
        
        # Create causal mask for the query part
        # [1, 1, Query_Len, Query_Len]
        causal_mask = torch.triu(torch.ones(query_len, query_len, device=query_states.device), diagonal=1).bool()
        causal_mask = ~causal_mask # Lower triangular is True
        
        if past_len > 0:
            # Concatenate with past mask (all visible)
            # [1, 1, Query_Len, Past_Len] -> True
            past_mask = torch.ones(1, 1, query_len, past_len, device=query_states.device, dtype=torch.bool)
            # [1, 1, Query_Len, Query_Len]
            query_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            
            # [1, 1, Query_Len, Key_Len]
            full_mask = torch.cat([past_mask, query_mask], dim=-1)
        else:
            full_mask = causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply padding mask if provided (and matches shape)
        if attention_mask is not None:
              # Only if shape matches (which we checked above)
              if attention_mask.min() < -100:
                  full_mask_bias = torch.zeros_like(attention_mask)
                  full_mask_bias.masked_fill_(~full_mask, torch.finfo(query_states.dtype).min)
                  attention_mask = attention_mask + full_mask_bias
              else:
                  pass
        else:
             # No padding mask provided, create bias from our boolean mask
             attention_mask = torch.zeros(1, 1, query_len, key_len, device=query_states.device, dtype=query_states.dtype)
             attention_mask.masked_fill_(~full_mask, torch.finfo(query_states.dtype).min)

    # SDPA 调用 (支持 Flash Attention)
    attn_output = F.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=0.0 if not self.training else self.attention_dropout,
        is_causal=is_causal
    )

    # 记录时间
    if start_t is not None:
        torch.cuda.synchronize()
        end_t = time.time()
        if hasattr(self, "_timing_module"):
             self._timing_module.ATTENTION_TIMES.append((end_t - start_t) * 1000)


    # 5. 输出投影
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(*hidden_states.shape[:-1], -1)
    attn_output = self.dense(attn_output)

    return attn_output, None # SDPA 不返回 weights 以节省算力

# --- 3. 启用函数 ---
def enable_streaming_flash_attn(model: GPTNeoXForCausalLM, n_sink=4, window_size=256, debug=True):
    """
    重构：强制模型使用 SDPA 实现并注入流式缓存
    """
    # 1. 修改配置强制走 SDPA
    model.config._attn_implementation = "sdpa"
    
    # 2. Patch 所有的 Attention 层
    for layer in model.gpt_neox.layers:
        attn = layer.attention
        attn.forward = types.MethodType(patched_gpt_neox_attention_forward_sdpa, attn)
    
    # 3. Patch Model Forward 以自动注入 Streaming Cache
    if not hasattr(model, "_original_forward"):
        model._original_forward = model.forward

    def streaming_forward(self, input_ids=None, past_key_values=None, use_cache=None, position_ids=None, **kwargs):
        if use_cache and past_key_values is None:
            # FORCE DEBUG=True to see print statements
            past_key_values = StreamingDynamicCache(self.config, n_sink=n_sink, window_size=window_size, debug=debug)
        return self._original_forward(input_ids=input_ids, past_key_values=past_key_values, use_cache=use_cache, position_ids=position_ids, **kwargs)

    model.forward = types.MethodType(streaming_forward, model)
    print(f"StreamingLLM (SDPA/FlashAttn) enabled: Sink={n_sink}, Window={window_size}, Debug={debug}")