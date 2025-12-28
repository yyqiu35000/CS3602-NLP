import torch
import time
import types
from typing import Optional, Callable
from transformers.cache_utils import DynamicCache, Cache
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM, eager_attention_forward, ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb

# --- Global State for Attention Timing ---
ATTENTION_TIMES = []
TIMING_ENABLED = False

def reset_attention_timing():
    global ATTENTION_TIMES
    ATTENTION_TIMES = []

def enable_attention_timing_collection():
    global TIMING_ENABLED
    TIMING_ENABLED = True

def disable_attention_timing_collection():
    global TIMING_ENABLED
    TIMING_ENABLED = False

def get_attention_stats():
    if not ATTENTION_TIMES:
        return 0.0, 0.0
    t = torch.tensor(ATTENTION_TIMES)
    return t.mean().item(), t.std().item()

def get_raw_attention_times():
    return list(ATTENTION_TIMES)

# --- Streaming Cache Policy ---
class StreamingDynamicCache(DynamicCache):
    """
    实现了 StreamingLLM 的 Sink + Sliding Window (保留首部+滑动窗口) 驱逐策略。
    维护: [初始 Sink tokens] + [最近的 Sliding Window tokens]
    """
    def __init__(self, config, n_sink=4, window_size=256, debug=False):
        super().__init__(config=config)
        self.n_sink = n_sink
        self.window_size = window_size
        self.debug = debug
        self._seen_tokens_by_layer = {}

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        # 记录序列长度
        if layer_idx not in self._seen_tokens_by_layer:
            self._seen_tokens_by_layer[layer_idx] = 0
        self._seen_tokens_by_layer[layer_idx] += key_states.shape[-2]

        # 标准更新 (调用父类 DynamicCache)
        k_out, v_out = super().update(key_states, value_states, layer_idx, cache_kwargs)

        # 驱逐逻辑 (Eviction Logic)
        current_len = k_out.shape[-2]
        limit = self.n_sink + self.window_size
        
        # 调试信息打印 (每100步打印一次，或者在发生驱逐时打印)
        if self.debug and layer_idx == 0:
            if self._seen_tokens_by_layer[layer_idx] % 100 == 0:
                print(f"DEBUG: Step {self._seen_tokens_by_layer[layer_idx]} | Key Shape: {k_out.shape} | Limit: {limit}")

        # 当超过限制 + 缓冲区(64) 时触发驱逐，避免每一步都做 cat 操作
        if current_len > limit + 64: 
            k_sink = k_out[:, :, :self.n_sink, :]
            k_window = k_out[:, :, -self.window_size:, :]
            k_new = torch.cat([k_sink, k_window], dim=-2)
            
            v_sink = v_out[:, :, :self.n_sink, :]
            v_window = v_out[:, :, -self.window_size:, :]
            v_new = torch.cat([v_sink, v_window], dim=-2)
            
            if layer_idx < len(self.layers):
                self.layers[layer_idx].keys = k_new
                self.layers[layer_idx].values = v_new
            
            if self.debug and layer_idx == 0:
                # print(f"DEBUG EVICT: Layer 0 evict {current_len} -> {k_new.shape[-2]}")
                pass
                
            return k_out, v_out

        return k_out, v_out

    def evict_all_layers(self):
        """Force eviction on all layers (e.g. for PPL evaluation loop)"""
        for layer_idx in range(len(self.layers)):
            if not hasattr(self.layers[layer_idx], "keys") or self.layers[layer_idx].keys is None:
                continue
                
            k = self.layers[layer_idx].keys
            v = self.layers[layer_idx].values
            
            current_len = k.shape[-2]
            limit = self.n_sink + self.window_size
            
            # Use same buffer logic or strict limit? For PPL we usually want strict or consistent state.
            if current_len > limit:
                k_sink = k[:, :, :self.n_sink, :]
                k_window = k[:, :, -self.window_size:, :]
                self.layers[layer_idx].keys = torch.cat([k_sink, k_window], dim=-2)
                
                v_sink = v[:, :, :self.n_sink, :]
                v_window = v[:, :, -self.window_size:, :]
                self.layers[layer_idx].values = torch.cat([v_sink, v_window], dim=-2)

    def get_seq_length(self, layer_idx=0):
        if layer_idx in self._seen_tokens_by_layer:
            return self._seen_tokens_by_layer[layer_idx]
        return super().get_seq_length(layer_idx)

# --- Monkey Patching Logic ---
def patched_gpt_neox_attention_forward(
    self,
    hidden_states: torch.FloatTensor,
    attention_mask: torch.FloatTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    layer_past: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, 3 * self.head_size)

    qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states, value_states = qkv.chunk(3, dim=-1)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if layer_past is not None:
        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "partial_rotation_size": self.rotary_ndims,
            "cache_position": cache_position,
        }
        key_states, value_states = layer_past.update(key_states, value_states, self.layer_idx, cache_kwargs) 

        # --- StreamingLLM Fix: Manual Mask Construction ---
        # 解决 RoPE 需要逻辑位置而 Mask 需要物理位置的冲突
        # 1. RoPE 使用逻辑位置 (由 get_seq_length 保证)
        # 2. Mask 使用物理位置 (此处手动构建)
        if isinstance(layer_past, StreamingDynamicCache):
            b_sz, _, q_len, _ = query_states.shape
            k_len = key_states.shape[-2]
            
            # 如果是 Chunk 处理 (q_len > 1)，需要对 Chunk 部分应用 Causal Mask
            if q_len > 1:
                # 初始化全 0 (Visible) Mask
                min_val = torch.finfo(query_states.dtype).min
                new_mask = torch.zeros((b_sz, 1, q_len, k_len), device=query_states.device, dtype=query_states.dtype)
                
                # 构建 Causal Mask (上三角为负无穷)
                causal_mask = torch.full((q_len, q_len), min_val, device=query_states.device, dtype=query_states.dtype)
                causal_mask = torch.triu(causal_mask, diagonal=1)
                
                # 将 Causal Mask 应用到 Mask 的最后 q_len 列
                # 物理 Cache 的最后 q_len 个 Token 就是当前的 Query Chunk
                if k_len >= q_len:
                    new_mask[:, :, :, -q_len:] = causal_mask
                
                # 覆盖传入的 attention_mask
                attention_mask = new_mask
            else:
                # [Optimization] Decoding 阶段 (q_len=1)
                # 因为 Cache 中的所有 Token (Sink + Window) 对当前 Query 都是可见的 (都在过去)
                # 所以我们不需要任何 Mask (全 0 Mask 等价于 None)
                # 将 Mask 置为 None 以启用底层 "No Mask" 优化路径 (MatMul only)
                attention_mask = None

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
    # --- TIMING BLOCK ---
    start_t = None
    # Measure only if enabled AND we are in generation phase (1 token at a time)
    if TIMING_ENABLED and hidden_states.shape[1] == 1:
        torch.cuda.synchronize()
        start_t = time.time()
    # --------------------

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        head_mask=head_mask,
        **kwargs,
    )

    # --- TIMING RECORD ---
    if start_t is not None:
        torch.cuda.synchronize()
        end_t = time.time()
        ATTENTION_TIMES.append((end_t - start_t) * 1000) # ms
    # ---------------------

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.dense(attn_output)

    return attn_output, attn_weights

def patch_attention_layers(model):
    """Replaces the forward method of attention layers with the timed version."""
    if hasattr(model, "gpt_neox"):
        for layer in model.gpt_neox.layers:
            attn = layer.attention
            if not hasattr(attn, "_original_forward_streaming"):
                attn._original_forward_streaming = attn.forward
                attn.forward = types.MethodType(patched_gpt_neox_attention_forward, attn)

def unpatch_attention_layers(model):
    if hasattr(model, "gpt_neox"):
        for layer in model.gpt_neox.layers:
            attn = layer.attention
            if hasattr(attn, "_original_forward_streaming"):
                attn.forward = attn._original_forward_streaming
                del attn._original_forward_streaming

def enable_streaming_llm(model: GPTNeoXForCausalLM, n_sink=4, window_size=256, debug=False):
    """Enables StreamingLLM by patching model.forward to inject StreamingDynamicCache."""
    if hasattr(model, "_original_forward_streaming_patch"):
        model._streaming_config = (n_sink, window_size, debug)
    else:
        model._original_forward_streaming_patch = model.forward
        model._streaming_config = (n_sink, window_size, debug)

        def streaming_forward(self, input_ids=None, past_key_values=None, use_cache=None, **kwargs):
            n_sink, window_size, debug = self._streaming_config
            
            # Inject StreamingDynamicCache if starting fresh or if standard Cache is passed
            if use_cache:
                if past_key_values is None:
                    past_key_values = StreamingDynamicCache(self.config, n_sink=n_sink, window_size=window_size, debug=debug)
                elif isinstance(past_key_values, DynamicCache) and not isinstance(past_key_values, StreamingDynamicCache):
                    # Replace with Streaming Cache if it's empty/new
                    if past_key_values.get_seq_length() == 0:
                         past_key_values = StreamingDynamicCache(self.config, n_sink=n_sink, window_size=window_size, debug=debug)
            
            return self._original_forward_streaming_patch(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )
        
        model.forward = types.MethodType(streaming_forward, model)
    
    # Also patch attention layers for timing (optional, but requested)
    patch_attention_layers(model)

def disable_streaming_llm(model):
    if hasattr(model, "_original_forward_streaming_patch"):
        model.forward = model._original_forward_streaming_patch
        del model._original_forward_streaming_patch
    unpatch_attention_layers(model)
