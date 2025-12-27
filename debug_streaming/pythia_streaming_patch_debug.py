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
        # 恢复为返回逻辑长度，保证 RoPE 自动推断或模型内部逻辑正常（虽然我们显式传入了 position_ids，但模型其他部分可能依赖这个）
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

    # [DEBUG] 验证参与旋转的是仅当前 Chunk 还是包含缓存
    if self.layer_idx == 0:
        print(f"  [RoPE Check] Key Shape entering RoPE: {key_states.shape} (Current Input Only)")
        if position_embeddings is not None:
             cos, sin = position_embeddings
             # 反推 position_ids: cos 的形状通常是 [batch, seq_len, head_dim] 或类似
             # 我们通过检查 cos 的非零模式或者直接打印形状来推断
             # 对于 RoPE，cos 通常是预计算的，但其切片对应于 position_ids
             # 这里我们直接打印前几个 token 的 cos 值（取第一个维度的第一个元素作为特征）来观察是否有位移
             # 或者，如果 cache_position 被传入，我们也可以打印它
             pass
    
    # [DEBUG] 打印推断出的 Position IDs (通过 cache_position 或 hidden_states 形状推测)
    # 注意：在 GPT-NeoX 中，position_ids 通常是在模型最外层生成的，这里我们只能通过副作用观察
    # 但我们可以打印 cache_position 如果它存在
    if cache_position is not None and self.layer_idx == 0:
         print(f"  [Pos Check] Cache Position: {cache_position}")
    elif self.layer_idx == 0:
         # 尝试从 position_embeddings 推断 (仅作示意，因为这里拿不到原始 position_ids)
         pass

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

    # --- 修复 Mask 逻辑：手动构建 Mask 替代传入的 attention_mask ---
    # 原因：传入的 attention_mask 依赖逻辑长度 (get_seq_length)，而 KV Cache 是物理长度 (Sink+Window+Current)
    # 这会导致维度不匹配或未来信息泄露。我们需要基于物理长度构建正确的 Mask。
    
    query_len = query_states.shape[-2]
    key_len = key_states.shape[-2]
    
    # 只有当 key_len 和 query_len 不一致时（即有 Past Cache），才需要特殊处理
    # 且只有在 query_len > 1 (Chunk 处理) 时才需要复杂的 Causal Mask
    # 如果 query_len == 1 (Generation)，默认全是 Past，不需要 Causal Mask (或者说是全 1)
    
    if layer_past is not None:
         # 1. 丢弃传入的 attention_mask (通常是 [batch, 1, 1, logical_len] 或 [batch, 1, q_len, logical_len])
         #    因为它与物理 key_len 不匹配
         
         # 2. 构建新的物理 Mask
         # 目标形状: [batch, 1, query_len, key_len]
         
         if query_len > 1:
             # Chunk Prefill 阶段
             # Mask 结构: [Past (Sink+Window) | Current (Causal Lower Triangular)]
             
             past_len = key_len - query_len
             
             # 左半部分: Past Cache -> 全部可见 (True)
             past_mask = torch.ones(query_len, past_len, device=query_states.device, dtype=torch.bool)
             
             # 右半部分: Current Chunk -> 因果掩码 (下三角为 True)
             # diagonal=1 表示上三角(不含对角线)为 1，取反即下三角(含对角线)为 1
             chunk_mask = torch.triu(torch.ones(query_len, query_len, device=query_states.device, dtype=torch.bool), diagonal=1)
             chunk_mask = ~chunk_mask # 翻转为下三角
             
             # 拼接
             # [query_len, key_len]
             full_mask_2d = torch.cat([past_mask, chunk_mask], dim=-1)
             
             # 扩展维度 -> [1, 1, query_len, key_len]
             new_mask = full_mask_2d.unsqueeze(0).unsqueeze(0)
             
             # 转换为 float mask (0.0 for True, -inf for False)
             # 注意：GPTNeoX 的 attention 实现会加上 attention_mask，所以我们需要 0.0 和 -inf
             # 但如果 attention_mask 原本是 4D，它会被直接使用。
             # 我们这里直接替换。
             
             # 创建全 0 (可见) 
             attention_mask = torch.zeros_like(new_mask, dtype=query_states.dtype)
             # 将 False 的位置填为 -inf
             attention_mask.masked_fill_(~new_mask, torch.finfo(query_states.dtype).min)
             
         else:
             # Generation 阶段 (query_len = 1)
             # 单个 Token 可以看到所有 Past + Self
             # 所以全是 True (0.0)
             # [1, 1, 1, key_len]
             attention_mask = torch.zeros(1, 1, 1, key_len, device=query_states.device, dtype=query_states.dtype)
             
    # -----------------------------------------------------------

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

    if self.layer_idx == 0:
        print(f"\n[DEBUG Layer {self.layer_idx}]")
        print(f"  Query Shape: {query_states.shape}")
        print(f"  Key Shape:   {key_states.shape}")
        if layer_past is not None:
             print(f"  Cache Seq Len: {layer_past.get_seq_length()}")
        if attention_mask is not None:
              print(f"  Mask Shape:  {attention_mask.shape}")
              print(f"  Mask Dtype:  {attention_mask.dtype}")
              
              # Correctly calculate visible count based on dtype
              if attention_mask.dtype == torch.bool:
                  visible_count = attention_mask[0, 0, 0, :].sum().item()
              else:
                  # Assume float mask where visible is 0.0 and hidden is -inf
                  visible_count = (attention_mask[0, 0, 0, :] > -100).sum().item()
                  
              print(f"  First Query Visible Keys: {visible_count}")
              # Print full tensor for verification
              torch.set_printoptions(threshold=10000, edgeitems=100)
              import pdb
              print(f"  First Query Mask Values: {attention_mask[0, 0, 0, :]}")
        else:
             print(f"  Mask is None")

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
