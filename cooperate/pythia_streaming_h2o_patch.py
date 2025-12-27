import torch
import time
import types
from typing import Optional, Callable
from transformers.cache_utils import DynamicCache, Cache
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXForCausalLM,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
)

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

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ):
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
                print(
                    f"DEBUG: Step {self._seen_tokens_by_layer[layer_idx]} | Key Shape: {k_out.shape} | Limit: {limit}"
                )

        # 当超过限制 + 缓冲区(64) 时触发驱逐，避免每一步都做 cat 操作
        if current_len > limit + 64:
            k_sink = k_out[:, :, : self.n_sink, :]
            k_window = k_out[:, :, -self.window_size :, :]
            k_new = torch.cat([k_sink, k_window], dim=-2)

            v_sink = v_out[:, :, : self.n_sink, :]
            v_window = v_out[:, :, -self.window_size :, :]
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
            if (
                not hasattr(self.layers[layer_idx], "keys")
                or self.layers[layer_idx].keys is None
            ):
                continue

            k = self.layers[layer_idx].keys
            v = self.layers[layer_idx].values

            current_len = k.shape[-2]
            limit = self.n_sink + self.window_size

            # Use same buffer logic or strict limit? For PPL we usually want strict or consistent state.
            if current_len > limit:
                k_sink = k[:, :, : self.n_sink, :]
                k_window = k[:, :, -self.window_size :, :]
                self.layers[layer_idx].keys = torch.cat([k_sink, k_window], dim=-2)

                v_sink = v[:, :, : self.n_sink, :]
                v_window = v[:, :, -self.window_size :, :]
                self.layers[layer_idx].values = torch.cat([v_sink, v_window], dim=-2)

    def get_seq_length(self, layer_idx=0):
        if layer_idx in self._seen_tokens_by_layer:
            return self._seen_tokens_by_layer[layer_idx]
        return super().get_seq_length(layer_idx)


# --- H2O Cache Policy ---
class H2ODynamicCache(StreamingDynamicCache):
    """
    实现了 H2O (Heavy Hitter Oracle) 策略。
    保留: [Sinks] + [Heavy Hitters] + [Recent Window]
    """

    def __init__(
        self, config, n_sink=4, recent_window=32, max_capacity=256, debug=False
    ):
        # 注意：这里传给父类的 window_size 实际上是我们的 max_capacity (总预算)
        # 我们会重写 update 方法，所以父类的驱逐逻辑不会被执行，但我们需要它的结构
        super().__init__(config, n_sink=n_sink, window_size=max_capacity, debug=debug)

        self.max_capacity = max_capacity
        self.recent_window = recent_window
        self.n_sink = n_sink
        self.heavy_hitter_size = max_capacity - n_sink - recent_window

        # 存储每层的累积注意力分数
        # Key: layer_idx, Value: Tensor [current_seq_len]
        self.accumulated_scores = {}

    def update_scores(self, attn_weights, layer_idx):
        """
        接收来自 Attention 层的分数并累加。
        attn_weights: [batch, heads, q_len, k_len]
        """
        if layer_idx not in self.accumulated_scores:
            return

        # 我们主要关注 KV 的重要性，也就是 dim=-1 (k_len)
        # 对 Batch, Heads, Queries 进行求和，得到每个 Key 被关注的总量
        # sum over (0, 1, 2) -> [k_len]

        # 稳健性处理：处理可能的维度差异
        dims_to_sum = [0, 1]
        if attn_weights.dim() == 4:
            dims_to_sum.append(2)  # sum over q_len

        step_scores = attn_weights.sum(dim=dims_to_sum)

        # 确保数据在同一个设备上
        current_scores = self.accumulated_scores[layer_idx]

        # 对齐长度 (处理可能的边界情况)
        if step_scores.shape[0] == current_scores.shape[0]:
            self.accumulated_scores[layer_idx] += step_scores.detach()
        else:
            # 这种情况通常发生在 prefill 阶段之后或者维度不匹配
            # 简单起见，如果长度不匹配（且不是由于驱逐引起的），我们跳过更新或截断
            min_len = min(step_scores.shape[0], current_scores.shape[0])
            self.accumulated_scores[layer_idx][:min_len] += step_scores[
                :min_len
            ].detach()

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs=None,
    ):
        # 1. 记录长度
        if layer_idx not in self._seen_tokens_by_layer:
            self._seen_tokens_by_layer[layer_idx] = 0
        self._seen_tokens_by_layer[layer_idx] += key_states.shape[-2]

        # 2. 初始化分数存储 (如果是该层第一次运行)
        if layer_idx not in self.accumulated_scores:
            # 初始化为 0，长度为 0
            self.accumulated_scores[layer_idx] = torch.zeros(
                0, device=key_states.device
            )

        # 3. 扩展分数 Tensor (为新进来的 tokens 补 0)
        # key_states 是新进来的 token，长度通常为 1 (生成阶段)
        new_token_count = key_states.shape[-2]
        zeros = torch.zeros(new_token_count, device=key_states.device)
        self.accumulated_scores[layer_idx] = torch.cat(
            [self.accumulated_scores[layer_idx], zeros], dim=0
        )

        # 4. 标准 Update (追加 KV)
        k_out, v_out = super(DynamicCache, self).update(
            key_states, value_states, layer_idx, cache_kwargs
        )

        # 5. 驱逐逻辑 (H2O 核心)
        current_len = k_out.shape[-2]

        # 调试信息打印
        if self.debug and layer_idx == 0:
            if self._seen_tokens_by_layer[layer_idx] % 100 == 0:
                print(
                    f"[H2O DEBUG] Step {self._seen_tokens_by_layer[layer_idx]} | Key Shape: {k_out.shape} | Max Capacity: {self.max_capacity}"
                )

        # 使用 Lazy Eviction: 只有当超出 buffer (64) 时才进行昂贵的 TopK 操作
        if current_len > self.max_capacity + 64:

            # --- 确定要保留的索引 ---
            indices = torch.arange(current_len, device=k_out.device)
            scores = self.accumulated_scores[layer_idx]

            # A. Sink Mask (前 n_sink 个)
            sink_mask = indices < self.n_sink

            # B. Recent Window Mask (最后 recent_window 个)
            window_mask = indices >= (current_len - self.recent_window)

            # C. Candidates (中间的部分)
            # 我们只能从中间部分挑 Heavy Hitters
            candidate_mask = ~(sink_mask | window_mask)
            candidate_indices = indices[candidate_mask]
            candidate_scores = scores[candidate_mask]

            # D. Select Heavy Hitters
            # 选分数最高的 k 个
            num_hh = self.heavy_hitter_size
            if num_hh > 0 and len(candidate_scores) > 0:
                # TopK
                top_k = min(num_hh, len(candidate_scores))
                _, top_indices_local = torch.topk(candidate_scores, k=top_k)
                hh_indices = candidate_indices[top_indices_local]
            else:
                hh_indices = torch.tensor([], device=k_out.device, dtype=torch.long)

            # E. 合并所有保留的索引
            keep_indices = torch.cat(
                [indices[sink_mask], hh_indices, indices[window_mask]]
            )

            # F. 排序 (保持时间顺序非常重要！)
            keep_indices, _ = torch.sort(keep_indices)

            # --- 执行裁剪 ---

            # 1. 裁剪 KV Cache
            k_new = k_out[:, :, keep_indices, :]
            v_new = v_out[:, :, keep_indices, :]

            if layer_idx < len(self.layers):
                self.layers[layer_idx].keys = k_new
                self.layers[layer_idx].values = v_new

            # 2. 同步裁剪 分数 Tensor (必须一一对应)
            self.accumulated_scores[layer_idx] = scores[keep_indices]

            if self.debug and layer_idx == 0:
                print(
                    f"[H2O EVICT] Layer 0: {current_len} -> {k_new.shape[-2]} tokens (Sink:{self.n_sink}, HH:{len(hh_indices)}, Recent:{self.recent_window})"
                )

            return k_new, v_new

        return k_out, v_out

    def evict_all_layers(self):
        """Force eviction on all layers (for PPL evaluation)"""
        for layer_idx in range(len(self.layers)):
            if (
                not hasattr(self.layers[layer_idx], "keys")
                or self.layers[layer_idx].keys is None
            ):
                continue

            k = self.layers[layer_idx].keys
            v = self.layers[layer_idx].values

            current_len = k.shape[-2]

            if current_len > self.max_capacity:
                # 使用相同的 H2O 逻辑
                indices = torch.arange(current_len, device=k.device)
                scores = self.accumulated_scores[layer_idx]

                sink_mask = indices < self.n_sink
                window_mask = indices >= (current_len - self.recent_window)
                candidate_mask = ~(sink_mask | window_mask)

                candidate_indices = indices[candidate_mask]
                candidate_scores = scores[candidate_mask]

                num_hh = self.heavy_hitter_size
                if num_hh > 0 and len(candidate_scores) > 0:
                    top_k = min(num_hh, len(candidate_scores))
                    _, top_indices_local = torch.topk(candidate_scores, k=top_k)
                    hh_indices = candidate_indices[top_indices_local]
                else:
                    hh_indices = torch.tensor([], device=k.device, dtype=torch.long)

                keep_indices = torch.cat(
                    [indices[sink_mask], hh_indices, indices[window_mask]]
                )
                keep_indices, _ = torch.sort(keep_indices)

                self.layers[layer_idx].keys = k[:, :, keep_indices, :]
                self.layers[layer_idx].values = v[:, :, keep_indices, :]
                self.accumulated_scores[layer_idx] = scores[keep_indices]


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
        key_states, value_states = layer_past.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

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

    # [H2O] 强制获取 attention weights 用于反馈
    force_output_attentions = output_attentions or (
        layer_past is not None and hasattr(layer_past, "update_scores")
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        head_mask=head_mask,
        output_attentions=force_output_attentions,  # [H2O] 显式请求权重
        **kwargs,
    )

    # [H2O 核心逻辑] 反馈闭环：把 Attention 分数传回给 Cache
    if (
        layer_past is not None
        and hasattr(layer_past, "update_scores")
        and attn_weights is not None
    ):
        layer_past.update_scores(attn_weights, self.layer_idx)

    # --- TIMING RECORD ---
    if start_t is not None:
        torch.cuda.synchronize()
        end_t = time.time()
        ATTENTION_TIMES.append((end_t - start_t) * 1000)  # ms
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
                attn.forward = types.MethodType(
                    patched_gpt_neox_attention_forward, attn
                )


def unpatch_attention_layers(model):
    if hasattr(model, "gpt_neox"):
        for layer in model.gpt_neox.layers:
            attn = layer.attention
            if hasattr(attn, "_original_forward_streaming"):
                attn.forward = attn._original_forward_streaming
                del attn._original_forward_streaming


def enable_streaming_llm(
    model: GPTNeoXForCausalLM, n_sink=4, window_size=256, debug=False
):
    """Enables StreamingLLM by patching model.forward to inject StreamingDynamicCache."""
    if hasattr(model, "_original_forward_streaming_patch"):
        model._streaming_config = (n_sink, window_size, debug)
    else:
        model._original_forward_streaming_patch = model.forward
        model._streaming_config = (n_sink, window_size, debug)

        def streaming_forward(
            self, input_ids=None, past_key_values=None, use_cache=None, **kwargs
        ):
            n_sink, window_size, debug = self._streaming_config

            # Inject StreamingDynamicCache if starting fresh or if standard Cache is passed
            if use_cache:
                if past_key_values is None:
                    past_key_values = StreamingDynamicCache(
                        self.config, n_sink=n_sink, window_size=window_size, debug=debug
                    )
                elif isinstance(past_key_values, DynamicCache) and not isinstance(
                    past_key_values, StreamingDynamicCache
                ):
                    # Replace with Streaming Cache if it's empty/new
                    if past_key_values.get_seq_length() == 0:
                        past_key_values = StreamingDynamicCache(
                            self.config,
                            n_sink=n_sink,
                            window_size=window_size,
                            debug=debug,
                        )

            return self._original_forward_streaming_patch(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        model.forward = types.MethodType(streaming_forward, model)

    # Also patch attention layers for timing (optional, but requested)
    patch_attention_layers(model)


def disable_streaming_llm(model):
    if hasattr(model, "_original_forward_streaming_patch"):
        model.forward = model._original_forward_streaming_patch
        del model._original_forward_streaming_patch
    unpatch_attention_layers(model)


def enable_h2o_llm(
    model: GPTNeoXForCausalLM, n_sink=4, recent_window=32, max_capacity=256, debug=False
):
    """启用 H2O 模式"""
    # 存储配置
    model._h2o_config = (n_sink, recent_window, max_capacity, debug)

    # 如果还没有 patch 过 forward，先 patch
    if not hasattr(model, "_original_forward_streaming_patch"):
        model._original_forward_streaming_patch = model.forward

        def streaming_forward_wrapper(
            self, input_ids=None, past_key_values=None, use_cache=None, **kwargs
        ):
            # 获取配置
            if hasattr(self, "_h2o_config"):
                n_sink, recent_window, max_capacity, debug = self._h2o_config
                CacheClass = H2ODynamicCache
                cache_args = {
                    "n_sink": n_sink,
                    "recent_window": recent_window,
                    "max_capacity": max_capacity,
                    "debug": debug,
                }
            else:
                # 回退到 StreamingLLM 配置（如果存在）
                if hasattr(self, "_streaming_config"):
                    n_sink, window_size, debug = self._streaming_config
                    CacheClass = StreamingDynamicCache
                    cache_args = {
                        "n_sink": n_sink,
                        "window_size": window_size,
                        "debug": debug,
                    }
                else:
                    # 默认不使用任何特殊 Cache
                    return self._original_forward_streaming_patch(
                        input_ids=input_ids,
                        past_key_values=past_key_values,
                        use_cache=use_cache,
                        **kwargs,
                    )

            # 注入 Cache
            if use_cache:
                if past_key_values is None:
                    past_key_values = CacheClass(self.config, **cache_args)
                elif isinstance(past_key_values, DynamicCache) and not isinstance(
                    past_key_values, CacheClass
                ):
                    if past_key_values.get_seq_length() == 0:
                        past_key_values = CacheClass(self.config, **cache_args)

            return self._original_forward_streaming_patch(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        model.forward = types.MethodType(streaming_forward_wrapper, model)
    else:
        # 如果已经 patch 过了，只更新配置
        model._h2o_config = (n_sink, recent_window, max_capacity, debug)

    # 确保 Attention 层被 patch 过了
    patch_attention_layers(model)
