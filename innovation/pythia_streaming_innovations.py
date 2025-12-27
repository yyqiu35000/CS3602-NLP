import torch
import time
import types
from typing import Optional, Callable, List, Tuple
from transformers.cache_utils import DynamicCache, Cache
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM, eager_attention_forward, ALL_ATTENTION_FUNCTIONS, apply_rotary_pos_emb

# --- Import Global State for Attention Timing from Patch ---
# This ensures both original and innovation scripts share the same timing registry
from pythia_streaming_patch import (
    ATTENTION_TIMES,
    TIMING_ENABLED,
    reset_attention_timing,
    enable_attention_timing_collection,
    disable_attention_timing_collection,
    get_attention_stats,
    get_raw_attention_times
)

# Note: Since we import these, we don't need to redefine them.
# However, we need to make sure we append to the imported list.
# Since lists are mutable, importing ATTENTION_TIMES and appending to it works.
# But TIMING_ENABLED is a bool (immutable), so importing it directly won't allow us to modify the global in the other module easily
# unless we use the setter functions.
# The timing functions (reset, enable, disable) imported from patch module operate on the patch module's globals.
# So we should use those functions instead of accessing globals directly if possible.
# In patched_attention_forward_innovations, we check TIMING_ENABLED. 
# We need to read the value from the patch module.

import pythia_streaming_patch

# --- Innovation 2: POS Aware Streaming Cache ---
class POSStreamingCache(DynamicCache):
    """
    创新方向2：基于词性的语义感知驱逐
    保留: [Sink] + [Non-Stop-Words in Middle] + [Recent Window]
    """
    def __init__(self, config, tokenizer, n_sink=4, window_size=256, semantic_size=128, debug=False):
        super().__init__(config=config)
        self.n_sink = n_sink
        self.window_size = window_size
        self.semantic_size = semantic_size
        self.debug = debug
        self._seen_tokens_by_layer = {}
        
        # 停用词/功能词集合 (简单定义)
        # 常见英语功能词：the, a, an, of, in, to, is, are, ...
        # 我们用 tokenizer 编码一些常见词
        stop_words = ["the", "a", "an", "of", "in", "to", "is", "are", "it", "that", "this", "on", "for", "with", "as", "was", "at", "by", "\n", "."]
        self.stop_ids = set()
        for w in stop_words:
            # 注意：tokenizer 编码可能会带空格前缀，这里简单处理
            ids = tokenizer.encode(w, add_special_tokens=False)
            if ids: self.stop_ids.add(ids[0])
            ids_space = tokenizer.encode(" " + w, add_special_tokens=False)
            if ids_space: self.stop_ids.add(ids_space[0])
            
        # 维护当前 Cache 对应的 Token IDs
        # 假设所有层同步，我们只维护一份全局列表
        self.cached_token_ids = [] 

    def register_input_ids(self, input_ids: torch.Tensor):
        """记录新输入的 token ids"""
        # input_ids: [bsz, 1] usually
        ids_list = input_ids[0].tolist()
        self.cached_token_ids.extend(ids_list)

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        if layer_idx not in self._seen_tokens_by_layer:
            self._seen_tokens_by_layer[layer_idx] = 0
        self._seen_tokens_by_layer[layer_idx] += key_states.shape[-2]
        
        # 标准更新
        k_out, v_out = super().update(key_states, value_states, layer_idx, cache_kwargs)
        
        # 驱逐逻辑
        current_len = k_out.shape[-2]
        limit = self.n_sink + self.semantic_size + self.window_size
        
        # 确保 cached_token_ids 长度匹配 (仅在 layer 0 检查，防止多层重复报错)
        # 注意：这里有个同步问题，register_input_ids 是在 forward 前调用的，
        # 所以此时 cached_token_ids 应该已经包含了新 token。
        # 如果是首次预热，可能一次性进来很多 token。
        if layer_idx == 0:
             # 如果 cached_token_ids 比 current_len 短，说明没 register 对，或者 prefill 阶段
             pass

        if current_len > limit + 64:
            sink_end = self.n_sink
            window_start = current_len - self.window_size
            
            # 1. Sink
            k_sink = k_out[:, :, :sink_end, :]
            v_sink = v_out[:, :, :sink_end, :]
            
            # 2. Window
            k_window = k_out[:, :, window_start:, :]
            v_window = v_out[:, :, window_start:, :]
            
            # 3. Candidates
            k_cands = k_out[:, :, sink_end:window_start, :]
            v_cands = v_out[:, :, sink_end:window_start, :]
            
            # 筛选 Semantic (Non-Stop-Words)
            # 对应的 IDs
            # 注意：cached_token_ids 的长度应该等于 current_len (理想情况下)
            # 如果不等于，回退到普通 Streaming
            if len(self.cached_token_ids) == current_len:
                cand_ids = self.cached_token_ids[sink_end:window_start]
                
                # 找出非停用词的索引
                # 我们希望保留 'semantic_size' 个非停用词
                # 如果非停用词不够，就用停用词凑
                
                # 评分：非停用词=1，停用词=0。或者保留原始顺序
                keep_indices = []
                # 优先挑非停用词
                non_stop_indices = [i for i, tid in enumerate(cand_ids) if tid not in self.stop_ids]
                stop_indices = [i for i, tid in enumerate(cand_ids) if tid in self.stop_ids]
                
                # 截取
                needed = self.semantic_size
                picked_indices = non_stop_indices[:needed]
                if len(picked_indices) < needed:
                    # 不够，用停用词补（补最近的）
                    remain = needed - len(picked_indices)
                    picked_indices.extend(stop_indices[-remain:])
                
                picked_indices = sorted(picked_indices)
                picked_tensor = torch.tensor(picked_indices, device=k_out.device)
                
                k_semantic = k_cands.index_select(2, picked_tensor)
                v_semantic = v_cands.index_select(2, picked_tensor)
                
                # 更新 cached_token_ids
                # 新列表 = Sink + Picked_Middle + Window
                if layer_idx == 0: # 只在第0层更新全局列表
                    new_ids = (
                        self.cached_token_ids[:sink_end] + 
                        [cand_ids[i] for i in picked_indices] + 
                        self.cached_token_ids[window_start:]
                    )
                    self.cached_token_ids = new_ids
            else:
                # Fallback to simple middle truncation
                k_semantic = k_cands[:, :, -self.semantic_size:, :]
                v_semantic = v_cands[:, :, -self.semantic_size:, :]
                if layer_idx == 0:
                     # 简单截断 IDs
                     self.cached_token_ids = (
                         self.cached_token_ids[:sink_end] + 
                         self.cached_token_ids[sink_end:window_start][-self.semantic_size:] +
                         self.cached_token_ids[window_start:]
                     )

            k_new = torch.cat([k_sink, k_semantic, k_window], dim=-2)
            v_new = torch.cat([v_sink, v_semantic, v_window], dim=-2)
            
            if layer_idx < len(self.layers):
                self.layers[layer_idx].keys = k_new
                self.layers[layer_idx].values = v_new
            
            if self.debug and layer_idx == 0 and self._seen_tokens_by_layer[layer_idx] % 100 == 0:
                print(f"DEBUG POS: Step {self._seen_tokens_by_layer[layer_idx]} | Pruned {current_len} -> {k_new.shape[-2]}")
                
            return k_out, v_out
            
        return k_out, v_out

    def get_seq_length(self, layer_idx=0):
        if layer_idx in self._seen_tokens_by_layer:
            return self._seen_tokens_by_layer[layer_idx]
        return super().get_seq_length(layer_idx)

# --- Innovation 3: Semantic Block Streaming Cache (RAG Style) ---
class SemanticBlockStreamingCache(DynamicCache):
    """
    创新方向3：基于语义块的检索 (RAG 思想) + 层级自适应窗口 (Adaptive Window)
    保留: [Sink] + [Relevant Blocks (RAG)] + [Recent Window]
    
    策略：
    - 底层 (0-10): 强局部性，大 Window (480), 小 Semantic (32)
    - 中层 (11-21): 均衡，中 Window (256), 中 Semantic (256)
    - 高层 (22-31): 强语义，小 Window (128), 大 Semantic (384)
    """
    def __init__(self, config, n_sink=4, base_window_size=256, base_semantic_size=256, block_size=16, compress=False, semantic_dynamic=False, debug=False):
        super().__init__()
        self.config = config
        self.n_sink = n_sink
        self.block_size = block_size
        self.compress = compress  # 是否启用块压缩 (Mean Pooling)
        self.semantic_dynamic = semantic_dynamic # 是否启用基于语义相似度的动态分块
        self.debug = debug
        self._seen_tokens_by_layer = {}
        
        # 存储 Block Centroids (语义中心)
        # key: layer_idx, value: list of Tensors [1, head_dim]
        self.block_centroids = {}
        # 存储当前的 Query 用于检索
        self.current_query = {}
        
        # 存储上一轮的 Semantic 长度 (用于区分 Compressed 和 Raw 边界)
        self.prev_semantic_len = {}

        # 预计算每层的配置
        self.layer_configs = {}
        num_layers = config.num_hidden_layers
        
        # total_budget = base_window_size + base_semantic_size # e.g. 512
        
        for i in range(num_layers):
            # 禁用层级自适应逻辑，强制使用用户指定的参数
            # if i < num_layers // 3:
            #     # Bottom Layers: Focus on Local Context
            #     w = int(total_budget * 0.9) # ~460
            #     s = total_budget - w
            # elif i < 2 * num_layers // 3:
            #     # Middle Layers: Balanced
            #     w = int(total_budget * 0.5) # ~256
            #     s = total_budget - w
            # else:
            #     # Top Layers: Focus on Long-range Semantic
            #     w = int(total_budget * 0.25) # ~128
            #     s = total_budget - w
            
            # 使用固定参数
            w = base_window_size
            s = base_semantic_size

            # 确保 semantic size 是 block_size 的倍数
            s = (s // block_size) * block_size
            
            self.layer_configs[i] = {
                "window_size": w,
                "max_blocks": max(1, s // block_size)
            }
            # if debug:
            #     print(f"Layer {i}: Window={w}, Semantic={s} ({s//block_size} blocks)")

    def set_current_query(self, layer_idx, query_states):
        """记录当前 Query 用于计算与 Block 的相似度"""
        # query_states: [bsz, num_heads, seq_len, head_dim]
        # 总是取最后一个 token 的 Query，并平均 Heads
        if query_states.shape[-2] >= 1:
             self.current_query[layer_idx] = query_states[:, :, -1, :].mean(dim=1).detach()

    def update(self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int, cache_kwargs=None):
        if layer_idx not in self._seen_tokens_by_layer:
            self._seen_tokens_by_layer[layer_idx] = 0
            self.block_centroids[layer_idx] = []
        
        self._seen_tokens_by_layer[layer_idx] += key_states.shape[-2]
        
        # 获取当前层的配置
        config = self.layer_configs.get(layer_idx, self.layer_configs[0]) # fallback
        window_size = config["window_size"]
        max_blocks = config["max_blocks"]
        
        # 标准更新
        k_out, v_out = super().update(key_states, value_states, layer_idx, cache_kwargs)
        
        # 驱逐逻辑
        current_len = k_out.shape[-2]
        
        # 根据压缩模式决定 limit
        # compress: limit = sink + max_blocks(tokens) + window
        # raw: limit = sink + max_blocks * block_size + window
        if self.compress:
            limit = self.n_sink + max_blocks + window_size
        else:
            limit = self.n_sink + (max_blocks * self.block_size) + window_size

        # Debug Print per 50 tokens (regardless of eviction)
        step = self._seen_tokens_by_layer[layer_idx]
        if self.debug and layer_idx == 0 and step % 50 == 0:
             print(f"DEBUG: Step {step} | Current KV: {current_len} | Limit: {limit}")

        if current_len > limit + 64: # buffer
            sink_end = self.n_sink
            window_start = current_len - window_size
            
            # 1. Sink
            k_sink = k_out[:, :, :sink_end, :]
            v_sink = v_out[:, :, :sink_end, :]
            
            # 2. Window
            k_window = k_out[:, :, window_start:, :]
            v_window = v_out[:, :, window_start:, :]
            
            # 3. Middle Candidates
            k_cands = k_out[:, :, sink_end:window_start, :]
            v_cands = v_out[:, :, sink_end:window_start, :]
            
            k_semantic = None
            v_semantic = None

            if self.compress:
                # --- Compressed Logic ---
                # k_cands 混合了 Old Compressed (0:prev) 和 New Raw (prev:)
                prev_len = self.prev_semantic_len.get(layer_idx, 0)
                
                # 边界检查: 如果是第一次 (prefill) 或者 reset，prev_len 可能不适用，假设全 Raw
                # Prefill 阶段通常 seq_len > 1，且 prev_len=0
                # 如果是 Generation 阶段，prev_len > 0
                
                # 分离
                if prev_len > 0 and prev_len < k_cands.shape[-2]:
                    k_old_compressed = k_cands[:, :, :prev_len, :]
                    v_old_compressed = v_cands[:, :, :prev_len, :]
                    
                    k_new_raw = k_cands[:, :, prev_len:, :]
                    v_new_raw = v_cands[:, :, prev_len:, :]
                else:
                    # 全 Raw (Prefill) 或 异常
                    k_old_compressed = k_cands[:, :, :0, :] # empty
                    v_old_compressed = v_cands[:, :, :0, :]
                    k_new_raw = k_cands
                    v_new_raw = v_cands
                
                # 压缩 New Raw
                num_new_raw = k_new_raw.shape[-2]
                num_new_blocks = num_new_raw // self.block_size
                
                # 处理 Remainder: 如果有余数，我们不能压缩，必须把它们推回 Window
                remainder = num_new_raw % self.block_size
                if remainder > 0:
                    # 将余数归还给 Window
                    # 注意：这意味着 window 变大了 `remainder`
                    k_window = torch.cat([k_new_raw[:, :, -remainder:, :], k_window], dim=-2)
                    v_window = torch.cat([v_new_raw[:, :, -remainder:, :], v_window], dim=-2)
                    
                    # 截断 k_new_raw
                    k_new_raw = k_new_raw[:, :, :-remainder, :]
                    v_new_raw = v_new_raw[:, :, :-remainder, :]
                    num_new_blocks = k_new_raw.shape[-2] // self.block_size
                
                # 压缩 New Raw Blocks
                if num_new_blocks > 0:
                    k_new_blocks = k_new_raw.view(k_new_raw.shape[0], k_new_raw.shape[1], num_new_blocks, self.block_size, -1)
                    v_new_blocks = v_new_raw.view(v_new_raw.shape[0], v_new_raw.shape[1], num_new_blocks, self.block_size, -1)
                    
                    if self.semantic_dynamic:
                        # --- Dynamic Compression based on Key Similarity ---
                        # 计算块内 token 的均值，但这里的块其实是按 block_size 划分的粗粒度单元
                        # 如果要真正的 dynamic，我们需要在 view 之前操作
                        # 为了性能，我们采用一种 Hierarchical Approach：
                        # 先把数据划分为 block_size/2 的小单元，然后基于相似度两两合并
                        
                        # 但为了响应 "不一定要16组固定压缩为一块"，我们实现一个更简单的：
                        # 将数据 reshape 为 (N*2, block_size/2, D)，然后计算相邻 Block 的相似度
                        # 合并最相似的。
                        
                        # 简化实现：Variational Pooling
                        # 在 Block 内部，计算 Key 的 Variance。
                        # 如果 Variance 大，说明这个 Block 包含丰富信息，也许我们应该保留更多（比如拆成2个）
                        # 如果 Variance 小，说明是冗余信息，可以合并。
                        # 但由于输出必须是 Fixed Size (num_new_blocks)，这是一个分配问题。
                        
                        # 极简动态方案：Similarity Weighted Mean
                        # 虽然输出还是 1 个向量，但不是简单的算术平均，而是加权平均。
                        # 权重取决于该 Token 与 Block Center 的相似度？
                        # 不，这改变不了存储效率。
                        
                        # 回归用户需求： "根据语义相似性动态压缩"
                        # 意思是：边界不固定。
                        # 算法：
                        # 1. 计算所有相邻 Token 的 Cosine Sim (k_new_raw)
                        # 2. 找出最不相似的 (num_new_blocks - 1) 个边界点 (Peaks in 1-Sim)
                        # 3. 根据边界切分
                        # 4. Pool 每个片段
                        
                        # 这比 view().mean() 慢，但符合需求。
                        # 仅对 k_new_raw (通常较小，如 64-128 tokens) 操作，速度可接受。
                        
                        # 1. Calc Adjacency Similarity (Cosine)
                        # k_new_raw: [bsz, heads, seq, dim]
                        # norm: [bsz, heads, seq, 1]
                        k_norm = k_new_raw / (k_new_raw.norm(dim=-1, keepdim=True) + 1e-6)
                        # sim: [bsz, heads, seq-1]
                        sims = (k_norm[:, :, :-1] * k_norm[:, :, 1:]).sum(dim=-1)
                        # dissim = 1 - sim
                        dissims = 1.0 - sims
                        
                        # 2. Find Boundaries
                        # 我们需要 num_new_blocks 个块，所以需要 num_new_blocks - 1 个切分点
                        # 也就是找到 dissim 最大的前 (num_new_blocks - 1) 个位置
                        k_req = num_new_blocks - 1
                        if k_req > 0 and k_req < dissims.shape[-1]:
                            # Sum over heads to find common boundaries across heads (faster & stable)
                            # dissims_avg: [bsz, seq-1]
                            dissims_avg = dissims.mean(dim=1) 
                            
                            # Top-K boundaries
                            # values, indices: [bsz, k_req]
                            _, boundary_indices = torch.topk(dissims_avg, k_req, dim=-1)
                            boundary_indices, _ = torch.sort(boundary_indices)
                            
                            # 3. Segment & Pool
                            # 由于 Tensor 操作难以处理变长切片，我们使用循环 (Batch size 通常为 1)
                            # 优化：Masked Sum
                            
                            # 构建 Segment IDs [bsz, seq]
                            # 初始全 0，在 boundary_indices + 1 处 +1
                            # boundary_indices 指向的是 "Gap"，即 index i 和 i+1 之间
                            # 所以第 i 个 token 属于 segment sum(boundary_indices < i)
                            
                            # 构造 mask: [bsz, seq]
                            # boundaries map to indices in [0, seq-2]
                            # seq_len = k_new_raw.shape[-2]
                            bsz, heads, seq_len, head_dim = k_new_raw.shape
                            
                            # segment_ids: [bsz, seq_len]
                            # 比较慢的方法，但逻辑清晰
                            k_new_compressed_list = []
                            v_new_compressed_list = []
                            
                            for b in range(bsz):
                                boundaries = boundary_indices[b].tolist()
                                boundaries = [-1] + boundaries + [seq_len - 1]
                                
                                k_segs = []
                                v_segs = []
                                
                                for i in range(len(boundaries) - 1):
                                    start = boundaries[i] + 1
                                    end = boundaries[i+1] + 1
                                    # Pool [start:end]
                                    k_seg = k_new_raw[b, :, start:end, :].mean(dim=1, keepdim=True) # [heads, 1, dim]
                                    v_seg = v_new_raw[b, :, start:end, :].mean(dim=1, keepdim=True)
                                    k_segs.append(k_seg)
                                    v_segs.append(v_seg)
                                    
                                k_new_compressed_list.append(torch.cat(k_segs, dim=1)) # [heads, num_blocks, dim]
                                v_new_compressed_list.append(torch.cat(v_segs, dim=1))
                            
                            k_new_compressed = torch.stack(k_new_compressed_list, dim=0) # [bsz, heads, num_blocks, dim]
                            v_new_compressed = torch.stack(v_new_compressed_list, dim=0)
                            
                        else:
                             # Fallback if too few tokens
                            k_new_compressed = k_new_raw.mean(dim=2, keepdim=True) # Pool all
                            v_new_compressed = v_new_raw.mean(dim=2, keepdim=True)
                            
                    else:
                        # Standard Fixed Stride Pooling
                        k_new_compressed = k_new_blocks.mean(dim=3)
                        v_new_compressed = v_new_blocks.mean(dim=3)
                else:
                    k_new_compressed = k_new_raw[:, :, :0, :] # empty
                    v_new_compressed = v_new_raw[:, :, :0, :]

                # 合并 Candidates (全 Compressed)
                k_combined = torch.cat([k_old_compressed, k_new_compressed], dim=-2)
                v_combined = torch.cat([v_old_compressed, v_new_compressed], dim=-2)
                
                # 执行选择 (Top-K)
                # 注意：此时 k_combined 中的每个元素都是一个 Block (Compressed Token)
                # Centroids 就是它们自己 (因为已经 Mean 过了)
                centroids = k_combined.mean(dim=1) # [bsz, num_total_blocks, head_dim] (mean over heads)
                
                num_total_blocks = k_combined.shape[-2]
                
                if layer_idx in self.current_query and num_total_blocks > 0:
                    query = self.current_query[layer_idx].unsqueeze(1) # [bsz, 1, head_dim]
                    scores = torch.matmul(centroids, query.transpose(1, 2)).squeeze(-1) # [bsz, num_blocks]
                    
                    k_needed = min(max_blocks, num_total_blocks)
                    _, top_indices = torch.topk(scores, k_needed, dim=1)
                    top_indices, _ = torch.sort(top_indices)
                    
                    k_semantic = k_combined.index_select(2, top_indices[0])
                    v_semantic = v_combined.index_select(2, top_indices[0])
                else:
                    # Fallback: Keep latest
                    k_semantic = k_combined[:, :, -max_blocks:, :]
                    v_semantic = v_combined[:, :, -max_blocks:, :]

            else:
                # --- Original Raw Logic (Unchanged but adapted) ---
                num_cands = k_cands.shape[-2]
                num_blocks = num_cands // self.block_size
                
                if num_blocks > 0:
                    k_blocks = k_cands[:, :, :num_blocks*self.block_size, :].view(
                        k_cands.shape[0], k_cands.shape[1], num_blocks, self.block_size, -1
                    )
                    v_blocks = v_cands[:, :, :num_blocks*self.block_size, :].view(
                        v_cands.shape[0], v_cands.shape[1], num_blocks, self.block_size, -1
                    )
                    
                    centroids = k_blocks.mean(dim=3).mean(dim=1)
                    
                    if layer_idx in self.current_query:
                        query = self.current_query[layer_idx].unsqueeze(1)
                        scores = torch.matmul(centroids, query.transpose(1, 2)).squeeze(-1)
                        
                        k_needed = min(max_blocks, num_blocks)
                        _, top_indices = torch.topk(scores, k_needed, dim=1)
                        top_indices, _ = torch.sort(top_indices)
                        
                        k_selected = k_blocks.index_select(2, top_indices[0])
                        v_selected = v_blocks.index_select(2, top_indices[0])
                        
                        k_semantic = k_selected.view(k_cands.shape[0], k_cands.shape[1], -1, k_cands.shape[-1])
                        v_semantic = v_selected.view(v_cands.shape[0], v_cands.shape[1], -1, v_cands.shape[-1])
                    else:
                        k_semantic = k_cands[:, :, -max_blocks*self.block_size:, :]
                        v_semantic = v_cands[:, :, -max_blocks*self.block_size:, :]
                else:
                    k_semantic = k_cands[:, :, :0, :]
                    v_semantic = v_cands[:, :, :0, :]

            # Update prev_semantic_len
            self.prev_semantic_len[layer_idx] = k_semantic.shape[-2]

            k_new = torch.cat([k_sink, k_semantic, k_window], dim=-2)
            v_new = torch.cat([v_sink, v_semantic, v_window], dim=-2)
            
            if self.debug and layer_idx == 0:
                semantic_count = k_semantic.shape[-2]
                semantic_blocks = semantic_count if self.compress else semantic_count // self.block_size
                print(f"DEBUG: Step {step} | Evicting -> Total KV: {k_new.shape[-2]} (Sink {k_sink.shape[-2]} + Semantic {k_semantic.shape[-2]} + Window {k_window.shape[-2]}) | Semantic Blocks: {semantic_blocks} ({'Compressed' if self.compress else 'Raw'})")
            
            if layer_idx < len(self.layers):
                self.layers[layer_idx].keys = k_new
                self.layers[layer_idx].values = v_new
            
            return k_out, v_out

        return k_out, v_out

    def get_seq_length(self, layer_idx=0):
        if layer_idx in self._seen_tokens_by_layer:
            return self._seen_tokens_by_layer[layer_idx]
        return super().get_seq_length(layer_idx)

# --- Patching Functions ---

def patched_attention_forward_innovations(
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
    # 强制输出 attention 以便 H2O 获取分数
    force_output_attentions = True 
    
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, 3 * self.head_size)

    qkv = self.query_key_value(hidden_states).view(hidden_shape).transpose(1, 2)
    query_states, key_states, value_states = qkv.chunk(3, dim=-1)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if layer_past is not None:
        # --- INNOVATION HOOK: Register Query for Semantic Block ---
        # MUST be done BEFORE layer_past.update() which triggers eviction
        if isinstance(layer_past, SemanticBlockStreamingCache):
            layer_past.set_current_query(self.layer_idx, query_states)
        # ---------------------------------------------------

        cache_kwargs = {
            "sin": sin,
            "cos": cos,
            "partial_rotation_size": self.rotary_ndims,
            "cache_position": cache_position,
        }
        key_states, value_states = layer_past.update(key_states, value_states, self.layer_idx, cache_kwargs) 

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    # --- TIMING BLOCK ---
    start_t = None
    if pythia_streaming_patch.TIMING_ENABLED and hidden_states.shape[1] == 1:
        torch.cuda.synchronize()
        start_t = time.time()
    # --------------------

    # Call attention
    need_weights = output_attentions

    # --- StreamingLLM Fix: Manual Mask Construction (Synced with patch) ---
    # 解决 RoPE 需要逻辑位置而 Mask 需要物理位置的冲突
    if isinstance(layer_past, (SemanticBlockStreamingCache, POSStreamingCache)):
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
            
            # Ensure is_causal is False because we are providing a manual mask
            if "is_causal" in kwargs:
                kwargs["is_causal"] = False
        else:
            # [Optimization] Decoding 阶段 (q_len=1)
            # 因为 Cache 中的所有 Token (Sink + Window) 对当前 Query 都是可见的 (都在过去)
            # 所以我们不需要任何 Mask (全 0 Mask 等价于 None)
            # 将 Mask 置为 None 以启用底层 "No Mask" 优化路径 (MatMul only)
            attention_mask = None

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        head_mask=head_mask,
        output_attentions=need_weights,
        **kwargs,
    )

    # --- TIMING RECORD ---
    if start_t is not None:
        torch.cuda.synchronize()
        end_t = time.time()
        pythia_streaming_patch.ATTENTION_TIMES.append((end_t - start_t) * 1000)
    # ---------------------
    
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.dense(attn_output)
    
    # Return original signature (unless user requested attentions, but usually model.forward doesn't expect them from layer)
    # The layer output is usually (hidden_states, attentions) if output_attentions is True.
    # GPTNeoXLayer handles this.
    return attn_output, attn_weights

def patch_attention_layers_innovations(model):
    if hasattr(model, "gpt_neox"):
        for layer in model.gpt_neox.layers:
            attn = layer.attention
            if not hasattr(attn, "_original_forward_innovations"):
                attn._original_forward_innovations = attn.forward
                attn.forward = types.MethodType(patched_attention_forward_innovations, attn)

def unpatch_attention_layers_innovations(model):
    if hasattr(model, "gpt_neox"):
        for layer in model.gpt_neox.layers:
            attn = layer.attention
            if hasattr(attn, "_original_forward_innovations"):
                attn.forward = attn._original_forward_innovations
                del attn._original_forward_innovations

def enable_innovation_llm(
    model: GPTNeoXForCausalLM, 
    tokenizer=None,
    method="heavy_hitter", # or "pos_aware", "semantic_block"
    n_sink=4, 
    window_size=256, 
    extra_size=128, # heavy_size or semantic_size
    compress=False, # Add compress param
    semantic_dynamic=False, # Add dynamic param
    debug=False
):
    if hasattr(model, "_original_forward_innovations"):
        model._innovation_config = (method, n_sink, window_size, extra_size, debug, tokenizer, compress, semantic_dynamic)
    else:
        model._original_forward_innovations = model.forward
        model._innovation_config = (method, n_sink, window_size, extra_size, debug, tokenizer, compress, semantic_dynamic)

        def innovation_forward(self, input_ids=None, past_key_values=None, use_cache=None, **kwargs):
            method, n_sink, window_size, extra_size, debug, tokenizer, compress, semantic_dynamic = self._innovation_config
            
            # Inject Cache
            if use_cache:
                if past_key_values is None:
                    if method == "pos_aware":
                        past_key_values = POSStreamingCache(
                            self.config, tokenizer, n_sink=n_sink, window_size=window_size, semantic_size=extra_size, debug=debug
                        )
                    elif method == "semantic_block":
                         # default block_size=16
                         block_size = 16
                         # Pass base sizes for adaptive logic
                         past_key_values = SemanticBlockStreamingCache(
                            self.config, n_sink=n_sink, base_window_size=window_size, base_semantic_size=extra_size, block_size=block_size, compress=compress, semantic_dynamic=semantic_dynamic, debug=debug
                         )

                elif isinstance(past_key_values, DynamicCache) and not isinstance(past_key_values, (POSStreamingCache, SemanticBlockStreamingCache)):
                     if past_key_values.get_seq_length() == 0:
                        if method == "pos_aware":
                            past_key_values = POSStreamingCache(
                                self.config, tokenizer, n_sink=n_sink, window_size=window_size, semantic_size=extra_size, debug=debug
                            )
                        elif method == "semantic_block":
                             block_size = 16
                             past_key_values = SemanticBlockStreamingCache(
                                self.config, n_sink=n_sink, base_window_size=window_size, base_semantic_size=extra_size, block_size=block_size, compress=compress, semantic_dynamic=semantic_dynamic, debug=debug
                             )

            # --- INNOVATION HOOK: Register Input IDs for POS Cache ---
            if isinstance(past_key_values, POSStreamingCache) and input_ids is not None:
                past_key_values.register_input_ids(input_ids)
            # ---------------------------------------------------------

            return self._original_forward_innovations(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs
            )
        
        model.forward = types.MethodType(innovation_forward, model)
    
    patch_attention_layers_innovations(model)

def disable_innovation_llm(model):
    if hasattr(model, "_original_forward_innovations"):
        model.forward = model._original_forward_innovations
        del model._original_forward_innovations
    unpatch_attention_layers_innovations(model)
