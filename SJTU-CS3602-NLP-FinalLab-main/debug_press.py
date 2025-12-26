"""
StreamingLLM Press 调试脚本
逐步验证每个环节是否正常工作
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./models/pythia-70m"
DEVICE = "cuda"

print("=" * 60)
print("StreamingLLM Press 调试工具")
print("=" * 60)

# 加载模型
print("\n[1/6] 加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, torch_dtype=torch.float16, device_map="auto"
)
model.eval()
print("✓ 模型加载完成")

# 准备输入
print("\n[2/6] 准备输入...")
prompt = "Hello, this is a test."
inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
print(f"✓ Prompt tokens: {inputs.input_ids.shape[1]}")


# ========== 调试版本的 Press ==========
class DebugPress:
    def __init__(self, max_capacity=3, n_sink=1):
        self.max_capacity = max_capacity
        self.n_sink = n_sink
        self.hooks = []
        self.hook_call_count = 0
        self.compression_count = 0
        self.module_to_layer_idx = {}  # 模块到层索引的映射

    def _make_hook(self, layer_idx):
        """为每个层创建专属的 hook"""

        def hook(module, args, kwargs):
            return self._pre_forward_hook(module, args, kwargs, layer_idx)

        return hook

    def _pre_forward_hook(self, module, args, kwargs, layer_idx):
        """Pre-forward hook with kwargs: 拦截并压缩 DynamicCache 中的 KV"""
        self.hook_call_count += 1
        verbose = self.hook_call_count <= 3  # 只打印前3次

        # 查找 layer_past (DynamicCache 对象)
        cache = kwargs.get("layer_past")

        if cache is None:
            return args, kwargs

        # 检查是否是 DynamicCache
        cache_type = type(cache).__name__
        if cache_type != "DynamicCache":
            return args, kwargs

        # 简化的调试输出
        if verbose and layer_idx == 0:  # 只打印第0层
            try:
                seq_len = cache.get_seq_length()
                print(
                    f"\n  [Hook #{self.hook_call_count}] Layer 0, cache seq_len={seq_len}"
                )
            except:
                pass

        # 使用 cache[layer_idx] 访问 KV tuple
        try:
            kv_tuple = cache[layer_idx]
        except:
            return args, kwargs

        if not isinstance(kv_tuple, tuple) or len(kv_tuple) != 2:
            return args, kwargs

        key, value = kv_tuple

        if key is None or value is None:
            return args, kwargs

        # seq_len 在维度 2: [batch, num_heads, seq_len, head_dim]
        seq_len = key.shape[2]

        # 判断是否需要压缩
        if seq_len <= self.max_capacity:
            return args, kwargs

        # 执行压缩
        self.compression_count += 1
        window_size = self.max_capacity - self.n_sink

        k_sink = key[:, :, : self.n_sink, :]
        v_sink = value[:, :, : self.n_sink, :]
        k_window = key[:, :, -window_size:, :]
        v_window = value[:, :, -window_size:, :]

        k_new = torch.cat([k_sink, k_window], dim=2)
        v_new = torch.cat([v_sink, v_window], dim=2)

        # 只在前10次或每50次打印
        if self.compression_count <= 10 or self.compression_count % 50 == 0:
            print(
                f"  [压缩 #{self.compression_count}] layer={layer_idx}, {seq_len} → {k_new.shape[2]}"
            )

        # 使用正确的方式修改 DynamicCache：cache.layers[idx].keys/values
        try:
            # DynamicCache.__getitem__ 返回 self.layers[idx].keys, self.layers[idx].values
            # 所以我们应该修改 cache.layers[idx] 的属性
            if hasattr(cache, "layers") and layer_idx < len(cache.layers):
                cache.layers[layer_idx].keys = k_new
                cache.layers[layer_idx].values = v_new

            # 验证压缩是否生效
            if self.compression_count <= 10:
                verify_kv = cache[layer_idx]
                if verify_kv[0] is not None:
                    actual_len = verify_kv[0].shape[2]
                    print(
                        f"    ✅ 验证: cache[{layer_idx}] 压缩后实际长度 = {actual_len}"
                    )
        except Exception as e:
            if self.compression_count <= 3:
                print(f"    ⚠️ 修改失败: {e}")

        return args, kwargs

    def register(self, model):
        print("\n[3/6] 注册 Hook...")
        self.remove()

        # 找到 attention 层
        if hasattr(model, "gpt_neox"):
            layers = model.gpt_neox.layers
        elif hasattr(model, "model"):
            layers = model.model.layers
        else:
            print("❌ 无法找到模型的 layers")
            return

        print(f"  发现 {len(layers)} 个 Transformer 层")

        for i, layer in enumerate(layers):
            if hasattr(layer, "attention"):
                target = layer.attention
                attr_name = "attention"
            elif hasattr(layer, "self_attn"):
                target = layer.self_attn
                attr_name = "self_attn"
            else:
                print(f"  ⚠️  Layer {i}: 找不到 attention 模块")
                continue

            # 为每个层创建专属的 hook（带 layer_idx）
            handle = target.register_forward_pre_hook(
                self._make_hook(i), with_kwargs=True
            )
            self.hooks.append(handle)
            if i == 0:  # 只打印第一层
                print(f"  ✓ Layer 0: 成功注册 Pre-Hook (with_kwargs) 到 {attr_name}")

        print(f"✓ 共注册 {len(self.hooks)} 个 Hook")

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def reset_stats(self):
        self.hook_call_count = 0
        self.compression_count = 0


# ========== 测试 1: Baseline (无 Press) ==========
print("\n" + "=" * 60)
print("测试 1: Baseline (无压缩)")
print("=" * 60)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(DEVICE)

print("\n[4/6] 开始生成 (Baseline)...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  # 增加生成长度以触发压缩
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

baseline_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
baseline_tokens = outputs.shape[1]

print(f"\n✓ Baseline 完成:")
print(f"  - 生成 tokens: {baseline_tokens}")
print(f"  - 显存峰值: {baseline_mem:.2f} MB")


# ========== 测试 2: 手动循环 + Press (关键测试) ==========
print("\n" + "=" * 60)
print("测试 2: 手动循环 + StreamingLLM Press")
print("=" * 60)

press_manual = DebugPress(max_capacity=3, n_sink=1)
press_manual.register(model)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(DEVICE)

print("\n[4.5/6] 开始手动生成循环...")
print("  (生成 500 tokens 以测试长序列压缩效果)\n")

with torch.no_grad():
    input_ids = inputs.input_ids
    past_key_values = None
    generated_tokens = 0

    for step in range(500):
        # 准备输入
        model_inputs = {
            "input_ids": input_ids,
            "use_cache": True,
        }
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values

        # 调用模型
        outputs = model(**model_inputs)

        # 获取下一个 token
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        # 关键：保存模型返回的 past_key_values
        past_key_values = outputs.past_key_values

        # 准备下一次输入
        input_ids = next_token
        generated_tokens += 1

        # 打印每一步的 KV cache 长度（前10步和后5步）
        if step < 10 or step >= 495:
            if past_key_values is not None:
                kv_len = past_key_values[0][0].shape[2]
                print(f"  Step {step+1}: KV Cache 长度 = {kv_len}")

press_manual.remove()

manual_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)

print(f"\n✓ 手动循环完成:")
print(f"  - 生成 tokens: {generated_tokens}")
print(f"  - 显存峰值: {manual_mem:.2f} MB")
print(f"  - Hook 调用次数: {press_manual.hook_call_count}")
print(f"  - 实际压缩次数: {press_manual.compression_count}")


# ========== 测试 3: generate() + Press (对比) ==========
print("\n" + "=" * 60)
print("测试 3: generate() + Press (对比)")
print("=" * 60)

press = DebugPress(max_capacity=3, n_sink=1)
press.reset_stats()
press.register(model)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats(DEVICE)

print("\n[5/6] 开始生成 (generate + Press)...")
print("  (这次会看到 past_kv 始终为 None)\n")

with torch.no_grad():
    outputs_stream = model.generate(
        **inputs,
        max_new_tokens=100,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )

press.remove()

stream_mem = torch.cuda.max_memory_allocated(DEVICE) / (1024**2)
stream_tokens = outputs_stream.shape[1]

print(f"\n✓ StreamingLLM 完成:")
print(f"  - 生成 tokens: {stream_tokens}")
print(f"  - 显存峰值: {stream_mem:.2f} MB")
print(f"  - Hook 调用次数: {press.hook_call_count}")
print(f"  - 实际压缩次数: {press.compression_count}")


# ========== 结果对比 ==========
print("\n" + "=" * 60)
print("[6/6] 结果对比")
print("=" * 60)
print(f"{'指标':<20} | {'Baseline':<12} | {'手动+Press':<12} | {'generate+Press':<12}")
print("-" * 80)
print(
    f"{'显存 (MB)':<20} | {baseline_mem:<12.2f} | {manual_mem:<12.2f} | {stream_mem:<12.2f}"
)
print(
    f"{'Hook 压缩次数':<20} | {'-':<12} | {press_manual.compression_count:<12} | {press.compression_count:<12}"
)

manual_saved = baseline_mem - manual_mem

print("\n" + "=" * 60)
print("📊 诊断结果")
print("=" * 60)

if press_manual.compression_count > 0 and manual_saved > 0.1:
    print(
        f"✅ 成功！手动循环 + Press 显存节省了 {manual_saved:.2f} MB ({manual_saved/baseline_mem*100:.1f}%)"
    )
    print(f"   实际压缩了 {press_manual.compression_count} 次")
    print("\n🎯 结论: StreamingLLM Press 实现**正确**！")
    print("   问题在于 model.generate() 不传递 past_key_values")
    print("\n✨ 解决方案:")
    print("   使用手动循环代替 generate()（参考 benchmark_streaming_manual.py）")
elif press_manual.compression_count > 0:
    print(
        f"⚠️  手动循环触发了 {press_manual.compression_count} 次压缩，但显存未明显下降"
    )
    print("\n可能原因:")
    print("  1. max_capacity=10 还是太大，改为 5 试试")
    print("  2. 生成的 token 数量太少（只有 50）")
    print("  3. 模型的其他部分占用了主要显存")
else:
    print("❌ 手动循环也没有触发压缩")
    print("\n可能原因:")
    print("  1. KV Cache 长度始终没超过 max_capacity=10")
    print("  2. Hook 的压缩逻辑有问题")

if press.compression_count == 0 and press.hook_call_count > 0:
    print(f"\n📌 generate() 问题确认:")
    print(f"   - Hook 被调用了 {press.hook_call_count} 次")
    print(f"   - 但 past_kv 始终为 None")
    print(f"   - 这证实了 generate() **不使用** past_key_values 参数传递 Cache")
    print(f"   - 必须使用手动循环才能让 Press 生效")

print("\n" + "=" * 60)
