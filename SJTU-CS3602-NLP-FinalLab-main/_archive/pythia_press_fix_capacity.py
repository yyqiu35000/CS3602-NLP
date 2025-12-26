import torch


class PythiaStreamingLLMPress:
    def __init__(self, max_capacity: int = 512, n_sink: int = 4):
        """
        Args:
            max_capacity (int): KV Cache 的最大容量 (例如 512)。
                                当 Cache 超过这个长度时，强制触发压缩。
            n_sink (int): 开头保留的 Token 数 (Attention Sinks)。
        """
        self.max_capacity = max_capacity
        self.n_sink = n_sink
        self.hooks = []

    def _kv_crop_hook(self, module, args, output):
        attn_output, past_key_value = output

        if past_key_value is None:
            return output

        key, value = past_key_value
        seq_len = key.shape[2]

        # === 核心修改：只有当长度超过 max_capacity 时才剪枝 ===
        if seq_len <= self.max_capacity:
            return output

        # 我们需要保留的总长度就是 max_capacity
        # 其中 Sink 占了 self.n_sink
        # 剩下的给 Recent Window
        window_size = self.max_capacity - self.n_sink

        # 切片 A: Sink (开头)
        k_sink = key[:, :, : self.n_sink, :]
        v_sink = value[:, :, : self.n_sink, :]

        # 切片 B: Rolling Window (最近的 N 个)
        # 取最后 window_size 个
        k_window = key[:, :, -window_size:, :]
        v_window = value[:, :, -window_size:, :]

        # 拼接
        k_new = torch.cat([k_sink, k_window], dim=2)
        v_new = torch.cat([v_sink, v_window], dim=2)

        return (attn_output, (k_new, v_new))

    def register(self, model):
        self.remove()
        # 适配 Pythia
        layers = (
            model.gpt_neox.layers if hasattr(model, "gpt_neox") else model.model.layers
        )
        for layer in layers:
            # 兼容不同命名
            target = layer.attention if hasattr(layer, "attention") else layer.self_attn
            handle = target.register_forward_hook(self._kv_crop_hook)
            self.hooks.append(handle)

    def remove(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def __call__(self, model):
        self.model_ref = model
        return self

    def __enter__(self):
        self.register(self.model_ref)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
