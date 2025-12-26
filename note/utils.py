import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class StreamingLLM:
    def __init__(self, n_sink: int = 4, window_size: int = 256):
        self.n_sink = n_sink
        self.window_size = window_size

    def build_context(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.size(1) <= self.n_sink + self.window_size:
            return input_ids
        sink = input_ids[:, : self.n_sink]
        tail = input_ids[:, -self.window_size :]
        return torch.cat([sink, tail], dim=1)

    def compress_cache(self, cache) -> None:
        for layer in cache.layers:
            keys = layer.keys
            values = layer.values
            bsz, num_heads, seq_len, head_dim = keys.shape
            if seq_len <= self.n_sink + self.window_size:
                continue
            device = keys.device
            sink_end = min(self.n_sink, seq_len)
            tail_len = min(self.window_size, seq_len - sink_end)
            if tail_len <= 0:
                keep_idx = torch.arange(0, sink_end, device=device)
            else:
                tail_start = seq_len - tail_len
                keep_prefix = torch.arange(0, sink_end, device=device)
                keep_tail = torch.arange(tail_start, seq_len, device=device)
                keep_idx = torch.cat([keep_prefix, keep_tail], dim=0)
            keys = keys.index_select(2, keep_idx)
            values = values.index_select(2, keep_idx)
            layer.keys = keys
            layer.values = values

    @torch.no_grad()
    def generate(
        self,
        model: AutoModelForCausalLM,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
    ) -> torch.Tensor:
        device = next(model.parameters()).device
        input_ids = input_ids.to(device)
        outputs = model(input_ids, use_cache=True)
        cache = outputs.past_key_values
        self.compress_cache(cache)
        generated = input_ids
        last_token = generated[:, -1:]
        for _ in range(max_new_tokens):
            outputs = model(last_token, use_cache=True, past_key_values=cache)
            logits = outputs.logits[:, -1, :]
            cache = outputs.past_key_values
            self.compress_cache(cache)
            if temperature is not None and temperature > 0:
                logits = logits / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, top_k)
                min_values = v[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_values, torch.full_like(logits, -float("inf")), logits)
            if top_p is not None and 0 < top_p < 1:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, -float("inf"))
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            last_token = next_token
        return generated

def load_model_and_tokenizer(model_id: str, torch_dtype=torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return model, tokenizer

def streaming_generate_from_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    n_sink: int = 4,
    window_size: int = 256,
    max_new_tokens: int = 50,
) -> str:
    wrapper = StreamingLLM(n_sink=n_sink, window_size=window_size)
    encoded = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded.input_ids
    generated_ids = wrapper.generate(model, input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

