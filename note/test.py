from utils import load_model_and_tokenizer, StreamingLLM
import time
import torch


def benchmark_speed_streaming_kv(
    model,
    tokenizer,
    prompt: str,
    n_sink: int = 4,
    window_size: int = 256,
    num_tokens: int = 50,
    batch_size: int = 1,
):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    input_ids = input_ids.repeat(batch_size, 1)

    stream = StreamingLLM(n_sink=n_sink, window_size=window_size)

    start = time.time()
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
    cache = outputs.past_key_values
    stream.compress_cache(cache)
    generated = input_ids
    last_token = generated[:, -1:]

    ttft = None

    for step in range(num_tokens):
        step_start = time.time() if step == 0 else None
        with torch.no_grad():
            outputs = model(last_token, use_cache=True, past_key_values=cache)
        cache = outputs.past_key_values
        stream.compress_cache(cache)
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        last_token = next_token
        if step == 0:
            ttft = time.time() - start

    end = time.time()
    total_time = end - start
    if ttft is None:
        ttft = total_time

    token_phase_time = max(total_time - ttft, 1e-6)
    total_tokens = num_tokens * batch_size
    tpot = token_phase_time / total_tokens
    throughput = total_tokens / token_phase_time

    num_params = sum(p.numel() for p in model.parameters())
    prompt_len = input_ids.shape[1]
    total_tokens_processed = (prompt_len + num_tokens) * batch_size
    total_flops = 2.0 * num_params * total_tokens_processed
    avg_flops_per_token = total_flops / total_tokens_processed

    return {
        "ttft": ttft,
        "tpot": tpot,
        "throughput": throughput,
        "total_time": total_time,
        "total_flops": total_flops,
        "avg_flops_per_token": avg_flops_per_token,
    }


def benchmark_speed_baseline(
    model,
    tokenizer,
    prompt: str,
    num_tokens: int = 50,
    batch_size: int = 1,
):
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    input_ids = input_ids.repeat(batch_size, 1)

    start = time.time()
    ttft = None

    generated = input_ids
    cache = None

    for step in range(num_tokens):
        with torch.no_grad():
            if cache is None:
                outputs = model(generated, use_cache=True)
            else:
                outputs = model(generated[:, -1:], use_cache=True, past_key_values=cache)
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=1)
        if step == 0:
            ttft = time.time() - start

    end = time.time()
    total_time = end - start
    if ttft is None:
        ttft = total_time

    token_phase_time = max(total_time - ttft, 1e-6)
    total_tokens = num_tokens * batch_size
    tpot = token_phase_time / total_tokens
    throughput = total_tokens / token_phase_time

    num_params = sum(p.numel() for p in model.parameters())
    prompt_len = input_ids.shape[1]
    total_tokens_processed = (prompt_len + num_tokens) * batch_size
    total_flops = 2.0 * num_params * total_tokens_processed
    avg_flops_per_token = total_flops / total_tokens_processed

    return {
        "ttft": ttft,
        "tpot": tpot,
        "throughput": throughput,
        "total_time": total_time,
        "total_flops": total_flops,
        "avg_flops_per_token": avg_flops_per_token,
    }


def evaluate_ppl_streaming_kv(
    model,
    tokenizer,
    text: str,
    n_sink: int = 4,
    window_size: int = 256,
    max_tokens: int = 2000,
):
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[:, :max_tokens].to(next(model.parameters()).device)
    seq_len = input_ids.size(1)
    if seq_len < 2:
        return float("inf")

    stream = StreamingLLM(n_sink=n_sink, window_size=window_size)
    cache = None
    losses = []
    ce = torch.nn.CrossEntropyLoss(reduction="none")

    for pos in range(seq_len - 1):
        cur = input_ids[:, pos:pos + 1]
        with torch.no_grad():
            if cache is None:
                outputs = model(cur, use_cache=True)
            else:
                outputs = model(cur, use_cache=True, past_key_values=cache)
            cache = outputs.past_key_values
            stream.compress_cache(cache)
            logits = outputs.logits[:, -1, :]
            target = input_ids[:, pos + 1]
            loss = ce(logits, target).mean()
            losses.append(loss)

    ppl = torch.exp(torch.stack(losses).mean())
    return ppl.item()


def evaluate_ppl_baseline(
    model,
    tokenizer,
    text: str,
    max_tokens: int = 2000,
):
    enc = tokenizer(text, return_tensors="pt")
    input_ids = enc.input_ids[:, :max_tokens].to(next(model.parameters()).device)
    seq_len = input_ids.size(1)
    if seq_len < 2:
        return float("inf")

    max_length = model.config.max_position_embeddings
    stride = 512
    nlls = []
    prev_end = 0
    ce = torch.nn.CrossEntropyLoss(reduction="none")

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        if trg_len <= 0:
            break
        cur = input_ids[:, begin:end]
        target = cur.clone()
        target[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(cur, labels=target)
            loss = outputs.loss

        nlls.append(loss)
        prev_end = end
        if end == seq_len:
            break

    if not nlls:
        return float("inf")
    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()


model_id = "EleutherAI/pythia-2.8b"
model, tokenizer = load_model_and_tokenizer(model_id)

from datasets import load_dataset

def load_long_text_from_dataset(
    dataset_name: str = "wikitext",
    split: str = "test",
    limit_samples: int = 1,
    max_chars: int | None = None,
) -> str:
    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        
        # 正确获取文本的方法：
        # 方法1：使用列表推导
        texts = [item["text"] for item in ds.select(range(limit_samples))]
        # 或者方法2：直接切片
        # texts = [ds[i]["text"] for i in range(min(limit_samples, len(ds)))]
        
        text = "\n\n".join(texts)
    
    elif dataset_name == "pg19":
        ds = load_dataset("pg19", split=split, streaming=True)
        sample = next(iter(ds))
        text = sample["text"]
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    if max_chars is not None and len(text) > max_chars:
        text = text[:max_chars]
    
    return text
# 固定一次抽样
text_wiki = load_long_text_from_dataset(
    dataset_name="wikitext",
    split="train",
    limit_samples=100,
    max_chars=100000,
)

text_pg19 = load_long_text_from_dataset(
    dataset_name="pg19",
    split="test",
    limit_samples=1,
    max_chars=100000,
)

print("Wiki text length:", len(text_wiki))
print("PG-19 text length:", len(text_pg19))


# PPL测试
text_wiki_ppl = text_wiki[:1000]  # 2048 tokens
ppl_base_wiki = evaluate_ppl_baseline(model, tokenizer, text_wiki_ppl, max_tokens=2000)
print("Baseline PPL on WikiText (2000 tokens):", ppl_base_wiki)

ppl_stream_wiki_256 = evaluate_ppl_streaming_kv(model, tokenizer, text_wiki_ppl, n_sink=8, window_size=256, max_tokens=2000)
print("StreamingLLM(256) PPL on WikiText (2000 tokens):", ppl_stream_wiki_256)

ppl_stream_wiki_512 = evaluate_ppl_streaming_kv(model, tokenizer, text_wiki_ppl, n_sink=8, window_size=512, max_tokens=2000)
print("StreamingLLM(512) PPL on WikiText (2000 tokens):", ppl_stream_wiki_512)

# 加速测试
prompt_wiki = text_wiki[:2000]
speed_base_wiki = benchmark_speed_baseline(model, tokenizer, prompt_wiki, num_tokens=1500, batch_size=1)
print("Baseline speed on WikiText (1500 tokens):", speed_base_wiki)

speed_stream_wiki_256 = benchmark_speed_streaming_kv(model, tokenizer, prompt_wiki, n_sink=8, window_size=256, num_tokens=1500, batch_size=1)
print("StreamingLLM(256) speed on WikiText (1500 tokens):", speed_stream_wiki_256)

speed_stream_wiki_512 = benchmark_speed_streaming_kv(model, tokenizer, prompt_wiki, n_sink=8, window_size=512, num_tokens=1500, batch_size=1)
print("StreamingLLM(512) speed on WikiText (1500 tokens):", speed_stream_wiki_512)

# PPL测试  
text_pg19_ppl = text_pg19[:10000]  # 2048 tokens
ppl_base_pg19 = evaluate_ppl_baseline(model, tokenizer, text_pg19_ppl, max_tokens=2000)
print("Baseline PPL on PG-19 (2000 tokens):", ppl_base_pg19)

ppl_stream_pg19_256 = evaluate_ppl_streaming_kv(model, tokenizer, text_pg19_ppl, n_sink=8, window_size=256, max_tokens=2000)
print("StreamingLLM(256) PPL on PG-19 (2000 tokens):", ppl_stream_pg19_256)

ppl_stream_pg19_512 = evaluate_ppl_streaming_kv(model, tokenizer, text_pg19_ppl, n_sink=8, window_size=512, max_tokens=2000)
print("StreamingLLM(512) PPL on PG-19 (2000 tokens):", ppl_stream_pg19_512)

# 加速测试
prompt_pg19 = text_pg19[:2000]
speed_base_pg19 = benchmark_speed_baseline(model, tokenizer, prompt_pg19, num_tokens=1500, batch_size=1)
print("Baseline speed on PG-19 (1500 tokens):", speed_base_pg19)

speed_stream_pg19_256 = benchmark_speed_streaming_kv(model, tokenizer, prompt_pg19, n_sink=8, window_size=256, num_tokens=1500, batch_size=1)
print("StreamingLLM(256) speed on PG-19 (1500 tokens):", speed_stream_pg19_256)

speed_stream_pg19_512 = benchmark_speed_streaming_kv(model, tokenizer, prompt_pg19, n_sink=8, window_size=512, num_tokens=1500, batch_size=1)
print("StreamingLLM(512) speed on PG-19 (1500 tokens):", speed_stream_pg19_512)