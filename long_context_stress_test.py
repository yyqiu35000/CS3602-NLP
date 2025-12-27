"""
长文本压力测试：验证 H2O 在极限场景下的优势
目标：证明在 Baseline OOM 的情况下，H2O 仍能稳定运行
"""

import os
import torch
import time
import gc
import sys
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from pythia_streaming_h2o_patch import (
    enable_h2o_llm,
    disable_streaming_llm,
    patch_attention_layers,
    reset_attention_timing,
    enable_attention_timing_collection,
    disable_attention_timing_collection,
    get_attention_stats,
)

# HuggingFace 配置（可选：设置镜像加速）
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

model_id = "EleutherAI/pythia-2.8b"
device = "cuda"


def load_long_text_for_stress_test(target_length=10000):
    """加载并拼接文本，达到目标长度"""
    print(f"正在准备长度为 {target_length} tokens 的测试文本...")

    # 从 HuggingFace 加载 wikitext
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])

    # 如果不够长，重复拼接
    while len(text.split()) < target_length * 1.5:  # 预留余量
        text = text + "\n\n" + text

    return text


def stress_test_generation(model, tokenizer, prompt_text, target_tokens, config_name):
    """
    压力测试：生成指定长度的文本，记录显存和速度
    """
    print(f"\n{'='*60}")
    print(f"测试配置: {config_name}")
    print(f"目标生成长度: {target_tokens} tokens")
    print(f"{'='*60}")

    try:
        # Tokenize prompt (限制长度，避免超出模型限制)
        inputs = tokenizer(
            prompt_text, return_tensors="pt", truncation=True, max_length=256
        ).to(device)
        input_len = inputs.input_ids.shape[1]
        print(f"Prompt 长度: {input_len} tokens")

        # 检查是否会超出模型最大长度
        max_model_length = 2048  # Pythia-2.8b 的位置编码限制
        if input_len + target_tokens > max_model_length:
            print(
                f"⚠️  警告: Prompt ({input_len}) + 目标生成 ({target_tokens}) = {input_len + target_tokens} > {max_model_length}"
            )
            print(f"    可能会出现性能下降或位置编码越界")
            if config_name == "baseline":
                print(f"    Baseline 不建议超过模型限制，将限制生成长度")
                target_tokens = min(target_tokens, max_model_length - input_len - 10)
                print(f"    调整为: {target_tokens} tokens")

        # 重置显存统计
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()

        # 预热
        model.generate(
            **inputs,
            max_new_tokens=10,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # 开始正式测试
        print(f"开始生成 {target_tokens} tokens...")
        start_time = time.time()

        reset_attention_timing()
        enable_attention_timing_collection()

        outputs = model.generate(
            **inputs,
            max_new_tokens=target_tokens,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,  # 贪婪解码，确保稳定性
        )

        torch.cuda.synchronize()
        total_time = time.time() - start_time
        disable_attention_timing_collection()

        # 统计
        actual_tokens = outputs.shape[1] - input_len
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        throughput = actual_tokens / total_time
        avg_latency = total_time / actual_tokens * 1000  # ms per token

        avg_attn, std_attn = get_attention_stats()

        print(f"\n✅ 成功完成")
        print(f"  实际生成: {actual_tokens} tokens")
        print(f"  总耗时: {total_time:.2f} s")
        print(f"  峰值显存: {peak_mem:.2f} GB")
        print(f"  吞吐量: {throughput:.2f} tok/s")
        print(f"  平均延迟: {avg_latency:.2f} ms/tok")
        print(f"  平均 Attention 时间: {avg_attn*1000:.2f} ms")

        return {
            "success": True,
            "actual_tokens": actual_tokens,
            "total_time": total_time,
            "peak_mem_gb": peak_mem,
            "throughput": throughput,
            "avg_latency_ms": avg_latency,
            "avg_attn_ms": avg_attn * 1000,
        }

    except torch.cuda.OutOfMemoryError as e:
        print(f"\n❌ 显存不足 (OOM)")
        print(f"  错误信息: {str(e)[:100]}...")
        return {
            "success": False,
            "error": "OOM",
            "peak_mem_gb": torch.cuda.max_memory_allocated() / (1024**3),
        }
    except Exception as e:
        print(f"\n❌ 其他错误: {str(e)[:200]}")
        return {
            "success": False,
            "error": str(e)[:100],
        }


def run_progressive_stress_test():
    """
    渐进式压力测试：从小到大测试不同长度
    """
    print("\n" + "=" * 60)
    print("长文本压力测试")
    print("=" * 60)

    # 加载模型
    print(f"\n正在加载模型: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map=device
    )

    # 准备测试文本
    prompt_text = load_long_text_for_stress_test(target_length=2000)

    # 测试配置
    # 注意：Pythia-2.8b 最大位置编码是 2048，超过会性能极度下降
    test_configs = [
        # (名称, 类型, 生成长度)
        # Baseline: 只测试它能承受的范围
        ("baseline", "baseline", 1000),
        ("baseline", "baseline", 1500),  # 512 prompt + 1500 = 2012 (接近极限)
        # H2O: 测试长文本能力
        ("h2o_8_32_256", "h2o", 1000),
        ("h2o_8_32_256", "h2o", 1500),
        ("h2o_8_32_256", "h2o", 3000),
        ("h2o_8_32_256", "h2o", 5000),
        ("h2o_8_32_256", "h2o", 10000),
        ("h2o_8_64_512", "h2o", 10000),
    ]

    results = []

    for config_name, config_type, gen_length in test_configs:
        # 应用配置
        if config_type == "baseline":
            disable_streaming_llm(model)
            patch_attention_layers(model)
        elif config_type == "h2o":
            enable_h2o_llm(
                model, n_sink=8, recent_window=32, max_capacity=256, debug=False
            )

        # 清理显存
        torch.cuda.empty_cache()
        gc.collect()

        # 运行测试
        result = stress_test_generation(
            model, tokenizer, prompt_text, gen_length, config_name
        )
        result["config"] = config_name
        result["gen_length"] = gen_length
        results.append(result)

        # 如果 baseline OOM 了，后续更长的就不测了
        if config_type == "baseline" and not result["success"]:
            print(f"\n⚠️  Baseline 在 {gen_length} tokens 时 OOM，跳过后续更长测试")
            break

    # 打印汇总表格
    print("\n" + "=" * 80)
    print("压力测试结果汇总")
    print("=" * 80)
    print(
        f"| {'配置':<15} | {'长度':<8} | {'状态':<6} | {'峰值显存(GB)':<12} | {'吞吐量(tok/s)':<14} | {'延迟(ms)':<10} |"
    )
    print(f"| {'-'*15} | {'-'*8} | {'-'*6} | {'-'*12} | {'-'*14} | {'-'*10} |")

    for r in results:
        status = "✅" if r["success"] else "❌OOM"
        peak_mem = f"{r['peak_mem_gb']:.2f}" if "peak_mem_gb" in r else "N/A"
        throughput = f"{r['throughput']:.2f}" if r["success"] else "-"
        latency = f"{r['avg_latency_ms']:.2f}" if r["success"] else "-"

        print(
            f"| {r['config']:<15} | {r['gen_length']:<8} | {status:<6} | {peak_mem:<12} | {throughput:<14} | {latency:<10} |"
        )

    print("=" * 80)

    # 保存结果
    import json

    with open("stress_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: stress_test_results.json")


if __name__ == "__main__":
    run_progressive_stress_test()
