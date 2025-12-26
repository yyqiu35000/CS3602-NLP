import os

# 1. 设置镜像加速（必须放在 import transformers 之前）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# 模型在本地的路径
model_path = "./models/pythia-70m"

print(f"正在准备下载/加载模型：{model_path} ...")

# 2. 加载 Tokenizer (分词器)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 3. 加载模型
# device_map="auto" 会自动检测你的显卡并使用它
# torch_dtype=torch.float16 使用半精度加载，节省显存且速度更快
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype=torch.float16
)

print("模型加载完成！")
print(f"模型当前运行在: {model.device}")

# 4. 测试运行：生成一段文本
input_text = "Hello, I am a student learning AI. My goal is to"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)

print("正在生成文本...")
outputs = model.generate(input_ids, max_new_tokens=50, do_sample=True, temperature=0.7)

print("-" * 30)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("-" * 30)
