import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import snapshot_download

# 指定下载目录
local_dir = "./models/pythia-2.8b"

print("开始下载...")
snapshot_download(
    repo_id="EleutherAI/pythia-2.8b",
    local_dir=local_dir,
    local_dir_use_symlinks=False,  # 确保下载的是真实文件而不是快捷方式
)
print(f"下载完成，文件保存在: {local_dir}")
