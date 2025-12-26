import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"PyTorch编译时使用的CUDA版本: {torch.version.cuda}")  # 可能是None如果安装的是CPU版
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")