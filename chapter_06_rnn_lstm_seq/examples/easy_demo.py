import os
import sys

import torch
import torch.nn as nn

sys.path.append(os.getcwd())

from utils import count_parameters



# 简单配置
input_dim = 10   # 假设 One-hot 编码维度
hidden_dim = 20  # 记忆容量
num_layers = 1
batch_size = 3
seq_len = 5      # 序列长度 "Hello"

# 定义 LSTM
# 偏好：batch_first=True
lstm = nn.LSTM(
    input_size = input_dim,
    hidden_size = hidden_dim,
    num_layers = num_layers,
    batch_first = True
)

# 构造虚构输入 (Batch, Seq, Feature)
x = torch.randn(batch_size, seq_len, input_dim)

# 前向传播
# out: 所有时间步的隐状态 (Batch, Seq, Hidden)
# (h_n, c_n): 最后一个时间步的隐状态和细胞状态
out, (h_n, c_n) = lstm(x)

print(f"Input Shape: {x.shape}")        # torch.Size([3, 5, 10])
print(f"Output Shape: {out.shape}")     # torch.Size([3, 5, 20]) -> 包含每一步的特征
print(f"Hidden State Shape: {h_n.shape}") # torch.Size([1, 3, 20]) -> (Layers, Batch, Hidden)

# 验证: Output 的最后一步是否等于 Hidden State 的最后一步?
# 注意: 如果是双向 LSTM，这里会有所不同
assert torch.allclose(out[:, -1, :], h_n[-1, :, :])
print(">> Simple verification passed: Output last step matches Hidden State.")

