# PyTorch 维度变换操作函数。

---

### 1. 重塑类 (Reshaping)

**核心函数：** `view`, `reshape`, `flatten`

#### 1.1 `torch.view` / `torch.reshape`

* **学术定义：** 改变张量的形状（Shape），但不改变其元素总数。`view` 只能处理内存连续（Contiguous）的张量，`reshape` 则会自动处理内存复制。
* **通俗解释：** 像玩橡皮泥，总量不变，把它从“长条”搓成“方块”。
* **简单例子 (理解语法)：**
将一个 12 个元素的向量变成  的矩阵。
```python
x = torch.arange(12) # shape: (12)
y = x.view(3, 4)     # shape: (3, 4)

```


* **进阶例子 (LLM Multi-head Attention)：**
**场景：** 在 Transformer 中，将隐藏层维度拆分为多头（Heads）。
假设 `batch_size=32`, `seq_len=128`, `hidden_dim=768` (其中 12 个头，每个头维度 64)。
```python
# Input: [Batch, Seq_Len, Hidden_Dim]
x = torch.randn(32, 128, 768)

# Output: [Batch, Seq_Len, Num_Heads, Head_Dim]
# 使用 view 将最后一维拆开
x_heads = x.view(32, 128, 12, 64)

```



#### 1.2 `torch.flatten`

* **学术定义：** 将指定范围内的维度“压平”为一维。
* **通俗解释：** 把千层饼拍扁成一张薄饼。
* **简单例子 (理解语法)：**
```python
x = torch.randn(2, 3, 4)
# 把后面两维拍扁: (2, 3*4) -> (2, 12)
y = x.flatten(start_dim = 1)

```


* **进阶例子 (CNN/RL 策略网络)：**
**场景：** 在强化学习中，处理图像输入的 Feature Map 进入全连接层之前。
```python
# Input: [Batch, Channels, Height, Width] -> [32, 64, 7, 7]
features = torch.randn(32, 64, 7, 7)

# Flatten relevant dims for Linear Layer input
# Output: [Batch, Features] -> [32, 64*7*7] -> [32, 3136]
flat_features = features.flatten(start_dim = 1)

```



---

### 2. 交换类 (Swapping)

**核心函数：** `permute`, `transpose`

#### 2.1 `torch.permute`

* **学术定义：** 按照指定索引重新排列所有维度的顺序。
* **通俗解释：** 乾坤大挪移，想把第几维放哪就放哪。
* **简单例子 (理解语法)：**
```python
x = torch.randn(2, 3, 4)
# 变成 (4, 2, 3) -> 原来的第2维放最前，第0维放中间，第1维放最后
y = x.permute(2, 0, 1)

```


* **进阶例子 (Transformer 维度调整)：**
**场景：** Multi-head Attention 计算中，为了让 `Seq_Len` 参与矩阵乘法，通常需要把 `Num_Heads` 移到 `Seq_Len` 前面。
```python
# 接上面的 view 例子: [Batch, Seq_Len, Num_Heads, Head_Dim]
x_heads = torch.randn(32, 128, 12, 64)

# 目标: [Batch, Num_Heads, Seq_Len, Head_Dim]
# 交换维度1和维度2
query = x_heads.permute(0, 2, 1, 3)

```



#### 2.2 `torch.transpose`

* **学术定义：** 仅交换**两个**指定的维度。
* **通俗解释：** 只能两两互换，不如 `permute` 灵活，但特定场景更简洁。
* **进阶例子 (矩阵乘法准备)：**
**场景：** 计算 Attention Score  时，需要转置 Key 的最后两维。
```python
# Key: [Batch, Heads, Seq_Len, Head_Dim]
K = torch.randn(32, 12, 128, 64)

# Transpose last two dims for matmul: [Batch, Heads, Head_Dim, Seq_Len]
K_t = K.transpose(-1, -2)

```



---

### 3. 增减类 (Adding/Removing)

**核心函数：** `unsqueeze`, `squeeze`

* **简单例子：** `unsqueeze` 加个壳，`squeeze` 把空壳扔掉。
* **进阶例子 (广播机制 Broadcasting)：**
**场景：** 在大模型中应用 Attention Mask。Mask 形状通常是 2D 的，但需要加到 4D 的 Attention Score 上。
```python
# Scores: [Batch, Heads, Seq_Len, Seq_Len]
scores = torch.randn(32, 12, 128, 128)

# Mask: [Batch, Seq_Len] (比如 padding mask)
mask = torch.ones(32, 128)

# 为了让 Mask 能加到 Scores 上，需要扩展维度: [Batch, 1, 1, Seq_Len]
# 在第1维和第2维插入维度
mask_expanded = mask.unsqueeze(1).unsqueeze(2)

# 此时 mask_expanded 可以通过广播机制与 scores 相加

```



---

### 4. 拼接类 (Combining)

**核心函数：** `cat` (concatenate), `stack`

#### 4.1 `torch.cat`

* **学术定义：** 在现有维度上连接张量。
* **通俗解释：** 接龙，绳子变长了。
* **进阶例子 (KV Cache 推理加速)：**
**场景：** LLM 生成过程中，将新生成的 Key/Value 拼接到缓存中。
```python
# Cache: [Batch, Heads, Past_Len, Head_Dim]
past_key = torch.randn(1, 12, 50, 64)

# New Token Key: [Batch, Heads, 1, Head_Dim]
new_key = torch.randn(1, 12, 1, 64)

# 在 Sequence 维度 (dim=2) 拼接 -> [1, 12, 51, 64]
current_key = torch.cat([past_key, new_key], dim = 2)

```

#### 4.2 `torch.stack`

* **学术定义：** 沿着新维度连接张量。
* **通俗解释：** 叠罗汉，高度增加了。
* **进阶例子 (RL Experience Replay)：**
**场景：** 强化学习中，从 Replay Buffer 采样多个 Step 的状态，打包成 Batch。


### 总结对比表

| 函数 | 操作本质 | 典型学术场景 (LLM/RL) |
| --- | --- | --- |
| `view` / `reshape` | **重组** (维度乘积不变) | 将 `Hidden_Dim` 拆解为 `Num_Heads * Head_Dim` |
| `flatten` | **展平** (多维变一维) | CNN 提取特征后进入 MLP 决策层 |
| `permute` | **全排列** (任意换位) | 将 `(B, L, H, D)` 转换为 `(B, H, L, D)` 以并行计算 Attention |
| `transpose` | **两两换** (仅换一对) | 矩阵转置，计算  |
| `unsqueeze` | **升维** (插个 1) | 制作 Mask，使其能广播到多头 Attention 矩阵上 |
| `cat` | **拼接** (接龙) | 推理时的 KV Cache 更新；多模态特征融合 |
| `stack` | **堆叠** (增高) | 将多个时间步的 State 打包成 Batch |

