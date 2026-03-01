# Chapter 07: Attention Mechanism & Transformer (Encoder-Decoder)

## 0. 小白先读（3-5 分钟）
这一章你会从零实现 Transformer 的核心三件套：
1. Scaled Dot-Product Attention 与 Multi-Head Attention。
2. 固定正弦余弦位置编码（Sinusoidal Positional Encoding）。
3. 完整 Encoder-Decoder Transformer，并在 Toy Copy 任务上训练验证。

### 0.1 先跑再学（最小命令）
```bash
# 1 epoch 冒烟，验证训练与落盘流程
python chapter_07_attention_mechanism/train.py --epochs 1 --num_samples 2000

# 只做形状与 mask 检查
python chapter_07_attention_mechanism/train.py --shape_check_only
```

完整代码级讲解见：`chapter_07_attention_mechanism/CODE_LOGIC_README.md`

### 0.2 术语速查表
| 术语 | 一句话解释 |
| :--- | :--- |
| Query / Key / Value | 注意力计算中的查询、键、值向量。 |
| Scaled Dot-Product | 用 `QK^T / sqrt(d_k)` 计算相关性分数。 |
| Multi-Head Attention | 将注意力拆成多个子空间并行建模后再拼接。 |
| Causal Mask | 屏蔽未来位置，保证自回归解码时不“偷看答案”。 |
| Padding Mask | 屏蔽补齐 token，避免无效位置参与注意力。 |
| Encoder-Decoder Attention | Decoder 读取 Encoder 表示的跨注意力机制。 |
| Position-wise FFN | 每个位置独立共享参数的两层前馈网络。 |
| Post-LN | 残差相加后再做 LayerNorm 的结构。 |

### 0.3 这章代码入口在哪
| 文件 | 作用 | 入口 |
| :--- | :--- | :--- |
| `attention.py` | Scaled Dot-Product + Multi-Head Attention | `MultiHeadAttention.forward()` |
| `positional_encoding.py` | 固定正弦余弦位置编码 | `SinusoidalPositionalEncoding.forward()` |
| `encoder.py` / `decoder.py` | Encoder/Decoder 单层结构 | `TransformerEncoderLayer.forward()` |
| `transformer.py` | 完整 Seq2Seq Transformer 主体 | `Seq2SeqTransformer.forward()` |
| `masks.py` | Padding/Causal mask 工具函数 | `build_padding_mask()`, `build_causal_mask()` |
| `model.py` | 兼容导出层（旧导入路径仍可用） | 包装导出 |
| `dataset.py` | Toy Copy 数据集与批处理拼装 | `ToyCopyDataset`, `ToyCopyCollator` |
| `train.py` | 训练、验证、形状检查、预测样例导出 | `main()` |
| `__init__.py` | 公共 API 导出 | 包级导入 |

### 0.4 通俗桥接
可以把注意力理解为“动态检索”：每个位置会主动去查询整段序列里哪些 token 最相关；多头机制相当于并行使用多个“检索视角”，最后再融合。

### 0.5 常见误区与排错
1. 误区：只加位置编码，不做 mask 也能正确训练。排错：Decoder 必须同时使用 causal mask 和 padding mask。
2. 误区：`d_model` 与 `num_heads` 可任意组合。排错：`d_model` 必须能被 `num_heads` 整除。
3. 误区：只看 loss 即可。排错：本章同时记录 token-level accuracy（忽略 padding）。

---

## 1. 核心原理

### 1.1 Scaled Dot-Product Attention
给定查询、键、值矩阵：

$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

其中：
- $d_k$ 为每个头的键向量维度。
- $M$ 为 mask（被屏蔽位置会被设为极小值）。

### 1.2 Multi-Head Attention
多头注意力会把 `d_model` 拆成 `num_heads` 个子空间：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, ..., head_h)W^O
$$

每个头独立计算注意力，最后拼接并投影回 `d_model`。

### 1.3 Sinusoidal Position Encoding
固定位置编码为：

$$
PE(pos, 2i) = \sin\left(pos / 10000^{2i/d_{model}}\right)
$$

$$
PE(pos, 2i + 1) = \cos\left(pos / 10000^{2i/d_{model}}\right)
$$

它不引入可学习参数，能让模型感知相对与绝对位置信息。

### 1.4 Encoder-Decoder Transformer 结构
- Encoder Layer：`Self-Attention + FFN + Residual + LayerNorm`。
- Decoder Layer：`Masked Self-Attention + Cross-Attention + FFN + Residual + LayerNorm`。
- 本章使用 Post-LN（残差后归一化）。

---

## 2. 训练任务：Toy Seq2Seq Copy

### 2.1 数据规则
- 特殊 token：`PAD = 0`, `BOS = 1`, `EOS = 2`
- 普通 token 范围：`[3, vocab_size - 1]`
- 样本构造：
  - `src = tokens + [EOS]`
  - `tgt_in = [BOS] + tokens`
  - `tgt_out = tokens + [EOS]`

### 2.2 训练指标
- 损失：`CrossEntropyLoss(ignore_index = PAD)`
- 指标：`token_acc`（仅统计非 PAD 位置）

### 2.3 训练产物
训练结束后会在本章目录生成：
- `checkpoints/transformer_copy_best.pth`
- `results/metrics.json`
- `results/run_config.json`
- `results/predictions.json`

---

## 3. 推荐实验
1. 基础冒烟：
```bash
python chapter_07_attention_mechanism/train.py --epochs 1 --num_samples 2000
```

2. 增大模型容量观察收敛：
```bash
python chapter_07_attention_mechanism/train.py --d_model 256 --num_heads 8 --epochs 3
```

3. 只验证结构正确性（不训练）：
```bash
python chapter_07_attention_mechanism/train.py --shape_check_only
```

---

## 4. 与前几章的衔接
- 与 Chapter 06 的 RNN/LSTM 相比，Transformer 不依赖时间步递归，训练并行性更好。
- Chapter 03 的 LayerNorm、AdamW 与本章训练配置直接呼应。
- Chapter 05 的残差连接思想在 Transformer Block 中同样是核心稳定器。
