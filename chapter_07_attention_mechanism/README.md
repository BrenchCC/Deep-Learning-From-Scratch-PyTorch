# Chapter 07: Attention Mechanism Core - Theoretical Review

## 3分钟先读（零基础导读）
- 本章只聚焦注意力机制核心，不引入完整 Transformer 堆叠。
- 你会从机制层理解 `Q/K/V`、`Scaled Dot-Product` 与 `Masking` 的必要性。
- 代码层面对应三件事：`demo.py`（机制演示）、`attention.py`（核心算子）、`train.py`（masked copy 训练验证）。

## 术语速查（本章高频）
| 术语 | 一句话解释 |
|---|---|
| Query (Q) | 当前 token 发出的检索请求 |
| Key (K) | 每个候选 token 提供的匹配标签 |
| Value (V) | 被检索后真正返回的信息内容 |
| Scaled Dot-Product | `QK^T / sqrt(d_k)` 的相关性打分机制 |
| Attention Score | softmax 后的注意力权重分布 |
| Context Vector | 按注意力权重加权后的聚合表示 |
| Padding Mask | 屏蔽 PAD 位，阻止无效信息进入上下文 |
| Causal Mask | 屏蔽未来位置，保证自回归因果性 |
| Temperature | 控制 softmax 尖锐程度的缩放因子 |
| Entropy of Attention | 注意力分布不确定性的度量 |

## 理论 -> 代码映射表
| 理论主题 | 对应代码位置 | 你会看到什么 |
|---|---|---|
| Scaled Dot-Product Attention | `attention.py` | score 计算、mask 注入、context 聚合 |
| Padding/Causal Mask | `masks.py` | `[B,1,1,S]` 与 `[1,1,S,S]` 布尔 mask |
| 单层自注意力建模 | `model.py` | Embedding -> Attention -> Add&Norm -> Linear |
| Masked Copy 训练目标 | `train.py` | 仅对 mask 位置计算交叉熵损失 |
| 机制可视化输出 | `demo.py` | score/context shape 与样例落盘 |

## 常见误区 + 最小运行命令
### 常见误区
- 误区 1：`Q/K/V` 是三份独立语义。  
  正解：它们通常来自同一输入的不同线性投影，语义由任务驱动形成。
- 误区 2：不做缩放也能稳定训练。  
  正解：不除以 `sqrt(d_k)` 时，softmax 可能饱和，梯度退化。
- 误区 3：mask 可以省略。  
  正解：不屏蔽 PAD 会引入噪声，不做 causal mask 会产生信息泄漏。
- 误区 4：所有位置都计算损失更充分。  
  正解：本章任务目标是重建“被遮挡位置”，只在 mask 位监督更符合任务定义。

### 最小运行命令
```bash
python chapter_07_attention_mechanism/demo.py
python chapter_07_attention_mechanism/train.py --epochs 1 --num_samples 2000
```

## 章节概述（Overview）
在 RNN 时代，模型通过时间步递归携带状态；注意力机制则直接让每个位置与全局位置交互。它不是“记住过去”，而是“随时检索全局”。

本章要解决的核心问题是：
1. 如何定义“相关性”。
2. 如何把相关性转成可训练权重。
3. 如何用这些权重稳定聚合信息并用于预测。

---

## 1. Scaled Dot-Product Attention

### 1.1 学术阐述（Academic Elaboration）
注意力计算本质是一个可微分检索过程：

$$
\mathrm{Attention}(Q, K, V) = \mathrm{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

其中 `M` 为 mask 矩阵。被屏蔽位置在 softmax 前被压到极小值，softmax 后概率近似 0。

### 1.2 通俗解释（Intuitive Explanation）
把注意力想象成“按问题查资料”：
- `Q`：我想找什么？
- `K`：每条资料的目录标签是什么？
- `V`：资料真实内容是什么？

你不会平均阅读所有资料，而是按匹配度分配注意力，再做加权阅读。

### 1.3 数学推导（Mathematical Derivation）
若 `q_i` 与 `k_i` 为零均值、单位方差独立变量，则：

$$
q\cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

其方差随 `d_k` 增大。若不缩放，logit 绝对值容易变大，softmax 更尖锐，梯度接近 0。除以 `sqrt(d_k)` 能稳定 logit 分布，提升训练可控性。

### 1.4 形状级推导（Tensor Shape Derivation）
设：

- `Q ∈ R^{B×S_q×d_k}`
- `K ∈ R^{B×S_k×d_k}`
- `V ∈ R^{B×S_k×d_v}`

则：

1. `QK^T ∈ R^{B×S_q×S_k}`
2. `A = softmax(...) ∈ R^{B×S_q×S_k}`
3. `C = AV ∈ R^{B×S_q×d_v}`

因此 context 的序列长度来自 query 端，信息来源来自 key/value 端。

### 1.5 Softmax 雅可比与梯度稳定性
对单个 query 行的 softmax 输出 `a = softmax(z)`，其雅可比矩阵：

$$
\frac{\partial a_i}{\partial z_j} = a_i(\delta_{ij} - a_j)
$$

结论：
1. 当某个 `a_i` 接近 1，其它接近 0 时，绝大多数偏导接近 0，梯度传播变弱。
2. 缩放 `z / sqrt(d_k)` 可以降低过尖分布出现概率，缓解梯度退化。

### 1.6 注意力熵视角（Entropy View）
对每个 query 的注意力分布 `a`：

$$
H(a) = -\sum_j a_j \log a_j
$$

- 高熵：关注更分散，偏向全局融合。
- 低熵：关注更尖锐，偏向硬检索。

温度 `\tau` 形式：

$$
a = softmax(z/\tau)
$$

- `\tau ↓`：分布更尖，熵下降。
- `\tau ↑`：分布更平，熵上升。

`1/sqrt(d_k)` 可以理解为一种维度相关的温度校准。

### 1.7 简单与复杂示例（Examples）
- **Simple Example**：句子长度为 4，某个 query 对第 2 个 key 匹配最高，softmax 后第 2 位权重接近 1，context 主要来自第 2 位 value。
- **Complex Example**：长序列中一个词需要跨越 20+ token 读取约束信息，注意力可直接建立全局依赖，不需要像 RNN 那样跨多步传递状态。

---

## 2. Masking 机制

### 2.1 学术阐述（Academic Elaboration）
Masking 在 score 空间施加结构约束，而不是在 value 空间硬截断。其优点是：
1. 与 softmax 概率解释一致。
2. 保持端到端可导。
3. 统一处理 padding 与因果约束。

### 2.2 通俗解释（Intuitive Explanation）
Mask 相当于给检索系统设置“不可见条目”：
- Padding Mask：把空白占位符从候选列表里删掉。
- Causal Mask：考试时不能偷看后面的答案。

### 2.3 数学表达（Mathematical Form）
对任意被屏蔽位置 `j`，令：

$$
M_{ij} = -\infty
$$

则：

$$
\mathrm{Softmax}(z_i + M_{ij}) \approx 0
$$

该位置不会对 context 产生贡献。

### 2.4 掩码广播规则（Broadcast Rules）
本仓库采用：
1. Padding mask: `[B, 1, 1, S_k]`
2. Causal mask: `[1, 1, S_q, S_k]`

其并集后可广播到 score 张量，保证 batch 与序列维语义一致。

### 2.5 错误掩码的后果
1. 掩码维度错配：会触发 shape error 或错误广播。
2. 掩码语义反转（True/False 反了）：模型会“只看被屏蔽位”。
3. 不做 padding mask：在变长 batch 中，PAD 会污染统计并影响收敛。

---

## 3. 本章训练任务：Masked Copy

### 3.1 学术阐述（Academic Elaboration）
任务定义：给定部分 token 被 mask 的输入序列 `x`，预测原始 token `y`。监督只作用于被 mask 索引集合 `\Omega`：

$$
\mathcal{L} = -\frac{1}{|\Omega|}\sum_{i\in\Omega}\log p_\theta(y_i|x)
$$

这迫使模型利用全局上下文恢复缺失信息，而不是做逐位复制。

### 3.2 通俗解释（Intuitive Explanation）
像“完形填空”：题目挖掉几个词，你需要利用整句上下文把它补回来。

### 3.3 梯度层面解释
对 logits `l_i` 的交叉熵梯度：

$$
\frac{\partial \mathcal{L}}{\partial l_i} = p_i - \mathbb{1}[i=y]
$$

仅在 `i ∈ Ω` 的位置激活该梯度，等价于把优化资源聚焦在“缺失重建”子问题。

### 3.4 为什么用单层模型
- 目的不是追求 SOTA，而是隔离变量。
- 单层结构更容易把模型行为归因到注意力本身。
- 便于后续在 Chapter 08 中平滑升级到 MHA + Encoder-Decoder。

---

## 4. 复杂度与可扩展性

### 4.1 时间与空间复杂度
对序列长度 `S`、隐藏维度 `d`：

1. score 计算时间复杂度 `O(S^2 d)`。
2. score 存储空间复杂度 `O(S^2)`。
3. 与 RNN 串行依赖相比，注意力并行性更强，但长序列下二次项代价更高。

### 4.2 工程上的启示
- 短到中等序列：注意力通常更高效、建模更直接。
- 超长序列：需要稀疏注意力或线性注意力变体降低 `S^2` 成本。

---

## 5. 代码入口
| 文件 | 功能 | 入口 |
| :--- | :--- | :--- |
| `attention.py` | Scaled Dot-Product Attention | `ScaledDotProductAttention.forward()` |
| `masks.py` | Padding/Causal Mask | `build_padding_mask()`, `build_causal_mask()` |
| `dataset.py` | 掩码复制数据集 | `MaskedCopyDataset`, `MaskedCopyCollator` |
| `model.py` | 单层自注意力模型 | `SingleLayerSelfAttentionModel.forward()` |
| `demo.py` | 随机向量 attention demo | `main()` |
| `train.py` | 掩码复制训练入口 | `main()` |

## 6. 训练产物
- `chapter_07_attention_mechanism/results/attention_demo.json`
- `chapter_07_attention_mechanism/results/metrics.json`
- `chapter_07_attention_mechanism/results/predictions.json`
- `chapter_07_attention_mechanism/results/run_config.json`
- `chapter_07_attention_mechanism/checkpoints/ch07_single_attn_best.pth`

## 7. 与 Chapter 08 的衔接
本章回答“注意力单算子怎么工作”。
下一章回答“如何把注意力升级成完整 Transformer 系统”：
1. Multi-Head 并行子空间建模。
2. 位置编码注入顺序。
3. Encoder-Decoder 端到端训练与生成。
