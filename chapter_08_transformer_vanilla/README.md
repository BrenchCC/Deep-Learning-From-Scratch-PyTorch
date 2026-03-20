# Chapter 08: Vanilla Transformer - Theoretical Review

## 3分钟先读（零基础导读）
- 本章把 Chapter 07 的注意力核心扩展为完整 Encoder-Decoder。
- 你会系统掌握四个构件：`Multi-Head Attention`、`FeedForward`、`Add & Norm`、`Positional Encoding`。
- 代码层面通过同一入口 `train.py --task sort|translate` 跑通两个任务。

## 术语速查（本章高频）
| 术语 | 一句话解释 |
|---|---|
| Multi-Head Attention | 并行多个注意力子空间后融合，提高表示能力 |
| FeedForward Network | 对每个位置独立作用的非线性映射层 |
| Add & Norm | 残差连接 + LayerNorm，稳定深层训练 |
| Positional Encoding | 在无递归结构中显式注入位置信息 |
| Encoder-Decoder Attention | Decoder 读取 Encoder memory 的跨注意力 |
| Teacher Forcing | 训练时喂入真实前缀 token 的策略 |
| Autoregressive Decoding | 推理时逐步生成下一个 token 的过程 |
| Exposure Bias | 训练输入分布与推理输入分布不一致 |
| Perplexity | 语言建模常用不确定性度量 |

## 理论 -> 代码映射表
| 理论主题 | 对应代码位置 | 你会看到什么 |
|---|---|---|
| Scaled Dot-Product + Multi-Head | `attention.py` | 拆头、并行注意力、合并投影 |
| Positional Encoding | `positional_encoding.py` | 正弦余弦位置编码公式实现 |
| Encoder Block | `encoder.py` | Self-Attn -> Add&Norm -> FFN -> Add&Norm |
| Decoder Block | `decoder.py` | Masked Self-Attn -> Cross-Attn -> FFN |
| Vanilla Transformer 主体 | `transformer.py` | encode/decode/forward/greedy_decode |
| 双任务数据建模 | `dataset.py` | `SortDataset` 与 `ToyTranslationDataset` |
| 训练与评估 | `train.py` | Teacher Forcing 训练 + greedy 推理样例 |

## 常见误区 + 最小运行命令
### 常见误区
- 误区 1：Transformer 不需要位置编码。  
  正解：自注意力本身对顺序置换不敏感，必须补充位置先验。
- 误区 2：训练和推理是同一个输入分布。  
  正解：训练常用 Teacher Forcing，推理用自回归，存在 exposure bias。
- 误区 3：多头只是“多份重复计算”。  
  正解：每个头对应不同线性子空间，能捕捉不同关系模式。
- 误区 4：只看 loss 就够了。  
  正解：需要同时看 token-level accuracy 和样例预测可读性。

### 最小运行命令
```bash
python chapter_08_transformer_vanilla/train.py --task sort --epochs 1 --num_samples 2000
python chapter_08_transformer_vanilla/train.py --task translate --epochs 1
```

## 章节概述（Overview）
RNN 通过时间递归积累状态，Transformer 通过全局注意力直接建模任意两点依赖，训练并行性显著提升。

本章核心问题：
1. 如何把单次注意力扩展为可堆叠网络层。
2. 如何在不递归的前提下注入位置信息。
3. 如何把编码与解码耦合成可训练的 Seq2Seq 系统。

---

## 1. Transformer 架构总览

### 1.1 学术阐述（Academic Elaboration）
本章实现的是简化版 Encoder-Decoder Transformer：

1. Encoder Layer：
- Multi-Head Self-Attention
- Add & Norm
- Position-wise FFN
- Add & Norm

2. Decoder Layer：
- Masked Multi-Head Self-Attention
- Add & Norm
- Encoder-Decoder Cross-Attention
- Add & Norm
- Position-wise FFN
- Add & Norm

实现采用 Post-LN 形式（残差后归一化）。

### 1.2 通俗解释（Intuitive Explanation）
- Encoder 像“全文精读并做笔记”。
- Decoder 像“边写答案边查笔记”。
- Masked Self-Attention 防止“提前看到未来答案”。

### 1.3 层级函数表达
记 Encoder 第 `l` 层函数为 `E_l`，Decoder 第 `l` 层函数为 `D_l`：

$$
H^{(0)} = \mathrm{Embed}_{src}(X) + PE, \quad H^{(l)} = E_l(H^{(l-1)})
$$

$$
G^{(0)} = \mathrm{Embed}_{tgt}(Y_{<t}) + PE, \quad G^{(l)} = D_l(G^{(l-1)}, H^{(L_e)})
$$

最终 logits：

$$
\mathrm{logits}_t = W_o G_t^{(L_d)} + b_o
$$

---

## 2. Multi-Head Attention 深入分析

### 2.1 学术阐述（Academic Elaboration）
设 `d_model` 被拆成 `h` 个头，每头维度 `d_k = d_model / h`。第 `i` 个头：

$$
\mathrm{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

最终输出：

$$
\mathrm{MHA}(Q, K, V) = \mathrm{Concat}(\mathrm{head}_1, \ldots, \mathrm{head}_h)W^O
$$

### 2.2 通俗解释（Intuitive Explanation）
多个头像多个“关系观察员”：
- 有的头关注局部搭配。
- 有的头关注长程依赖。
- 有的头关注语序或对齐线索。

单头只能给一种观察视角，多头能并行捕捉多种关系。

### 2.3 可表达性直觉
单头输出本质是一个 attention kernel 下的线性组合。多头通过不同投影矩阵构造多个 kernel，再经 `W^O` 融合，相当于在多个关系子空间做混合专家式聚合，表达能力更高。

### 2.4 复杂度讨论（Complexity）
对序列长度 `S`：
- 单层注意力时间复杂度主项 `O(S^2 · d_model)`。
- score 存储复杂度主项 `O(S^2)`。
- 与 RNN `O(S · d_model^2)` 对比：
  - Transformer 单层并行度高。
  - 长序列时 `S^2` 成本会成为瓶颈。

---

## 3. Positional Encoding 的必要性

### 3.1 学术阐述（Academic Elaboration）
自注意力对输入排列是置换等变的，不天然知道“第几个 token”。正弦余弦位置编码定义：

$$
PE(pos, 2i) = \sin\left(pos / 10000^{2i / d_{\mathrm{model}}}\right)
$$

$$
PE(pos, 2i + 1) = \cos\left(pos / 10000^{2i / d_{\mathrm{model}}}\right)
$$

### 3.2 置换等变简证
若输入序列应用置换矩阵 `P`，纯 self-attention（无位置编码）满足：

$$
\mathrm{Attn}(PX) = P\,\mathrm{Attn}(X)
$$

说明模型只知道“关系结构”，不知道“绝对位置”。位置编码打破这一对称性。

### 3.3 相对位移性质直觉
对同一频率分量 `\omega`：

$$
\sin((p+k)\omega),\cos((p+k)\omega)
$$

可由 `\sin(p\omega),\cos(p\omega)` 与 `k` 的三角恒等式组合得到，因此模型更容易编码相对位移信息。

### 3.4 通俗解释（Intuitive Explanation）
如果没有位置编码，模型会把“猫咬狗”和“狗咬猫”看成近似同一袋词。位置编码就是给每个词打上“坐标系”。

---

## 4. Add & Norm 与训练稳定性

### 4.1 学术阐述（Academic Elaboration）
残差连接为深层网络提供近似恒等路径：

$$
x_{l+1} = \mathrm{LN}(x_l + F_l(x_l))
$$

在反向传播中，梯度可通过恒等分支直接传递，缓解深层网络退化。

### 4.2 为什么 LayerNorm 适配序列建模
LayerNorm 对单样本特征维归一化，不依赖 batch 统计。对于可变长度序列和小 batch 训练更稳定，推理与训练统计一致性更好。

---

## 5. 训练目标与推理机制

### 5.1 学术阐述（Academic Elaboration）
训练阶段采用 Teacher Forcing：输入 `tgt_input`（真实前缀）预测 `tgt_output`（右移目标），损失为忽略 PAD 的交叉熵。

$$
\mathcal{L} = -\sum_t \log p_\theta(y_t \mid y_{<t}, x)
$$

该式是条件序列概率分解：

$$
p(y\mid x)=\prod_t p(y_t\mid y_{<t},x)
$$

负对数似然（NLL）即对应上式求和。

### 5.2 困惑度（Perplexity）
对 token 平均 NLL 为 `\ell`，困惑度：

$$
\mathrm{PPL}=\exp(\ell)
$$

PPL 越低表示模型对目标序列不确定性越低。

### 5.3 推理与 exposure bias
推理阶段采用 greedy decoding：每步把上一步预测拼回输入，直到 `EOS` 或达最大长度。训练用真前缀，推理用模型前缀，这一分布差异即 exposure bias 来源。

### 5.4 通俗解释（Intuitive Explanation）
- 训练：老师一步一步喂标准答案前缀。
- 推理：模型自己接话，前一步错会影响后一步。

---

## 6. 本章双任务设计

### 6.1 任务一：数字排序（sort）
- 输入：随机整数 token 序列。
- 输出：升序序列。
- 价值：检验模型是否学到“全局重排规则”。

### 6.2 任务二：小翻译数据（translate）
- 输入：`data/toy_translation_pairs.tsv` 本地翻译对。
- 输出：目标词序列。
- 价值：检验标准 Seq2Seq 编解码闭环。

### 6.3 为什么两个任务都保留
- sort：结构规则清晰，易定位模型逻辑问题。
- translate：语义映射更接近真实 NLP 场景。
- 二者组合能同时覆盖“规则学习”和“映射学习”。

---

## 7. 代码入口
| 文件 | 功能 | 入口 |
| :--- | :--- | :--- |
| `attention.py` | Scaled Dot-Product + Multi-Head | `MultiHeadAttention.forward()` |
| `positional_encoding.py` | 正弦余弦位置编码 | `SinusoidalPositionalEncoding.forward()` |
| `feed_forward.py` | Position-wise FFN | `PositionwiseFeedForward.forward()` |
| `encoder.py` | Encoder Layer | `TransformerEncoderLayer.forward()` |
| `decoder.py` | Decoder Layer | `TransformerDecoderLayer.forward()` |
| `transformer.py` | 完整 Vanilla Transformer | `VanillaTransformer.forward()` |
| `dataset.py` | Sort/Translate 数据集与 collator | `SortDataset`, `ToyTranslationDataset` |
| `train.py` | 双任务统一训练入口 | `train_main()` |

## 8. 训练产物
排序任务：
- `results/sort_metrics.json`
- `results/sort_predictions.json`
- `checkpoints/transformer_sort_best.pth`

翻译任务：
- `results/translate_metrics.json`
- `results/translate_predictions.json`
- `checkpoints/transformer_translate_best.pth`

通用配置：
- `results/run_config.json`

## 9. 向后衔接
1. 承接 Chapter 07 的注意力机制本体。
2. 为后续扩展 Pre-LN、学习率调度、label smoothing、beam search 打基础。
3. 可进一步扩展到更大词表和真实平行语料。
