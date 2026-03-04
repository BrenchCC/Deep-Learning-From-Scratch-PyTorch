# Chapter 09: Efficient Attention - MQA, GQA, MLA（大模型注意力变体）

## 3分钟先读（零基础导读）
- 本章核心问题：在保持注意力表达能力的同时，如何降低 KV Cache 占用并提升长序列推理效率。
- 你会看到四个可切换变体：`MHA`、`MQA`、`GQA`、`MLA`。
- 本章代码有三条主线：
  1. `demo.py`：同一 hidden states 输入下对比四变体输出统计与 KV Cache。
  2. `benchmark.py`：扫描不同 `seq_len` 的延迟和显存代理指标。
  3. `train.py`：在统一 toy next-token 任务上对比四变体训练表现。

## 术语速查（本章高频）
| 术语 | 一句话解释 |
|---|---|
| KV Cache | 自回归推理中缓存历史 $K/V$ 的机制，避免重复计算 |
| MHA | 多头注意力，每个头都有独立 K/V 投影 |
| MQA | 多 Query 共享一组 K/V，显著减少 KV Cache |
| GQA | Query 头按组共享 K/V，在 MHA 与 MQA 间折中 |
| MLA | 先把 K/V 压到低秩 latent，再重构参与注意力 |
| Low-Rank Compression | 用低维 latent 近似表示高维特征，降低存储成本 |
| Tokens/s | 每秒处理 token 数，粗略衡量吞吐 |

## 理论 -> 代码映射表
| 理论主题 | 对应代码位置 | 你会看到什么 |
|---|---|---|
| MHA 标准形式 | `mha.py` | $Q/K/V$ 全头独立 |
| MQA 共享 K/V | `mqa.py` | Query 多头，K/V 单头共享 |
| GQA 分组共享 | `gqa.py` | $\frac{\text{num\_heads}}{\text{num\_kv\_heads}}$ 组映射 |
| MLA 低秩压缩 | `mla.py` | `compress -> expand -> attention` |
| 统一配置与复杂度估算 | `common.py` | `AttentionConfig` 与 KV Cache 估算函数 |
| 统一模型壳 | `model.py` | `build_attention_block` + `EfficientAttentionLM` |
| 同输入对比实验 | `demo.py` | 输出统计、相似度、缓存压缩比 |
| 延迟/缓存基准 | `benchmark.py` | seq_len 扫描曲线 |
| 训练对比 | `train.py` | `--variant all` 统一训练对比 |

## 常见误区 + 最小运行命令
### 常见误区
- 误区 1：只要换成 MQA，一定不会损失表达能力。  
  正解：共享 K/V 可能限制头间多样性，通常需要任务层面验证。
- 误区 2：GQA 只是“实现细节”，没有理论意义。  
  正解：GQA 是显式的容量-效率折中，能连续调节共享强度。
- 误区 3：MLA 等于“把维度硬砍掉”。  
  正解：MLA 是压缩后再重构，关键是低秩近似质量与恢复映射能力。
- 误区 4：只看 loss，不看 KV Cache。  
  正解：高效注意力的目标是“效果 + 成本”联合优化。

### 最小运行命令
```bash
python chapter_09_efficient_attention/demo.py
python chapter_09_efficient_attention/benchmark.py --seq_lens 128,256,512
python chapter_09_efficient_attention/train.py --variant all --epochs 1 --num_samples 2000
```

---

## 1. 问题背景：为什么需要高效注意力
在自回归推理中，历史 token 的 $K/V$ 会被缓存。若序列长度为 $S$，batch 为 $B$，每个 $K$ 或 $V$ 的通道数为 $C_{kv}$，元素字节数为 $b$，则 KV Cache 近似为：

$$
\text{KVCacheBytes} \approx 2 \cdot B \cdot S \cdot C_{kv} \cdot b
$$

其中系数 $2$ 来自同时缓存 $K$ 与 $V$。

当 $S$ 很大时，KV Cache 往往成为推理内存瓶颈。

---

## 2. 学术阐述：四种注意力的统一视角

### 2.1 标准缩放点积注意力

$$
\mathrm{Attention}(Q, K, V) = \mathrm{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V
$$

其中 $M$ 是 mask（padding 或 causal）。

### 2.2 MHA（Multi-Head Attention）
每个头有独立 $W_i^Q, W_i^K, W_i^V$：

$$
\text{head}_i = \mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V)
$$

$$
\mathrm{MHA}(Q, K, V) = \mathrm{Concat}(\text{head}_1, ..., \text{head}_h)W^O
$$

KV 通道（单个 $K$ 或 $V$）为 $d_{\text{model}}$。

### 2.3 MQA（Multi-Query Attention）
多个 Query 头共享同一组 $K/V$：

$$
Q_i = XW_i^Q, \quad K = XW^K, \quad V = XW^V
$$

其中 $i = 1, \ldots, h$。KV 通道降为：

$$
C_{kv}^{MQA} = \frac{d_{model}}{h}
$$

### 2.4 GQA（Grouped-Query Attention）
将 $h$ 个 query 头分成 $g$ 组，每组共享一组 $K/V$：

$$
C_{kv}^{GQA} = g \cdot \frac{d_{model}}{h}
$$

当 $g = h$ 时退化为 MHA；当 $g = 1$ 时退化为 MQA。

### 2.5 MLA（Multi-Head Latent Attention）
先压缩：

$$
c_t^K = W_{down}^K h_t, \quad c_t^V = W_{down}^V h_t, \quad c_t^K, c_t^V \in \mathbb{R}^{r}
$$

再重构：

$$
\tilde{K}_t = W_{up}^K c_t^K, \quad \tilde{V}_t = W_{up}^V c_t^V
$$

再进入标准注意力计算。若缓存的是 latent，则 KV 通道（单个 $K$ 或 $V$）近似：

$$
C_{kv}^{MLA} = r
$$

这就是“低秩压缩换缓存”的核心。

### 2.6 从线性代数看表达瓶颈（Why Sharing/Compression Works and Fails）
设 $H \in \mathbb{R}^{S \times d}$ 为一段序列的 hidden states。

对于教学实现中的 MLA，重构可写为：

$$
\tilde{K} = H W_{down}^K W_{up}^K,\quad
\tilde{V} = H W_{down}^V W_{up}^V
$$

因为 $W_{down} \in \mathbb{R}^{d \times r}, W_{up} \in \mathbb{R}^{r \times d}$，所以：

$$
\mathrm{rank}(W_{down}W_{up}) \le r
$$

这意味着 $K/V$ 被限制在至多 $r$ 维子空间中。  
当 $r$ 太小，信息会发生不可逆压缩；当 $r$ 足够大，压缩误差会下降，但缓存优势减弱。

对于 MQA/GQA，共享并不是“降维”，而是“减少独立 $K/V$ 基底数”：
1. MQA：所有 query 头使用同一组 $K/V$ 基底，头间差异主要来自 $Q$。
2. GQA：每组 query 头共享一组 $K/V$ 基底，表达自由度介于 MHA 与 MQA 之间。

### 2.7 MLA 重构误差的 logit 扰动上界
定义某个 token 的 key 重构误差：

$$
e_t^K = K_t - \tilde{K}_t
$$

对任意 query 向量 $q$，缩放前后 logit 误差满足：

$$
\left|\frac{q^\top K_t}{\sqrt{d_k}} - \frac{q^\top \tilde{K}_t}{\sqrt{d_k}}\right|
= \frac{|q^\top e_t^K|}{\sqrt{d_k}}
\le \frac{\|q\|_2\|e_t^K\|_2}{\sqrt{d_k}}
$$

结论：
1. 重构误差 $\|e_t^K\|$ 越小，attention score 扰动越小。
2. $d_k$ 越大时，缩放项 $1/\sqrt{d_k}$ 有助于抑制 score 扰动幅度。
3. MLA 调参本质是在 $r$（压缩率）与 $\|e_t^K\|$（近似误差）之间做折中。

### 2.8 Prefill 与 Decode 的复杂度分解
设 batch 为 $B$，序列长度为 $S$，当前解码步为 $t$。

Prefill（一次性输入整段上下文）主要项：
1. 注意力矩阵计算：$O(B S^2 d)$。
2. 线性投影：$O(B S d^2)$（常数因变体而异）。

Decode（自回归单步）主要项：
1. $Q$ 投影：$O(B d^2)$。
2. $QK^T$ 与 $AV$：$O(B t d)$。
3. 新 token 的 $K/V$ 投影：变体差异最大，决定增量开销与缓存带宽压力。

因此工程上常见现象是：
1. 短序列 prefill 阶段，四变体延迟差距有限。
2. 长上下文 decode 阶段，KV cache 与 K/V 投影成本差异被放大。

---

## 3. 通俗解释（含你指定的类比）
- **MHA**：8 个人（8 heads），每人一本专属字典（独立 KV）。
- **MQA**：8 个人，共用一本字典（大幅省空间）。
- **GQA**：8 个人分 2 组，每组一本字典（折中）。
- **MLA**：先把字典压缩成小抄（latent），用时再还原，既省空间又尽量保细节。

换句话说：
- MHA 追求“每个人的独立表达能力”。
- MQA 追求“最小缓存成本”。
- GQA 在能力与成本间找连续可调平衡。
- MLA 通过“先压缩再重建”走一条结构化压缩路线。

---

## 4. 参数、复杂度与 KV Cache 的分解对比
设 $d = d_{\text{model}}$, $h = \text{num\_heads}$, $g = \text{num\_kv\_heads}$, $r = \text{latent\_dim}$。

### 4.1 KV 通道（单个 K 或 V）
- MHA: $d$
- MQA: $\frac{d}{h}$
- GQA: $g \cdot \frac{d}{h}$
- MLA: $r$

### 4.2 KV Cache 近似

$$
\text{KVCache} \propto 2 \cdot B \cdot S \cdot C_{kv}
$$

因此在固定 $B,S$ 下，缓存占用只由 $C_{kv}$ 控制。

### 4.3 线性层参数量（忽略 bias）
按本章代码的投影结构，参数量近似为：

1. MHA:

$$
P_{MHA} = 4d^2
$$

2. MQA:

$$
P_{MQA} = 2d^2 + 2\frac{d^2}{h}
$$

3. GQA:

$$
P_{GQA} = 2d^2 + 2g\frac{d^2}{h}
$$

4. MLA（教学版）:

$$
P_{MLA} = 2d^2 + 4dr
$$

可见 MLA 是否省参数，关键在 $r$ 与 $d$ 的相对大小。

### 4.4 Decode 阶段新 token 的 K/V 投影开销代理
若只看新 token 的 K/V 投影乘加量级：

1. MHA: $2d^2$
2. MQA: $\frac{2d^2}{h}$
3. GQA: $\frac{2gd^2}{h}$
4. MLA（缓存 latent 的理想化视角）: $2dr$

这解释了为什么在大模型推理里，MQA/GQA/MLA 常被用于降低解码阶段成本。

### 4.5 本章实现边界（避免误读）
本章是教学实现，重点是机制与复杂度认知，不是工业级内核优化：
1. 使用常规 PyTorch 张量算子，不包含 FlashAttention/算子融合。
2. 训练与 demo 采用全序列前向，不实现生产级流式 cache 更新内核。
3. KV Cache 数值对比来自解析估算函数与实验统计，用于趋势判断。

### 4.6 内存预算约束下的选型反推（重要）
给定可用 KV Cache 预算 $M_{\text{budget}}$（bytes），则必须满足：

$$
2 \cdot B \cdot S \cdot C_{kv} \cdot b \le M_{budget}
$$

可得允许的最大 KV 通道：

$$
C_{kv}^{\max} = \frac{M_{\text{budget}}}{2BSb}
$$

于是得到选型约束：
1. MHA 可行条件：$d \le C_{kv}^{\max}$。
2. MQA 可行条件：$\frac{d}{h} \le C_{kv}^{\max}$。
3. GQA 可行条件：

$$
g \cdot \frac{d}{h} \le C_{kv}^{\max}
$$

$$
g \le \left\lfloor \frac{h \, C_{kv}^{\max}}{d} \right\rfloor
$$

4. MLA 可行条件：

$$
r \le C_{kv}^{\max}
$$

这给了一个直接工程规则：  
先由预算求 $C_{kv}^{\max}$，再决定是调 $g$（GQA）还是调 $r$（MLA）。

### 4.7 效果-成本联合目标（理论化选型）
高效注意力选型可写成联合优化问题，而不是只比较单一指标。设变体为 $v \in \{\text{MHA}, \text{MQA}, \text{GQA}, \text{MLA}\}$，参数为 $\theta_v$，定义：

$$
\mathcal{J}(v, \theta_v) =
\mathcal{L}_{task}(\theta_v) +
\lambda_{mem} \cdot \mathrm{Mem}(v) +
\lambda_{lat} \cdot \mathrm{Lat}(v)
$$

其中：
1. $\mathcal{L}_{task}$ 表示任务损失（例如 next-token CE）。
2. $\mathrm{Mem}(v)$ 可由 KV cache 公式估算。
3. $\mathrm{Lat}(v)$ 可由 benchmark 的平均时延统计近似。

当内存约束更严格时，增大 $\lambda_{mem}$；当实时延迟更严格时，增大 $\lambda_{lat}$。  
这解释了为什么“同一个最优变体”在不同业务目标下不一定相同。

---

## 5. 固定数值算例（MB 级）
取：
- $B = 1$
- $S = 4096$
- $d = 512$
- $h = 8$
- $g = 2$
- $r = 64$
- $fp16$（$b = 2$ bytes）

### 5.1 各变体 $C_{kv}$
- MHA: $512$
- MQA: $64$
- GQA: $128$
- MLA: $64$

### 5.2 KV Cache 大小

$$
\text{KVCacheBytes} = 2 \cdot B \cdot S \cdot C_{kv} \cdot b
$$

计算结果：

| 变体 | KV Cache (Bytes) | KV Cache (MB) | 相对 MHA |
|---|---:|---:|---:|
| MHA | 8,388,608 | 8.000 | 1.0x |
| GQA ($g=2$) | 2,097,152 | 2.000 | 4.0x 节省 |
| MQA | 1,048,576 | 1.000 | 8.0x 节省 |
| MLA ($r=64$) | 1,048,576 | 1.000 | 8.0x 节省 |

该结果说明：在这个配置下，MLA 的缓存占用可与 MQA 同级。

### 5.3 参数量与增量投影开销（同一配置）
在 $d=512, h=8, g=2, r=64$ 下：

| 变体 | 参数量（不含 bias） | 相对 MHA | 新 token K/V 投影代理开销 | 相对 MHA |
|---|---:|---:|---:|---:|
| MHA | 1,048,576 | 1.0x | 524,288 | 1.0x |
| MQA | 589,824 | 0.56x | 65,536 | 0.125x |
| GQA | 655,360 | 0.63x | 131,072 | 0.25x |
| MLA | 655,360 | 0.63x | 65,536 | 0.125x |

可见在该配置下：
1. MQA/MLA 的增量 K/V 投影代理开销最低。
2. GQA 在参数与开销上位于 MHA 与 MQA 之间。
3. MLA 的优势是否成立，取决于 $r$ 是否足够小且重构误差可接受。

### 5.4 超长上下文扩展示例（$S = 32768$）
固定 $B = 1, d = 512, h = 8, g = 2, r = 64, fp16\ (b = 2)$：

$$
\text{KVCacheBytes} = 2 \cdot B \cdot S \cdot C_{kv} \cdot b
$$

结果如下：

| 变体 | KV Cache (MB) | 相对 MHA |
|---|---:|---:|
| MHA | 64.000 | 1.0x |
| GQA ($g=2$) | 16.000 | 4.0x 节省 |
| MQA | 8.000 | 8.0x 节省 |
| MLA ($r=64$) | 8.000 | 8.0x 节省 |

当 $S$ 从 $4096$ 提升到 $32768$，缓存需求等比例放大 $8x$，这也是高效注意力在长上下文中价值显著提升的直接原因。

---

## 6. Demo 设计与解释
### 6.1 输入
模拟 hidden states：

$$
X \in \mathbb{R}^{B \times S \times D}
$$

默认由 `torch.randn` 生成。

### 6.2 输出
`demo.py` 对每个变体输出：
1. 输出张量 shape、mean、std、L2 norm。
2. 相对 MHA 的 cosine similarity 与 L2 difference。
3. KV Cache bytes / MB 与压缩比。

并生成：
- `results/attention_compare.json`
- `images/kv_cache_comparison.png`
- `images/output_stat_comparison.png`

说明：
- `results/attention_compare.json` 同时包含 `demo` 与 `training` 两个顶层字段，便于统一复盘。

---

## 7. Simple / Complex 案例解读

### 7.1 Simple Case：短序列（$S=64$）
- 现象：四变体输出统计通常接近，速度差距未必极端。
- 解释：短序列下缓存压力不明显，收益更多体现在常数项。

### 7.2 Complex Case：长序列（$S \ge 1024$）
- 现象：MHA 的 KV Cache 增长最快，GQA/MQA/MLA 更平缓。
- 解释：随着 $S$ 增加，缓存线性放大，高效变体优势被放大。

### 7.3 选型建议（基于理论与本章实验）
1. 若首要目标是最小缓存：优先试 MQA，再看精度是否可接受。
2. 若担心 MQA 表达受限：优先试 GQA，通过调 $g$ 做连续折中。
3. 若希望“压缩可调且可分析”：试 MLA，重点扫描 $r$ 并监控重构误差影响。
4. 生产实践建议使用“两阶段筛选”：
   - 阶段 1：用预算约束（第 4.6 节）筛掉不可行方案。
   - 阶段 2：在可行集合内用 `benchmark + train` 选择 Pareto 最优点（效果-开销前沿）。

---

## 8. 代码入口
| 文件 | 功能 | 入口 |
| :--- | :--- | :--- |
| `common.py` | 配置与复杂度工具 | `AttentionConfig`, `estimate_kv_cache_bytes` |
| `mha.py` | 标准多头注意力 | `MultiHeadAttention.forward()` |
| `mqa.py` | 多 Query 共享 K/V | `MultiQueryAttention.forward()` |
| `gqa.py` | 分组共享 K/V | `GroupedQueryAttention.forward()` |
| `mla.py` | 低秩 latent 压缩注意力 | `MultiHeadLatentAttention.forward()` |
| `dataset.py` | toy next-token 数据 | `ToyNextTokenDataset`, `ToyNextTokenCollator` |
| `model.py` | 统一模型壳 + 工厂函数 | `EfficientAttentionLM`, `build_attention_block` |
| `demo.py` | 同输入四变体对比 | `main()` |
| `benchmark.py` | seq_len 扫描基准 | `main()` |
| `train.py` | 训练与指标落盘 | `train_main()` |

---

## 9. 训练产物
运行：
```bash
python chapter_09_efficient_attention/train.py --variant all --epochs 1 --num_samples 2000
```

默认保存：
- `results/metrics_mha.json`
- `results/metrics_mqa.json`
- `results/metrics_gqa.json`
- `results/metrics_mla.json`
- `results/predictions_mha.json`
- `results/predictions_mqa.json`
- `results/predictions_gqa.json`
- `results/predictions_mla.json`
- `results/attention_compare.json`
- `results/run_config.json`
- `checkpoints/mha_best.pth`
- `checkpoints/mqa_best.pth`
- `checkpoints/gqa_best.pth`
- `checkpoints/mla_best.pth`

---

## 10. 小结
本章核心不是“谁绝对最好”，而是建立一个工程视角：
1. 你可以通过 $C_{kv}$ 快速估算缓存压力。
2. 你可以通过 `demo + benchmark + train` 三层证据综合判断方案。
3. 你可以根据任务场景在 $MHA \leftrightarrow GQA \leftrightarrow MQA/MLA$ 间选择更合适的容量-效率平衡点。
