# Chapter 03: Optimization & Stability - Theoretical Review

## 章节概述 (Overview)

在 Chapter 02 中，我们通过 MLP 证明了神经网络作为通用函数拟合器的能力。然而，拥有一个强大的架构只是第一步。在深度学习（尤其是大模型训练）中，如何让网络**快速收敛**（Optimization）以及在深层结构中保持**数值稳定**（Stability）是决定模型能否训练成功的关键。

本章我们将深入探讨现代深度学习的两大支柱：
1.  **Optimization**: 从 SGD 到 AdamW 的演进，重点解析动量（Momentum）与自适应学习率机制。
2.  **Stability**: 对比 Batch Normalization (BN) 与 Layer Normalization (LN) 的本质区别，以及 Dropout 的集成学习视角。

---

## 1. 优化器演进 (Optimization Evolution)

### 1.1 随机梯度下降 (SGD) 与 动量 (Momentum)

#### 学术阐述 (Academic Elaboration)
SGD 是最基础的优化算法，它通过计算当前 mini-batch 的梯度来更新权重。然而，SGD 在面对**病态曲率**（Pathological Curvature，即在一个方向上坡度陡峭，而在另一个方向上平缓）时，容易发生剧烈的震荡，导致收敛缓慢。

Momentum（动量）方法引入了物理学中的“惯性”概念。它不仅考虑当前的梯度，还累积了历史梯度的指数加权平均。这使得参数更新在梯度方向一致的维度上加速，而在梯度方向改变的维度上抑制震荡。

#### 通俗解释 (Intuitive Explanation)
* **SGD**: 就像一个醉汉下山，每一步都只看脚下的路。如果遇到一个狭长的山谷，他会在山谷两壁之间来回撞击（震荡），很难沿着谷底向下走。
* **Momentum**: 就像一个沉重的铁球滚下山。即使遇到小坑或短暂的上坡，惯性（历史速度）也会带着它继续冲向谷底。在狭长山谷中，铁球的横向摆动会被互相抵消，而沿山谷向下的速度会越来越快。

#### 数学推导 (Mathematical Derivation)

**Vanilla SGD**:
$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta J(\theta_t)$$

**SGD with Momentum**:
引入速度变量 $v$（Velocity）：
$$v_{t+1} = \gamma \cdot v_t + \eta \cdot \nabla_\theta J(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

其中：
* $\eta$: 学习率 (Learning Rate)
* $\gamma$: 动量系数 (Momentum Coefficient, 通常为 0.9)
* $\nabla_\theta J$: 损失函数关于参数的梯度

#### 简单与复杂示例 (Examples)

* **Simple Example (凸函数)**:
    考虑 $f(x) = x^2$。SGD 和 Momentum 都能轻松找到最小值 $x=0$。Momentum 可能会因为惯性冲过头一点点，但能更快到达。

* **Complex Example (Rosenbrock Function / 香蕉函数)**:
    函数 $f(x, y) = (1-x)^2 + 100(y-x^2)^2$ 包含一个狭长的抛物线山谷。
    * **现象**: SGD 会在 $y=x^2$ 的山谷两壁间剧烈震荡，步长必须设得很小，收敛极慢。
    * **Momentum**: 能够积累沿 $x$ 轴正向的速度，抑制 $y$ 轴方向的震荡，从而快速穿过山谷到达全局最优解 $(1, 1)$。

---

### 1.2 自适应优化器: Adam & AdamW

#### 学术阐述 (Academic Elaboration)
SGD 系列对所有参数使用相同的学习率。但在稀疏特征或大模型中，不同参数的梯度量级差异巨大。**Adam (Adaptive Moment Estimation)** 结合了 Momentum (一阶动量) 和 RMSProp (二阶动量/梯度的平方) 的思想，为每个参数动态调整学习率。

**AdamW** 是 Adam 的修正版。研究发现，在 Adam 中直接使用 L2 正则化（Weight Decay）通常是不正确的，因为它与自适应学习率机制耦合了。AdamW 将 Weight Decay 从梯度更新步骤中**解耦**（Decoupled），直接作用于权重，这对 Transformer 类模型的泛化性能至关重要。

#### 通俗解释 (Intuitive Explanation)
* **Adam**: 想象你在下山，但这次你带了智能眼镜。对于平缓的路段（梯度小），眼镜告诉你“这路好走，步子迈大点”（增大有效学习率）；对于陡峭悬崖（梯度大），眼镜告诉你“小心，步子迈小点”（减小有效学习率）。同时，你依然保留了铁球的惯性（Momentum）。
* **AdamW vs Adam**: 在 Adam 中，减重（Weight Decay）是在计算步长时一起考虑的，容易被“智能眼镜”的缩放干扰。AdamW 则是每次走完一步后，无论步子大小，都强制减掉一点体重。这种“分步执行”更符合正则化的本意。

#### 数学推导 (Mathematical Derivation)

**Adam 更新规则**:
1.  计算梯度: $g_t$
2.  更新一阶矩 (Momentum): $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
3.  更新二阶矩 (Variance): $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
4.  偏差修正 (Bias Correction): $\hat{m}_t = m_t / (1 - \beta_1^t), \hat{v}_t = v_t / (1 - \beta_2^t)$
5.  参数更新:
    $$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

**AdamW 的区别**:
AdamW 在上述第 5 步之后，额外独立执行一步衰减：
$$\theta_{t+1} = \theta_{t+1} - \eta \cdot \lambda \cdot \theta_t$$
*(其中 $\lambda$ 是 weight decay 系数)*

#### 代码片段样式 (Snippet)

```python
# 标准 PyTorch 初始化 AdamW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr = 1e-3,
    betas = (0.9, 0.999),
    eps = 1e-8,
    weight_decay = 0.01
)
```

## 2. 稳定性与正则化 (Stability & Regularization)

### 2.1 Normalization: BatchNorm (BN) vs LayerNorm (LN)

#### 学术阐述 (Academic Elaboration)
归一化（Normalization）旨在解决**内部协变量偏移 (Internal Covariate Shift)** 问题，即每一层的输入分布在训练过程中不断变化，导致后续层需要不断适应新的分布。

* **Batch Normalization (BN)**: 在**Batch 维度** ($N$) 上进行归一化。它依赖于 mini-batch 的统计量 $(\mu, \sigma)$。对于 CV 任务（如 CNN），不同样本在同一通道（Channel）上的特征往往具有相似的物理意义，因此 BN 有效。
* **Layer Normalization (LN)**: 在**Feature 维度** ($C/D$) 上进行归一化。它对每个样本独立计算 $(\mu, \sigma)$。对于 NLP 任务（如 Transformer），不同样本（句子）长度不一，且 batch size 往往较小，BN 的统计量极不稳定，而 LN 不依赖 batch 大小，因此成为 LLM 的标配。

#### 通俗解释 (Intuitive Explanation)
想象你在批改试卷（数据）：
* **BN (横向比较)**: 你把全班 32 个同学（Batch=32）的“数学题第1题”（Channel 1）分数拿出来，算出平均分，看谁高谁低。这对图片很有效，因为大家的“第1题”都是“边缘检测”。
* **LN (纵向比较)**: 你只拿“张三”（Sample 1）这一张卷子，算出他所有题目的平均分，对其进行标准化。这对文本很有效，因为句子的长短不一，而且我们更关注一个句子里词与词之间的相对关系，而不是张三和李四之间的关系。

#### 数学推导 (Mathematical Derivation)

假设输入张量 $X$ 维度为 $(N, C, H, W)$ 或 $(N, L, D)$。

**Batch Normalization**:
对 Channel $c$ 的均值计算（跨 Batch $N$ 和空间 $H, W$）：

$$
\mu_c = \frac{1}{N \cdot H \cdot W} \sum_{n, h, w} x_{n, c, h, w}
$$

$$
\sigma_c^2 = \frac{1}{N \cdot H \cdot W} \sum_{n, h, w} (x_{n, c, h, w} - \mu_c)^2
$$

$$
\hat{x} = \frac{x - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} \cdot \gamma + \beta
$$

**Layer Normalization**:
对样本 $n$ 的均值计算（跨 Feature $D$）：

$$
\mu_n = \frac{1}{D} \sum_{d} x_{n, d}
$$

$$
\sigma_n^2 = \frac{1}{D} \sum_{d} (x_{n, d} - \mu_n)^2
$$

*(注意：Transformer 中 LN 通常是对每个 token 的 hidden dimension 进行归一化)*

#### 简单与复杂示例 (Examples)

* **Simple Example (2D Data)**:
    Input: `Batch=2, Dim=3`. Data: `[[1, 2, 3], [4, 5, 6]]`
    * **BN**: 计算 col 0 的 mean = (1+4)/2 = 2.5。归一化列。
    * **LN**: 计算 row 0 的 mean = (1+2+3)/3 = 2.0。归一化行。

* **Complex Example (LLM Training)**:
    Input: `Batch=32, Seq_Len=1024, Dim=4096`.
    * **为何不用 BN**: 如果使用 BN，我们需要跨 32 个句子在 Dim 0 上统计。但如果 batch size 因为显存限制变成 1，BN 无法计算方差（或者方差震荡极大），模型崩塌。
    * **为何用 LN**: LN 对每个 token 的 4096 维特征独立计算。无论 batch 是 1 还是 1000，该 token 的归一化结果不变。这保证了推理（Batch=1）时的数值一致性。

#### 代码片段样式 (Snippet)

```python
# PyTorch 中的实现差异
# BatchNorm: 通常用于 CNN (N, C, H, W)
bn = nn.BatchNorm2d(num_features = 64)

# LayerNorm: 通常用于 NLP (N, L, D), normalized_shape 为 Feature 维度
ln = nn.LayerNorm(normalized_shape = 512)
```

### 2.2 Dropout: 训练与推理的二象性

#### 学术阐述 (Academic Elaboration)
Dropout 是一种正则化技术，通过在训练过程中以概率 $p$ 随机将神经元输出置零，来防止模型过拟合。
Dropout 的核心解释是**集成学习 (Ensemble Learning)**。一个包含 $N$ 个神经元的网络，使用 Dropout 可以看作是 $2^N$ 个共享权重的子网络的隐式集成。

#### 通俗解释 (Intuitive Explanation)
* **训练时 (Training)**: 像是进行“生存训练”。每次训练随机让一部分神经元“罢工”。这迫使剩下的神经元必须更加强壮，不能依赖某个特定的同伴，必须学会独立提取鲁棒的特征。
* **推理时 (Inference)**: 到了真正考试的时候，所有神经元全员上岗。为了保持输出的数值总和与训练时一致（因为训练时只有 $(1-p)$ 的人干活），我们需要对数值进行缩放。

#### 数学推导 (Mathematical Derivation)

**Inverted Dropout (现代常用实现)**:
为了让推理阶段的代码更简单（无需缩放），我们在**训练阶段**就进行缩放。

设输入 $x$，mask向量 $m \sim \text{Bernoulli}(1-p)$ (保留概率为 $1-p$)。

**Training**:

$$
y = \frac{1}{1-p} (x \odot m)
$$

*(解释：除以1-p是为了放大保留下来的数值，保持期望值不变)*

**Inference**:

$$
y = x
$$

*(解释：无需任何操作)*

#### 简单与复杂示例 (Examples)

* **Simple Example**:
    向量 $x = [10, 20, 30, 40]$， $p=0.5$ 。
    * **Training**: 随机 mask 掉两个，比如 $[10, 0, 30, 0]$。放大 2 倍 -> $[20, 0, 60, 0]$。期望值 $\mathbb{E}[y] = [10, 20, 30, 40]$，与原值一致。
    * **Inference**: 直接输出 $[10, 20, 30, 40]$。

* **Complex Example (MC Dropout)**:
    通常 Dropout 在推理时关闭。但如果我们想估计模型的不确定性（Uncertainty），可以在**推理时强行开启 Dropout**，并进行多次前向传播。如果多次预测结果方差很大，说明模型对该输入不确定。这在自动驾驶或医疗诊断（需要可解释性）中非常有用。

### 2.3 RMSNorm: 更加高效的归一化

#### 学术阐述 (Academic Elaboration)
**RMSNorm** 是一种简化的 Layer Normalization。研究表明，LayerNorm 的成功主要归功于其对 **方差（Variance）** 的缩放，而 **均值中心化（Mean Centering）** 带来的收益微乎其微。
RMSNorm 移除了均值计算部分，仅利用均方根（RMS）进行缩放。这不仅减少了计算量（节省了 Mean 的计算和减法操作），还保持了数值稳定性，特别是在 FP16/BF16 训练下具有更好的鲁棒性。

#### 通俗解释 (Intuitive Explanation)
* **LayerNorm**: 像是一个严格的教官，先让大家按高矮排队算出平均身高，把每个人减去平均身高（归零），再统一缩放到标准范围。
* **RMSNorm**: 像是一个务实的教官。他认为大家的平均身高本来就差不多是 0（因为权重初始化通常是 0 均值），没必要每次都费劲算平均值。他直接看大家的“能量大小”（幅值），如果整体能量太强，就统一除以一个系数把能量压下来。简单粗暴，但效果一样好。

#### 数学推导 (Mathematical Derivation)

假设输入向量为 $x$，维度为 $D$。

**LayerNorm 回顾**:

$$
\bar{x} = \frac{x - \mu}{\sigma}, \quad \text{where } \mu = \frac{1}{D}\sum x_i
$$

**RMSNorm 计算**:
我们直接计算均方根 (RMS):

$$
\text{RMS}(x) = \sqrt{\frac{1}{D} \sum_{i=1}^{D} x_i^2 + \epsilon}
$$

归一化与缩放 (Scaling):

$$
\bar{x}_i = \frac{x_i}{\text{RMS}(x)} \cdot g_i
$$

*(其中 $g_i$ 是可学习的缩放参数，RMSNorm 去掉了 LN 中的偏置项 $\beta$)*

#### 简单与复杂示例 (Examples)

* **Simple Example**:
    向量 $x = [3, 4]$。
    * $\text{RMS} = \sqrt{(9+16)/2} = \sqrt{12.5} \approx 3.53$
    * Output $\approx [3/3.53, 4/3.53] \approx [0.85, 1.13]$ (保留了相对大小，限制了幅值)。

* **Complex Example (Llama 2 Architecture)**:
    在 Llama 2 中，RMSNorm 被放置在 Attention 和 FeedForward 层**之前** (Pre-Normalization)。这种组合配合 RMSNorm 的计算优势，使得模型在几千个 GPU 上训练时，通信和计算的开销更低，且数值更不容易溢出。

#### 代码片段样式 (Snippet)

```python
# 手写 RMSNorm (PyTorch)
class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.eps = eps
        # 只有缩放参数 gamma (weight)，没有偏置 beta
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # 计算 RMS
        # x.pow(2).mean(-1, keepdim=True) 计算均方
        # rsqrt 是 1/sqrt 的倒数平方根，速度更快
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
        return norm * self.weight
```

### 2.4 Gradient Clipping: 防止梯度爆炸的阀门

#### 学术阐述 (Academic Elaboration)
在深度神经网络（特别是 RNN 和深层 Transformer）中，反向传播时梯度的累积可能导致**梯度爆炸 (Exploding Gradient)**，使得参数更新步长过大，直接飞出最优解区域，导致 Loss 变为 NaN。
**Gradient Norm Clipping** 通过限制梯度向量的 $L_2$ 范数（Norm）来解决此问题。如果梯度的范数超过阈值，就对其进行等比例缩小（Rescaling），改变梯度的模长但不改变梯度的**方向**。

#### 通俗解释 (Intuitive Explanation)
* **现象**: 就像你在玩赛车游戏，突然踩了一脚油门（梯度极大），车子瞬间冲出赛道撞毁（Loss NaN）。
* **Clipping**: 给车子装一个“限速器”。不管你油门踩多深，如果速度表显示超过 100km/h，系统强制把速度限制在 100km/h，但方向盘的角度不变，保证你依然朝着正确的方向开，只是别开太快翻车了。

#### 数学推导 (Mathematical Derivation)

设所有参数的梯度拼接成一个大向量 $g$。设定阈值为 $C$ (max_norm)。

1.  计算总范数 (Total Norm):

$$
\|g\|_2 = \sqrt{\sum_{\theta \in \Theta} \|\nabla_\theta J(\theta)\|_2^2}
$$

2.  计算缩放系数 (Clipping Coefficient):

$$
\text{scale} = \min\left(1, \frac{C}{\|g\|_2 + \epsilon}\right)
$$

*(如果 $\|g\|_2 \le C$，scale 为 1，不做改变；否则 scale < 1)*

3.  更新梯度:

$$
g \leftarrow g \cdot \text{scale}
$$

#### 简单与复杂示例 (Examples)

* **Simple Example**:
    梯度 $g = [3, 4]$，模长为 5。设定 Clip Threshold $C = 2.5$。
    * 模长 5 > 2.5，需要缩放。
    * Scale = $2.5 / 5 = 0.5$。
    * 新梯度 $g_{new} = [1.5, 2.0]$。方向不变，模长变为 2.5。

* **Complex Example (Training Stability)**:
    在 GPT-3 等大模型训练初期，梯度往往极不稳定。如果不加 Clipping，第一次迭代 Loss 可能直接跳到 10000+。通常我们将 max_norm 设置为 1.0。这意味着所有的参数更新向量组合起来的长度不能超过 1.0。这是一种极强的稳定性保障。

#### 代码片段样式 (Snippet)

```python
# PyTorch 内置的 Gradient Clipping
# 通常在 loss.backward() 之后，optimizer.step() 之前调用
loss.backward()

# 对所有参数的梯度进行裁剪，max_norm 通常设为 1.0
torch.nn.utils.clip_grad_norm_(
    model.parameters(),
    max_norm = 1.0
)

optimizer.step()
```