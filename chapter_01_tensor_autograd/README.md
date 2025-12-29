# Chapter 01: Computational Graph & Autograd - Theoretical Review

## 1. 前言：深度学习的“引擎”

欢迎来到第一章。在编写任何具体的神经网络代码之前，作为一名算法工程师，我们需要深刻理解深度学习框架背后的核心引擎：**计算图（Computational Graph）**与**自动微分（Automatic Differentiation, AD）**。

虽然现代 PyTorch 封装得非常完美，只需调用 `.backward()`，但理解其底层的 **DAG（有向无环图）** 构建逻辑和 **Vector-Jacobian Product (VJP)** 的传播机制，是优化显存（如 Checkpointing 技术）、理解分布式训练（如 Pipeline Parallelism）以及手写复杂算子（Custom Autograd Function）的基础。

> [!NOTE]
> 为了更好地理解代码，本章额外撰写了 PyTorch 相关的 Tensor 维度操作代码以及说明，帮助读者更深入地理解计算图的构建与自动微分的实现。
>
> * [DimTransform Code with Pytorch](dim_transform_torch.py)
> * [Instruction Tutorial](tensor_dim_transform.md)

---

## 2. 核心概念：计算图与 DAG

### 2.1 学术阐述
计算图是一种将数学表达式表示为**有向无环图（Directed Acyclic Graph, DAG）**的形式。
* **节点（Nodes）**：表示变量（Tensor）或操作（Operation/Function）。在 PyTorch 中，节点通常承载了数据（`data`）和梯度（`grad`）。
* **边（Edges）**：表示数据流向（Data Dependency）。若节点 $B$ 的计算依赖于 $A$，则存在一条 $A \to B$ 的边。

### 2.2 通俗解释
想象一条精密的**流水线工厂**。
* 原材料（输入数据 $X$, 权重 $W$）是**叶子节点（Leaf Nodes）**。
* 加工机器（加法、乘法、ReLU）是**中间节点**。
* 最终产品（Loss）是**根节点**。
* **前向传播（Forward）**：原料顺着传送带变成产品的过程。
* **反向传播（Backward）**：当产品出现瑕疵（Loss），质检员拿着“责任清单”（梯度）逆着传送带走，告诉每台机器需要调整多少参数。

---

## 3. 核心机制：链式法则与自动微分

### 3.1 数学推导 (The Chain Rule)

深度学习使用的是**反向模式自动微分（Reverse-mode AD）**。

#### 标量形式 (Scalar Case)
假设 $y = f(u)$ 且 $u = g(x)$，即 $y = f(g(x))$。
根据链式法则：

$$
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial u} \cdot \frac{\partial u}{\partial x}
$$

#### 向量/矩阵形式 (Vector/Matrix Case - The Engineer's View)
这在工程中更为重要。假设 $\mathbf{y} \in \mathbb{R}^m$ 是 $\mathbf{x} \in \mathbb{R}^n$ 的函数。
导数不再是标量，而是**雅可比矩阵（Jacobian Matrix）** $J$：

$$
J = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial y_m}{\partial x_1} & \cdots & \frac{\partial y_m}{\partial x_n} \end{bmatrix}
$$

在反向传播中，我们需要计算标量 Loss $L$ 对向量 $\mathbf{x}$ 的梯度。PyTorch 实际上是在计算 **Vector-Jacobian Product (VJP)**：

$$
\nabla_{\mathbf{x}} L = (\frac{\partial \mathbf{y}}{\partial \mathbf{x}})^T \cdot \nabla_{\mathbf{y}} L
$$

工程上，我们不需要显式构建巨大的雅可比矩阵，而是利用算子的性质直接计算乘积。

---

## 4. 实例分析：从标量到矩阵

为了符合你的认知习惯，我们通过两层例子来拆解。

### 4.1 Simple Example: 标量复合函数
**场景**：计算 $z = (x \cdot y) + \sin(x)$ 的梯度。

1. **分解步骤**：
    * $a = x \cdot y$
    * $b = \sin(x)$
    * $z = a + b$
2. **反向传播推导**：
    我们要计算 $\frac{\partial z}{\partial x}$。
    
    $$
    \frac{\partial z}{\partial x} = \frac{\partial z}{\partial a} \cdot \frac{\partial a}{\partial x} + \frac{\partial z}{\partial b} \cdot \frac{\partial b}{\partial x}
    $$
    
    $$
    \frac{\partial z}{\partial x} = 1 \cdot y + 1 \cdot \cos(x) = y + \cos(x)
    $$

### 4.2 Complex Example: 全连接层 (Linear Layer) 的矩阵求导
**场景**：这是理解 LLM 内部 MLP 层的关键。
前向公式：$Y = XW + b$
* $X \in \mathbb{R}^{B \times I}$ (Batch, Input_dim)
* $W \in \mathbb{R}^{I \times O}$ (Input_dim, Output_dim)
* $b \in \mathbb{R}^{O}$
* $L$ 是最终的标量 Loss。

**假设**：我们已知上一层传回来的梯度（Upstream Gradient）$\frac{\partial L}{\partial Y}$（形状为 $B \times O$）。

**问题**：如何求 $\frac{\partial L}{\partial W}$ 和 $\frac{\partial L}{\partial X}$？

**推导**：
这是一个这一典型的 VJP 问题。利用矩阵微积分的迹（Trace）技巧或维度分析：
1. **对于 $W$**：
    
    $$
    \frac{\partial L}{\partial W} = X^T \cdot \frac{\partial L}{\partial Y}
    $$
    
    * *维度检查*：$(I \times B) \cdot (B \times O) \to (I \times O)$。与 $W$ 形状一致。
2. **对于 $X$**：
    
    $$
    \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \cdot W^T
    $$
    
    * *维度检查*：$(B \times O) \cdot (O \times I) \to (B \times I)$。与 $X$ 形状一致。

**工程启示**：
这就是为什么我们在手动实现 `Function.backward` 时，经常看到大量的 `transpose` 和 `matmul` 操作。如果不理解 VJP，很难写对维度的转换。

---

## 5. PyTorch 的动态图机制 (Define-by-Run)

### 5.1 静态图 vs 动态图
* **Static Graph (TensorFlow v1)**: Define-and-Run。先像写编译器一样定义好完整的图结构，然后塞入数据执行。优点是容易优化（算子融合），缺点是调试困难，无法使用 Python 原生控制流（if/else）。
* **Dynamic Graph (PyTorch)**: Define-by-Run。图的构建是在代码执行时动态生成的。

### 5.2 PyTorch 内部实现
当你执行 `z = x * y` 时，PyTorch 做了两件事：
1. 计算数值结果。
2. 构建图节点：
    * 结果 Tensor `z` 拥有一个 `grad_fn` 属性（例如 `<MulBackward0>`）。
    * 该 `grad_fn` 指向创建它的操作，并持有输入张量（`x`, `y`）的引用（如果是 leaf node）或它们的 `grad_fn`。

当我们调用 `loss.backward()` 时，引擎从根节点出发，沿着 `grad_fn` 链条调用每个 Function 的 `backward` 方法，累加梯度到 `.grad` 属性中。

---

## 6. 推导例子
### 6.1 网络和 Loss

#### 前向定义（单样本，标量 loss）

输入：
$x \in \mathbb{R}^2,\quad y \in \mathbb{R}$

参数：
- $W_1 \in \mathbb{R}^{2\times2},\ b_1 \in \mathbb{R}^2$
- $W_2 \in \mathbb{R}^{2\times2},\ b_2 \in \mathbb{R}^2$
- $W_3 \in \mathbb{R}^{1\times2},\ b_3 \in \mathbb{R}$

---

#### Forward（定义计算图）

$$
\begin{aligned}
h_1 &= W_1 x + b_1 \\
h_2 &= \text{ReLU}(h_1) \\
h_3 &= W_2 h_2 + b_2 \\
\hat y &= W_3 h_3 + b_3 \\
L &= (\hat y - y)^2
\end{aligned}
$$

**到这里为止，只是在定义函数 $L(\theta)$**，还没有任何“训练”发生。

---

### 6.2 反向传播总纲

我们要算的是：

$$
\boxed{
\nabla_\theta L
}
$$

也就是对所有参数的偏导。

反向传播本质是 **链式法则的系统化执行**：

$$
\frac{\partial L}{\partial \theta}=
\frac{\partial L}{\partial \hat y}
\cdot
\frac{\partial \hat y}{\partial h_3}
\cdot
\frac{\partial h_3}{\partial h_2}
\cdot
\frac{\partial h_2}{\partial h_1}
\cdot
\frac{\partial h_1}{\partial \theta}
$$

---

### 6.3 真正开始 backward（一步一步）

#### Step 0：反向传播起点

因为：

$$
L = (\hat y - y)^2
$$

所以：

$$
\boxed{
\frac{\partial L}{\partial \hat y}=
2(\hat y - y)
}
$$

这就是 **loss.backward() 的初始梯度**。

---

#### Step 1：第四层 Linear（$\hat y = W_3 h_3 + b_3$）

**梯度对参数**:

$$
\boxed{
\frac{\partial L}{\partial W_3}=
\frac{\partial L}{\partial \hat y}
\cdot
h_3^\top
}
$$

$$
\boxed{
\frac{\partial L}{\partial b_3}=
\frac{\partial L}{\partial \hat y}
}
$$

**梯度向前传（给下一层）**:

$$
\boxed{
\frac{\partial L}{\partial h_3}=
W_3^\top
\frac{\partial L}{\partial \hat y}
}
$$

这一步就是 **VJP（向量 × Jacobian）**。

---

#### Step 2：第三层 Linear（$h_3 = W_2 h_2 + b_2$）

**参数梯度**:

$$
\boxed{
\frac{\partial L}{\partial W_2}=
\frac{\partial L}{\partial h_3}
\cdot
h_2^\top
}
$$

$$
\boxed{
\frac{\partial L}{\partial b_2}=
\frac{\partial L}{\partial h_3}
}
$$

**继续往前传**:

$$
\boxed{
\frac{\partial L}{\partial h_2}=
W_2^\top
\frac{\partial L}{\partial h_3}
}
$$

---

#### Step 3：ReLU（这是第一个“非线性关卡”）

$$
h_2 = \text{ReLU}(h_1)
$$

ReLU 的导数是：

$$
\frac{\partial h_2}{\partial h_1}=
\mathbb{1}(h_1 > 0)
$$

所以：

$$
\boxed{
\frac{\partial L}{\partial h_1}=
\frac{\partial L}{\partial h_2}
\odot
\mathbb{1}(h_1 > 0)
}
$$

这一步就是为什么 **ReLU 会“截断梯度”**。

---

#### Step 4：第一层 Linear（$h_1 = W_1 x + b_1$）

**参数梯度**:

$$
\boxed{
\frac{\partial L}{\partial W_1}=
\frac{\partial L}{\partial h_1}
\cdot
x^\top
}
$$

$$
\boxed{
\frac{\partial L}{\partial b_1}=
\frac{\partial L}{\partial h_1}
}
$$

到此为止，**所有参数梯度都算完了**。

---

## 7. 完整反向流程总结

> [!TIP]
> **反向传播不是在“算 loss”，而是在把 loss 的导数，一层一层传回去，并在每一层顺手把参数的梯度记下来。**

顺序永远是：`loss → 输出层 → 中间层 → 输入层`

---

## 8. Code Example 浅析

本章代码将不依赖神经网络层（`nn.Linear`），而是直接操作 `Tensor` 来构建计算图，并手动实现 **Vector-Jacobian Product (VJP)**，以此来验证 PyTorch 自动微分的正确性。

1. **[反向传播 Code](autograd.py)**:
    * **Simple Example**: 对应理论部分的 $z = xy + \sin(x)$，验证标量链式法则。
    * **Complex Example**: 对应理论部分的 Linear Layer ($Y=XW+b$)，验证矩阵 VJP。
    * **Four-Layer Network**: 对应理论部分的四层神经网络，验证多参数、多层的反向传播。
    * **Core Logic**: 通过 `torch.autograd.grad` 与我们手写的矩阵微积分公式进行对比，确保误差在 `1e-6` 以内。

2. **[DimTransform Code with Pytorch](dim_transform.py)**:
   (参见上文 Instruction Tutorial)