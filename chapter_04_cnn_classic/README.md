# Chapter 04: Feature Extraction - Convolutional Neural Networks (CNN)

## 1. 核心原理 (Core Principles)

### 1.1 卷积与互相关 (Convolution vs Cross-correlation)

* **学术阐述**：
    在严格的数学定义中，卷积（Convolution）操作涉及两个函数 $f$ 和 $g$，其中 $g$ 需要经过**翻转（Flip）**后再与 $f$ 进行滑动点积。公式如下：
    $$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau)g(t - \tau) d\tau$$
    而在深度学习（包括 PyTorch, TensorFlow）中，我们通常所说的“卷积”实际上是**互相关（Cross-correlation）**。它省去了翻转核（Kernel）的步骤，直接进行滑动窗口的点积运算。

* **通俗解释**：
    想象你在用手电筒照一面墙，手电筒的光圈就是一个“卷积核（Kernel）”。你拿着手电筒从左上角开始，一步步向右、向下扫过整面墙（输入图像）。
    * **数学卷积**：你需要先把手电筒倒过来拿（上下左右翻转），再照墙。
    * **DL 卷积**：直接拿手电筒照墙。因为神经网络的参数是学习出来的，如果特征需要翻转，网络会自动学出翻转后的参数，所以不需要人工预先翻转。

* **简单的代码片段 (Conceptual)**：
    ```python
    import torch
    import torch.nn.functional as F

    # In Deep Learning, we use cross-correlation (no flipping)
    # input shape: (1, 1, 5, 5), weight shape: (1, 1, 3, 3)
    # output = F.conv2d(input, weight) 
    ```

### 1.2 局部连接与权值共享 (Local Connectivity & Weight Sharing)

这是 CNN 区别于全连接层（MLP）并能高效处理图像的两个核心假设。

1.  **局部连接 (Local Connectivity)**：
    * **解释**：输出特征图上的每一个神经元，只与输入图像的一小块区域（感受野）相连，而不是与所有像素相连。
    * **意义**：大幅减少了连接数量，符合图像特征的局部性（例如眼睛只是脸部的一部分，不需要看脚趾）。
2.  **权值共享 (Weight Sharing)**：
    * **解释**：同一个卷积核（Filter）在遍历整张图像时，其参数（权重 $W$ 和偏置 $b$）是固定的。
    * **意义**：参数量不再随图像尺寸增加而线性爆炸。同时实现了**平移不变性 (Translation Invariance)** —— 即无论猫在图片的左上角还是右下角，同一个“猫耳检测器”卷积核都能识别出来。

### 1.3 感受野 (Receptive Field)

* **概念**：
    CNN 输出特征图（Feature Map）上的一个像素点，在原始输入图像上所能“看”到的区域大小。层数越深，感受野越大，提取的特征越抽象（语义级）。

* **计算逻辑**：
    假设第 $l$ 层的感受野为 $RF_l$，其计算公式为递归形式：
    $$ RF_{l} = RF_{l-1} + (k_l - 1) \times \prod_{i=1}^{l-1} s_i $$
    其中：
    * $k_l$：第 $l$ 层的卷积核大小 (kernel size)。
    * $s_i$：第 $i$ 层的步长 (stride)。
    * $RF_0 = 1$ (输入层的单个像素)。

---

## 2. PyTorch 实践细节 (PyTorch Implementation Details)

### 2.1 `nn.Conv2d` 关键参数详解

构造函数：`nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)`

1.  **`in_channels` & `out_channels`**：
    * 输入特征图的深度（如 RGB 图像为 3）和输出特征图的深度（即卷积核的数量）。
2.  **`kernel_size` ($k$)**：
    * 卷积核的空间尺寸，通常为 1, 3, 5, 7。
3.  **`stride` ($s$)**：
    * 滑动的步长。$s > 1$ 会导致降采样（Downsampling）。
4.  **`padding` ($p$)**：
    * 边缘填充。通常用于保持特征图尺寸不变（如 $k=3, p=1$）。
5.  **`dilation` ($d$)**：
    * 空洞卷积（Atrous Convolution）。在卷积核元素之间插入空格，用于在不增加参数量的情况下扩大感受野。
6.  **`groups` ($g$)**：
    * 分组卷积。默认 $g=1$（全连接卷积）。
    * 当 $g = \text{in\\_channels} = \text{out\\_channels}$ 时，为 **Depthwise Convolution**（如 MobileNet），每个通道独立卷积，极大降低参数量。

### 2.2 维度变化公式 (Dimension Arithmetic)

对于输入维度 $(N, C_{in}, H_{in}, W_{in})$，经过 `Conv2d` 后输出 $(N, C_{out}, H_{out}, W_{out})$：

$$ 
H_{out} = \lfloor \frac{H_{in} + 2 \times \text{padding}[0] - \text{dilation}[0] \times (\text{kernel\\_size}[0] - 1) - 1}{\text{stride}[0]} + 1 \rfloor 
$$

*注：$W_{out}$ 的计算同理。*

### 2.3 池化层 (Pooling)

* **作用**：
    1.  **降采样**：减小特征图尺寸，降低计算量。
    2.  **不变性增强**：对微小的平移和形变具有鲁棒性。
* **权衡**：
    * **Max Pooling**：提取最显著特征（纹理、边缘），但丢失背景信息。
    * **Average Pooling**：保留背景信息，平滑特征，但可能模糊边缘。
    * *现代趋势*：在某些架构中（如 ResNet, Transformer），倾向于使用 Stride Convolution 代替 Pooling 进行降采样，以让网络自行学习如何保留信息。

---

## 3. 示例与推演 (Examples & Derivation)

### 3.1 Simple Example: 手动数值推演 (Manual Calculation)

**场景**：
* **Input**: $5 \times 5$ 单通道矩阵 (No padding, Stride = 1).
* **Kernel**: $3 \times 3$ 卷积核.

**数据**：

$$
\text{Input} = \begin{bmatrix}
1 & 1 & 1 & 0 & 0 \\
0 & 1 & 1 & 1 & 0 \\
0 & 0 & 1 & 1 & 1 \\
0 & 0 & 1 & 1 & 0 \\
0 & 1 & 1 & 0 & 0
\end{bmatrix}, \quad
\text{Kernel} = \begin{bmatrix}
1 & 0 & 1 \\
0 & 1 & 0 \\
1 & 0 & 1
\end{bmatrix}
$$

**计算输出特征图左上角第一个像素 ($O_{0,0}$)**：
覆盖区域为 Input 的左上 $3 \times 3$ 子块：

$$
\text{Region} = \begin{bmatrix}
1 & 1 & 1 \\
0 & 1 & 1 \\
0 & 0 & 1
\end{bmatrix}
$$

进行 **Element-wise Product** (Hadamard Product) 然后 **Sum**:

$$
\begin{aligned}
O_{0,0} &= (1 \times 1) + (1 \times 0) + (1 \times 1) \\
        &+ (0 \times 0) + (1 \times 1) + (1 \times 0) \\
        &+ (0 \times 1) + (0 \times 0) + (1 \times 1) \\
        &= 1 + 0 + 1 + 0 + 1 + 0 + 0 + 0 + 1 \\
        &= 4
\end{aligned}
$$

输出尺寸计算：$(5 - 3) / 1 + 1 = 3$，故输出为 $3 \times 3$ 矩阵。

### 3.2 Complex Example: 多通道工程化计算

**场景**：
我们设计一个标准的 CNN 特征提取层。
* **Input**: RGB 图像 Batch, Shape = $(N, 3, 32, 32)$。
* **Layer**: `nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1)`。

**Step 1: 维度变换计算**

$$
\begin{aligned}
H_{out} &= \lfloor \frac{32 + 2 \times 1 - 1 \times (3 - 1) - 1}{2} + 1 \rfloor \\
        &= \lfloor \frac{32 + 2 - 2 - 1}{2} + 1 \rfloor \\
        &= \lfloor \frac{31}{2} + 1 \rfloor = 15 + 1 = 16
\end{aligned}
$$

输出 Tensor 形状: $(N, 16, 16, 16)$。
*解释：Stride=2 导致尺寸减半（下采样）。*

**Step 2: 参数量统计 (Parameter Count)**
每一个输出通道对应一个独立的 3D 卷积核（维度为 $C_{in} \times K \times K$）。

$$
\begin{aligned}
\text{Params} &= (\text{kernel\\_h} \times \text{kernel\\_w} \times \text{in\\_channels}) \times \text{out\\_channels} + \text{bias} \\
              &= (3 \times 3 \times 3) \times 16 + 16 \\
              &= 27 \times 16 + 16 \\
              &= 432 + 16 = 448
\end{aligned}
$$

**Code Snippet (Reference Style)**:
```python
import torch
import torch.nn as nn

def calculate_cnn_layer_info():
    """
    Demonstrate complex CNN layer calculation.
    """
    # Define layers
    conv_layer = nn.Conv2d(
        in_channels = 3,
        out_channels = 16,
        kernel_size = 3,
        stride = 2,
        padding = 1
    )
    
    # Dummy input: Batch=8, RGB, 32x32
    dummy_input = torch.randn(8, 3, 32, 32)
    
    # Forward pass
    output = conv_layer(dummy_input)
    
    # Analysis
    print(f"Input Shape: {dummy_input.shape}")   # torch.Size([8, 3, 32, 32])
    print(f"Output Shape: {output.shape}")       # torch.Size([8, 16, 16, 16])
    print(f"Weight Shape: {conv_layer.weight.shape}") # torch.Size([16, 3, 3, 3])
    
    # Verify parameter count manually
    total_params = sum(p.numel() for p in conv_layer.parameters())
    expected_params = (3 * 3 * 3 * 16) + 16
    assert total_params == expected_params, f"Mismatch: {total_params} vs {expected_params}"

# This fits your preference for clear, verifiable logic.
```