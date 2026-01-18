# Deep-Learning-From-Scratch-PyTorch
> 从零开始的深度学习教程：从传统神经网络到现代Transformer，通过PyTorch实现深度学习的核心原理与前沿技术

## 🎯 项目目标
本项目旨在提供一个**全面、系统、易于理解**的深度学习教程，帮助学习者从基础概念逐步深入到前沿技术。通过"从零实现"的方式，让学习者真正掌握深度学习的核心原理和实现细节。

## ⭐ 项目亮点

### 🏆 已完成章节特色

#### 🔬 第1章：计算图与自动微分
- ✅ **独家实现**：从零构建自动微分引擎，理解PyTorch核心原理
- ✅ **可视化支持**：生成4层计算图，直观理解梯度传播
- ✅ **理论基础**：深入讲解VJP（向量雅可比积）机制

#### 🧠 第2章：万能逼近器（MLP）  
- ✅ **三种模式**：标准拟合、2D表面、外推实验，全面验证MLP能力
- ✅ **预训练模型**：提供三种场景的预训练权重，直接体验效果
- ✅ **数学证明**：结合实验验证万能逼近定理

#### ⚡ 第3章：优化与正则化
- ✅ **算法对比**：SGD、Adam、RMSprop等9种优化器性能对比
- ✅ **正则化技术**：L1/L2、Dropout、BatchNorm等完整实现
- ✅ **可视化分析**：训练曲线对比，直观理解各算法特点

#### 🖼️ 第4章：经典CNN
- ✅ **卷积数学**：手写卷积操作，理解卷积核工作原理
- ✅ **特征可视化**：特征图、滤波器可视化，洞察CNN内部机制
- ✅ **完整pipeline**：CIFAR-10数据集训练+推理+可视化一体化

#### 🚀 第5章：ResNet与现代CNN
- ✅ **理论深度**：340行详细文档，从数学推导到工程实现
- ✅ **A/B实验**：ResNet vs PlainNet对比，验证残差连接关键作用
- ✅ **现代技术**：Grad-CAM、零初始化、全局平均池化等前沿技术
- ✅ **预训练模型**：提供训练好的ResNet-18和PlainNet-18模型

### 📊 数据统计
- **📚 理论文档**：5章详细教程，累计1000+行理论说明
- **💻 代码实现**：30+个独立脚本，覆盖所有核心算法
- **🎯 实验场景**：15+个不同实验，验证各种技术效果
- **🏭 预训练模型**：8个预训练模型，直接体验最佳效果
- **📈 可视化图表**：20+张分析图表，直观理解技术差异

## ✨ 项目特点
- **理论与实践结合**：每个章节包含详细的理论讲解和可运行的代码实现
- **从零实现**：关键算法和模型均从零开始实现，避免黑盒调用
- **模块化设计**：代码结构清晰，便于理解和扩展
- **丰富的示例**：包含多种数据集和应用场景的示例
- **最新技术**：涵盖从传统神经网络到现代Transformer、LLM的完整技术栈
- **教学导向**：注重概念解释和直观理解，适合自学和教学使用
- **渐进式学习**：从基础概念到前沿技术，循序渐进的学习路径

## 📁 项目目录结构
```ascii
Deep-Learning-From-Scratch-PyTorch/
├── README.md              # 项目说明文档
├── requirements.txt       # 依赖库列表
├── utils/                 # 通用工具箱
│   ├── __init__.py
│   ├── device.py          # TorchDevice工具
│   ├── file_io_util.py    # 文件IO工具
│   ├── model_summary.py   # 模型参数查看工具
│   ├── seed.py            # 随机种子固定工具
│   └── timer.py           # 时间测量工具
├── chapter_01_tensor_autograd/   # 第1章：计算图与自动微分
├── chapter_02_nn_basics_mlp/      # 第2章：万能逼近器（MLP）
├── chapter_03_optimization_regularization/  # 第3章：优化与正则化
├── chapter_04_cnn_classic/        # 第4章：经典CNN
├── chapter_05_resnet_modern_cnn/  # 第5章：ResNet与现代CNN（已完成）
├── chapter_06_rnn_lstm_seq/       # 第6章：RNN系列（规划中）
├── chapter_07_attention_mechanism/ # 第7章：注意力机制（规划中）
├── chapter_08_transformer_vanilla/ # 第8章：原始Transformer（规划中）
├── chapter_09_efficient_attention/ # 第9章：高效注意力（规划中）
└── chapter_10_llm_modern_components/ # 第10章：LLM现代组件（规划中）
```

## 🎯 学习路径指南

### 🔰 初学者路径（建议学习时间：3-5天）
**目标**：掌握深度学习基础概念和PyTorch基本操作
1. **第1章** → 理解自动微分和计算图原理（2-3天）
2. **第2章** → 掌握MLP结构和万能逼近定理（3-4天）
3. **第3章** → 学习优化算法和正则化技术（4-5天）
4. **综合练习** → 使用所学知识解决简单回归/分类问题（3-5天）

### 🚀 进阶路径（建议学习时间：1-2周）
**目标**：掌握现代深度学习核心技术
1. **第4章** → 深入理解CNN原理和卷积操作（5-7天）
2. **第5章** → 掌握残差连接和现代CNN架构（7-10天）
3. **项目实践** → 结合CNN完成图像分类项目（7-14天）

### 🎆 高级路径（开发中）
**目标**：掌握序列建模和注意力机制
1. **第6-7章** → RNN/LSTM和注意力机制（开发中）
2. **第8-10章** → Transformer和现代LLM技术（开发中）

## 📊 技术演进图谱
```
基础层：自动微分 → 神经网络 → 优化正则化
    ↓
视觉层：经典CNN → ResNet架构 → 现代CNN变体
    ↓
序列层：RNN/LSTM → 注意力机制 → Transformer
    ↓
现代层：高效注意力 → 大语言模型 → 多模态模型
```

## 📦 安装指南

### 1. 克隆仓库
```bash
git clone https://github.com/BrenchCC/Deep-Learning-From-Scratch-PyTorch.git
cd Deep-Learning-From-Scratch-PyTorch
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 验证安装
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## 📖 章节内容

### 🔗 章节关联性说明
```
Chapter 1 (自动微分) 
    ↓ 基础：理解梯度计算
Chapter 2 (MLP) 
    ↓ 架构：基础网络结构
Chapter 3 (优化正则化) 
    ↓ 训练：模型优化技巧
Chapter 4 (CNN) 
    ↓ 视觉：图像特征提取
Chapter 5 (ResNet) 
    ↓ 深度：解决退化问题
Chapter 6-10 (序列→注意力→Transformer→LLM) 
    ↓ 现代：大模型技术栈
```

### 🎯 学习建议
- **按顺序学习**：章节间存在知识依赖，建议按编号顺序学习
- **理论+实践**：每个章节都包含理论文档和可运行代码，建议结合学习
- **动手实验**：每个章节都提供实验脚本，建议修改参数观察效果
- **项目应用**：完成基础篇后可尝试简单的深度学习项目

### 基础篇

#### [Chapter 01: 计算图与自动微分](chapter_01_tensor_autograd/) ✅
- **核心内容**：计算图（DAG）构建逻辑、向量雅可比积（VJP）传播机制
- **关键技术**：PyTorch动态图机制、手动实现自动微分、张量维度操作
- **应用场景**：优化显存（Checkpointing）、分布式训练、自定义算子
- **代码实现**：
  - [自动微分示例](chapter_01_tensor_autograd/autograd.py)
  - [图可视化](chapter_01_tensor_autograd/graph_visualization.py)
  - [张量维度操作](chapter_01_tensor_autograd/dim_transform_torch.py)

#### [Chapter 02: 万能逼近器（MLP）](chapter_02_nn_basics_mlp/) ✅
- **核心内容**：多层感知机结构、激活函数作用、万能逼近定理
- **关键技术**：全连接层、ReLU激活函数、损失函数设计
- **应用场景**：回归问题、分类问题、函数拟合
- **代码实现**：
  - [MLP实现](chapter_02_nn_basics_mlp/mlp.py)
  - [数据文件](chapter_02_nn_basics_mlp/data/)
  - [预训练模型](chapter_02_nn_basics_mlp/models/)

#### [Chapter 03: 优化与正则化](chapter_03_optimization_regularization/) ✅
- **核心内容**：优化算法原理、正则化技术、模型泛化
- **关键技术**：SGD、Adam、L1/L2正则化、Dropout、Batch Norm
- **应用场景**：模型训练优化、防止过拟合、提高泛化能力
- **代码实现**：
  - [优化器实验](chapter_03_optimization_regularization/exp_optimization.py)
  - [正则化实验](chapter_03_optimization_regularization/exp_regularization.py)
  - [标准化实验](chapter_03_optimization_regularization/exp_normalization.py)

### 进阶篇

#### [Chapter 04: 经典CNN](chapter_04_cnn_classic/) ✅
- **核心内容**：卷积神经网络原理、卷积与互相关的数学定义、局部连接与权值共享机制、感受野计算
- **关键技术**：卷积操作、池化层、BatchNorm、CNN架构设计、特征图可视化
- **应用场景**：图像分类、目标检测、特征提取、可视化理解
- **代码实现**：
  - [卷积数学演示](chapter_04_cnn_classic/demo_conv_math.py)
  - [CNN模型实现](chapter_04_cnn_classic/model.py)
  - [CIFAR-10训练](chapter_04_cnn_classic/train.py)
  - [图像推理与可视化](chapter_04_cnn_classic/inference.py)

#### [Chapter 05: ResNet与现代CNN](chapter_05_resnet_modern_cnn/) ✅
- **核心内容**：残差连接原理、退化问题解决方案、现代CNN架构演进、CNN发展史
- **关键技术**：ResNet残差块、Bottleneck结构、Shortcut连接、零初始化技巧、全局平均池化
- **理论深度**：梯度高速公路证明、参数效率分析、Pre-activation vs Post-activation、V1 vs V2对比
- **应用场景**：深度模型训练、图像分类、特征提取、网络架构设计、残差连接在Transformer中的应用基础
- **实验设计**：ResNet vs PlainNet对比实验，验证残差连接对深度网络训练的关键作用
- **代码实现**：
  - [ResNet模型实现](chapter_05_resnet_modern_cnn/src/model.py) - 支持BasicBlock和Bottleneck，可开关残差连接
  - [STL-10数据集处理](chapter_05_resnet_modern_cnn/src/dataset.py) - 完整的STL-10数据加载和预处理
  - [ResNet训练与对比实验](chapter_05_resnet_modern_cnn/src/model_train.py) - A/B测试验证残差连接效果
  - [Grad-CAM可视化](chapter_05_resnet_modern_cnn/src/cam.py) - 模型决策区域可视化
  - [模型推理](chapter_05_resnet_modern_cnn/src/inference.py) - 单张图像推理和特征提取
- **预训练模型**：
  - [ResNet-18 STL-10](chapter_05_resnet_modern_cnn/checkpoints/resnet18_stl10.pth) - 在STL-10上训练的ResNet-18模型
  - [PlainNet-18 STL-10](chapter_05_resnet_modern_cnn/checkpoints/plainnet18_stl10.pth) - 无残差连接的对比模型
- **可视化结果**：
  - [训练对比图](chapter_05_resnet_modern_cnn/results/loss_comparison.png) - ResNet vs PlainNet训练曲线对比
  - [Grad-CAM可视化](chapter_05_resnet_modern_cnn/results/cam_vis/) - 模型注意力热图

#### [Chapter 06: RNN系列](chapter_06_rnn_lstm_seq/) 🚧
- **核心内容**：循环神经网络原理、序列建模技术
- **关键技术**：RNN、LSTM、GRU、双向RNN
- **应用场景**：自然语言处理、时间序列预测、语音识别
- **状态**：规划中

### 高级篇

#### [Chapter 07: 注意力机制](chapter_07_attention_mechanism/) 🚧
- **核心内容**：注意力机制原理、各种注意力变体
- **关键技术**：自注意力、多头注意力、注意力可视化
- **应用场景**：机器翻译、文本摘要、图像描述
- **状态**：规划中

#### [Chapter 08: 原始Transformer](chapter_08_transformer_vanilla/) 🚧
- **核心内容**：Transformer架构原理、从零实现Transformer
- **关键技术**：编码器-解码器结构、位置编码、层归一化
- **应用场景**：机器翻译、语言建模、预训练模型
- **状态**：规划中

#### [Chapter 09: 高效注意力](chapter_09_efficient_attention/) 🚧
- **核心内容**：注意力机制的效率优化、各种高效注意力变体
- **关键技术**：线性注意力、局部注意力、稀疏注意力
- **应用场景**：长序列建模、大模型训练、资源受限环境
- **状态**：规划中

#### [Chapter 10: LLM现代组件](chapter_10_llm_modern_components/) 🚧
- **核心内容**：大语言模型的关键组件、现代LLM技术
- **关键技术**：缩放规律、指令微调、对齐技术、高效训练
- **应用场景**：对话系统、文本生成、代码生成
- **状态**：规划中


## 🚀 快速开始

### 🏃‍♂️ 5分钟快速体验
```bash
# 1. 克隆项目
git clone https://github.com/BrenchCC/Deep-Learning-From-Scratch-PyTorch.git
cd Deep-Learning-From-Scratch-PyTorch

# 2. 安装依赖
pip install -r requirements.txt

# 3. 验证环境
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"

# 4. 运行第一个示例（自动微分）
python chapter_01_tensor_autograd/autograd.py

# 5. 体验MLP函数拟合
python chapter_02_nn_basics_mlp/mlp.py --mode standard
```


## 🚀 详细使用方法

每个章节都是独立的，可以单独学习和运行。以下是已完成章节的详细使用示例：

### Chapter 01: 计算图与自动微分
```bash
# 查看自动微分示例
python chapter_01_tensor_autograd/autograd.py
# 生成计算图可视化
python chapter_01_tensor_autograd/graph_visualization.py
```

### Chapter 02: 万能逼近器（MLP）
```bash
# 运行MLP示例（三种模式：standard, 2d_surface, extrapolate）
python chapter_02_nn_basics_mlp/mlp.py --mode standard
# 运行2D表面拟合
python chapter_02_nn_basics_mlp/mlp.py --mode 2d_surface
# 运行外推实验
python chapter_02_nn_basics_mlp/mlp.py --mode extrapolate
```

### Chapter 03: 优化与正则化
```bash
# 比较不同优化器性能
python chapter_03_optimization_regularization/exp_optimization.py
# 比较不同正则化技术
python chapter_03_optimization_regularization/exp_regularization.py
# 比较不同标准化技术
python chapter_03_optimization_regularization/exp_normalization.py
```

### Chapter 04: 经典CNN
```bash
# 运行卷积数学演示
python chapter_04_cnn_classic/demo_conv_math.py
# 训练CNN模型（CIFAR-10数据集）
python chapter_04_cnn_classic/train.py --epochs 100 --batch_size 128 --lr 0.001
# 对自定义图像进行推理并可视化特征图
python chapter_04_cnn_classic/inference.py --img_dir ./chapter_04_cnn_classic/data/custom_imgs --model_path ./chapter_04_cnn_classic/results/best_model.pth
```

### Chapter 05: ResNet与现代CNN
```bash
# 训练ResNet-18模型（STL-10数据集）
python chapter_05_resnet_modern_cnn/src/model_train.py --epochs 100 --batch_size 64 --lr 0.1
# 训练PlainNet-18模型（无残差连接，对比实验）
python chapter_05_resnet_modern_cnn/src/model_train.py --model_type plainnet --epochs 100 --batch_size 64 --lr 0.1
# 使用预训练模型进行推理
python chapter_05_resnet_modern_cnn/src/inference.py --model_path ./chapter_05_resnet_modern_cnn/checkpoints/resnet18_stl10.pth --image_path ./chapter_05_resnet_modern_cnn/images/airplane.png
# Grad-CAM可视化
python chapter_05_resnet_modern_cnn/src/cam.py --model_path ./chapter_05_resnet_modern_cnn/checkpoints/resnet18_stl10.pth --image_path ./chapter_05_resnet_modern_cnn/images/airplane.png
```

## 📊 项目进度

### 🎯 总体完成度：**50%** (5/10 章节已完成)

#### ✅ 基础篇 (100% 完成)
- **Chapter 01**: 计算图与自动微分 ████████████ 100%
- **Chapter 02**: 万能逼近器（MLP） ████████████ 100%  
- **Chapter 03**: 优化与正则化 ████████████ 100%

#### ✅ 进阶篇 (67% 完成)
- **Chapter 04**: 经典CNN ████████████ 100%
- **Chapter 05**: ResNet与现代CNN ████████████ 100%
- **Chapter 06**: RNN系列 ░░░░░░░░░░ 0%

#### 🚧 高级篇 (0% 完成)
- **Chapter 07**: 注意力机制 ░░░░░░░░░░ 0%
- **Chapter 08**: 原始Transformer ░░░░░░░░░░ 0%
- **Chapter 09**: 高效注意力 ░░░░░░░░░░ 0%
- **Chapter 10**: LLM现代组件 ░░░░░░░░░░ 0%

### 📋 详细进度表

| 章节 | 状态 | 完成度 | 备注 | 预计发布时间 |
|------|------|--------|------|-------------|
| **Chapter 01: 计算图与自动微分** | ✅ 已完成 | 100% | 包含自动微分示例和图可视化 | 已发布 |
| **Chapter 02: 万能逼近器（MLP）** | ✅ 已完成 | 100% | 包含三种实验模式和预训练模型 | 已发布 |
| **Chapter 03: 优化与正则化** | ✅ 已完成 | 100% | 包含优化器、正则化和标准化实验 | 已发布 |
| **Chapter 04: 经典CNN** | ✅ 已完成 | 100% | 包含卷积数学演示、CNN模型实现、CIFAR-10训练和推理可视化 | 已发布 |
| **Chapter 05: ResNet与现代CNN** | ✅ 已完成 | 100% | 包含完整的CNN发展史、残差连接理论推导、ResNet vs PlainNet对比实验、Grad-CAM可视化、预训练模型和详细的理论文档 | 已发布 |
| **Chapter 06: RNN系列** | 🚧 开发中 | 30% | 基础RNN、LSTM、GRU实现 | 2025年1月 |
| **Chapter 07: 注意力机制** | 🚧 规划中 | 0% | 自注意力、多头注意力、注意力可视化 | 2025年1月 |
| **Chapter 08: 原始Transformer** | 🚧 规划中 | 0% | 完整Transformer实现、位置编码、层归一化 | 2025年2月 |
| **Chapter 09: 高效注意力** | 🚧 规划中 | 0% | 线性注意力、局部注意力、稀疏注意力 | 2025年2月 |
| **Chapter 10: LLM现代组件** | 🚧 规划中 | 0% | 缩放规律、指令微调、RLHF、高效训练 | 2025年2月 |

## 🤝 贡献指南
1. **Fork** 本仓库
2. **创建分支** (`git checkout -b feature/AmazingFeature`)
3. **开发测试**（确保代码可运行且有适当注释）
4. **提交更改** (`git commit -m 'Add: 详细描述你的更改'`)
5. **推送到分支** (`git push origin feature/AmazingFeature`)
6. **开启Pull Request**（详细描述更改内容和测试情况）

## 📊 项目统计
### 🎯 学习成果
```
✅ 基础算法：从0实现自动微分、线性回归、逻辑回归
✅ 神经网络：多层感知机、反向传播算法、参数初始化
✅ 现代CNN：ResNet残差网络、批量归一化、正则化技术
✅ 优化算法：SGD、Momentum、Adam等7种优化器
✅ 正则化：L1/L2正则化、Dropout、数据增强
✅ 评估指标：准确率、损失曲线、混淆矩阵、Grad-CAM
```

### 🌟 特色功能
```
🔄 渐进式学习：从数学原理到PyTorch实现的完整路径
📊 对比实验：A/B测试验证不同技术的效果
🎨 可视化：训练过程、模型结构、特征图可视化
💾 预训练模型：提供训练好的模型权重文件
📱 交互式演示：支持Jupyter Notebook和脚本两种模式
```

### 🎓 教学价值
```
📖 理论深度：每个算法都有详细的数学推导
💻 实践完整：从数据处理到模型部署的全流程
🔍 调试友好：详细的代码注释和错误处理
📊 结果直观：丰富的图表和可视化展示
📝 文档齐全：中英文双语注释和详细说明
```

## 🙏 致谢

### 📚 学术资源
- **[Deep Learning](http://www.deeplearningbook.org/)** - Ian Goodfellow, Yoshua Bengio 和 Aaron Courville 的经典教材
- **[PyTorch官方文档](https://pytorch.org/docs/stable/index.html)** - 提供了详细的API参考和教程
- **[CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)** - 斯坦福大学的深度学习课程
- **[fast.ai](https://www.fast.ai/)** - 实用的深度学习课程和资源

### 🛠️ 开源项目
- **[PyTorch](https://pytorch.org/)** - 深度学习框架，让本项目的实现成为可能
- **[NumPy](https://numpy.org/)** - 科学计算的基础库
- **[Matplotlib](https://matplotlib.org/)** - 数据可视化的强大工具
- **[scikit-learn](https://scikit-learn.org/)** - 机器学习算法的参考实现

### 🎓 参考内容
- **[3Blue1Brown](https://www.3blue1brown.com/)** - 用直观的方式解释数学概念
- **[Andrej Karpathy的博客](http://karpathy.github.io/)** - 深度学习的深入分析
- **[Distill.pub](https://distill.pub/)** - 交互式的机器学习研究出版物

### 👥 社区支持
感谢所有关注、使用和贡献本项目的开发者们！您的反馈和建议让这个项目变得更好。

特别感谢：
- 提出宝贵Issue和建议的用户们
- 分享学习心得和使用经验的同学们
- 帮助改进文档和代码的贡献者们

## 📄 许可证

本项目采用Apache License 2.0许可证，详情请见[LICENSE](LICENSE)文件。

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：
- Email: brenchchen.77@example.com
- GitHub Issues: https://github.com/BrenchCC/Deep-Learning-From-Scratch-PyTorch/issues

## 🌟 Star History

如果这个项目对您有帮助，请给它一个Star！

[![Star History Chart](https://api.star-history.com/svg?repos=BrenchCC/Deep-Learning-From-Scratch-PyTorch&type=Date)](https://star-history.com/#BrenchCC/Deep-Learning-From-Scratch-PyTorch&Date)

---

**Happy Learning! 🎉**