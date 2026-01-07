# Deep-Learning-From-Scratch-PyTorch
> 从零开始的深度学习教程：从传统神经网络到现代Transformer，通过PyTorch实现深度学习的核心原理与前沿技术

## 🎯 项目目标
本项目旨在提供一个**全面、系统、易于理解**的深度学习教程，帮助学习者从基础概念逐步深入到前沿技术。通过"从零实现"的方式，让学习者真正掌握深度学习的核心原理和实现细节。

## ✨ 项目特点
- **理论与实践结合**：每个章节包含详细的理论讲解和可运行的代码实现
- **从零实现**：关键算法和模型均从零开始实现，避免黑盒调用
- **模块化设计**：代码结构清晰，便于理解和扩展
- **丰富的示例**：包含多种数据集和应用场景的示例
- **最新技术**：涵盖从传统神经网络到现代Transformer、LLM的完整技术栈
- **教学导向**：注重概念解释和直观理解，适合自学和教学使用

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
├── chapter_05_resnet_modern_cnn/  # 第5章：ResNet与现代CNN（规划中）
├── chapter_06_rnn_lstm_seq/       # 第6章：RNN系列（规划中）
├── chapter_07_attention_mechanism/ # 第7章：注意力机制（规划中）
├── chapter_08_transformer_vanilla/ # 第8章：原始Transformer（规划中）
├── chapter_09_efficient_attention/ # 第9章：高效注意力（规划中）
└── chapter_10_llm_modern_components/ # 第10章：LLM现代组件（规划中）
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
- **核心内容**：卷积神经网络原理、经典CNN架构
- **关键技术**：卷积操作、池化层、LeNet、AlexNet、VGG
- **应用场景**：图像分类、目标检测、图像分割
- **代码实现**：
  - [卷积数学演示](chapter_04_cnn_classic/demo_conv_math.py)

#### [Chapter 05: ResNet与现代CNN](chapter_05_resnet_modern_cnn/) 🚧
- **核心内容**：残差连接、深度网络训练、现代CNN架构
- **关键技术**：ResNet、DenseNet、EfficientNet、MobileNet
- **应用场景**：深度模型训练、移动端部署、高效特征提取
- **状态**：规划中

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


## 🚀 使用方法

每个章节都是独立的，可以单独学习和运行。以下是已完成章节的使用示例：

### Chapter 01: 计算图与自动微分
```bash
cd chapter_01_tensor_autograd
# 查看自动微分示例
python autograd.py
# 生成计算图可视化
python graph_visualization.py
```

### Chapter 02: 万能逼近器（MLP）
```bash
cd chapter_02_nn_basics_mlp
# 运行MLP示例（三种模式：standard, 2d_surface, extrapolate）
python mlp.py --mode standard
# 运行2D表面拟合
python mlp.py --mode 2d_surface
# 运行外推实验
python mlp.py --mode extrapolate
```

### Chapter 03: 优化与正则化
```bash
cd chapter_03_optimization_regularization
# 比较不同优化器性能
python exp_optimization.py
# 比较不同正则化技术
python exp_regularization.py
# 比较不同标准化技术
python exp_normalization.py
```

### Chapter 04: 经典CNN
```bash
cd chapter_04_cnn_classic
# 运行卷积数学演示
python demo_conv_math.py
```

## 📊 项目进度

| 章节 | 状态 | 完成度 | 备注 |
|------|------|--------|------|
| Chapter 01: 计算图与自动微分 | ✅ 已完成 | 100% | 包含自动微分示例和图可视化 |
| Chapter 02: 万能逼近器（MLP） | ✅ 已完成 | 100% | 包含三种实验模式和预训练模型 |
| Chapter 03: 优化与正则化 | ✅ 已完成 | 100% | 包含优化器、正则化和标准化实验 |
| Chapter 04: 经典CNN | ✅ 已完成 | 100% | 包含卷积数学演示 |
| Chapter 05: ResNet与现代CNN | 🚧 规划中 | 0% | 待开发 |
| Chapter 06: RNN系列 | 🚧 规划中 | 0% | 待开发 |
| Chapter 07: 注意力机制 | 🚧 规划中 | 0% | 待开发 |
| Chapter 08: 原始Transformer | 🚧 规划中 | 0% | 待开发 |
| Chapter 09: 高效注意力 | 🚧 规划中 | 0% | 待开发 |
| Chapter 10: LLM现代组件 | 🚧 规划中 | 0% | 待开发 |

## 🤝 贡献指南

欢迎对本项目进行贡献！您可以通过以下方式参与：

1. **提交Issue**：报告bug或提出新功能建议
   - 发现代码错误或不一致之处
   - 建议新的实验或示例
   - 提出文档改进建议

2. **提交PR**：修复bug或添加新功能
   - 代码改进或优化
   - 添加新的实验或示例
   - 完善现有功能

3. **完善文档**：改进现有文档或添加新的教程
   - 修正理论解释中的错误
   - 添加更直观的解释或示例
   - 翻译文档到其他语言

4. **分享使用经验**：在Issues中分享您的学习心得或使用案例
   - 分享学习笔记或心得
   - 展示基于本项目的扩展应用
   - 提供教学反馈

### 贡献流程
1. Fork 本仓库
2. 创建您的特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启一个 Pull Request

## 📄 许可证

本项目采用MIT许可证，详情请见[LICENSE](LICENSE)文件。

## 🙏 致谢

感谢所有为深度学习发展做出贡献的研究者和工程师，特别感谢：

- PyTorch团队提供了优秀的深度学习框架
- 所有开源社区贡献者的无私分享
- 本项目的参考资源和教程作者

## 📞 联系方式

如有问题或建议，欢迎通过以下方式联系：
- Email: brenchchen.77@example.com
- GitHub Issues: https://github.com/BrenchCC/Deep-Learning-From-Scratch-PyTorch/issues

## 🌟 Star History

如果这个项目对您有帮助，请给它一个Star！

[![Star History Chart](https://api.star-history.com/svg?repos=BrenchCC/Deep-Learning-From-Scratch-PyTorch&type=Date)](https://star-history.com/#BrenchCC/Deep-Learning-From-Scratch-PyTorch&Date)

---

**Happy Learning! 🎉**