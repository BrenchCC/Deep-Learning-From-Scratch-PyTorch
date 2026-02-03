# Deep-Learning-From-Scratch-PyTorch
> 从零开始的深度学习教程：用 PyTorch 复现核心原理与现代方法（MLP / CNN / ResNet / RNN / Attention / Transformer / LLM）

## 项目亮点
- 从零实现关键组件：自动微分、优化器、CNN/ResNet、RNN 等
- 理论 + 代码 + 可视化：可运行脚本与图表并重
- 渐进式学习路径：基础到前沿逐层递进
- 提供部分预训练模型与对比实验

## 目录结构
```ascii
Deep-Learning-From-Scratch-PyTorch/
├── README.md
├── requirements.txt
├── utils/
├── chapter_01_tensor_autograd/
├── chapter_02_nn_basics_mlp/
├── chapter_03_optimization_regularization/
├── chapter_04_cnn_classic/
├── chapter_05_resnet_modern_cnn/
├── chapter_06_rnn_lstm_seq/
├── chapter_07_attention_mechanism/
├── chapter_08_transformer_vanilla/
├── chapter_09_efficient_attention/
└── chapter_10_llm_modern_components/
```

## 快速开始 🚀
```bash
# 1) 克隆项目
git clone https://github.com/BrenchCC/Deep-Learning-From-Scratch-PyTorch.git
cd Deep-Learning-From-Scratch-PyTorch

# 2) 安装依赖
pip install -r requirements.txt

# 3) 运行第一个示例
python chapter_01_tensor_autograd/autograd.py
```

## 小贴士 💡
> [!NOTE]
> 建议按章节顺序学习，避免知识断层  
> 训练类脚本耗时较长，可先用较小 `epochs` 快速验证  
> 可先跑示例脚本，确认环境无误后再进行完整实验

## 章节概览
| 章节 | 状态 | 重点 | 入口示例 |
|------|------|------|----------|
| 01 自动微分 | ✅ | 计算图 / VJP | `chapter_01_tensor_autograd/autograd.py` |
| 02 MLP | ✅ | 万能逼近 | `chapter_02_nn_basics_mlp/mlp.py` |
| 03 优化与正则化 | ✅ | 优化器 / 正则化 | `chapter_03_optimization_regularization/exp_optimization.py` |
| 04 经典 CNN | ✅ | 卷积 / 可视化 | `chapter_04_cnn_classic/train.py` |
| 05 ResNet | ✅ | 残差连接 | `chapter_05_resnet_modern_cnn/src/model_train.py` |
| 06 RNN/LSTM | ✅ | 序列建模 | `chapter_06_rnn_lstm_seq/main.py` |
| 07 注意力 | 🚧 | 注意力机制 | `chapter_07_attention_mechanism/` |
| 08 Transformer | 🚧 | 编码器-解码器 | `chapter_08_transformer_vanilla/` |
| 09 高效注意力 | 🚧 | 线性/稀疏注意力 | `chapter_09_efficient_attention/` |
| 10 LLM 组件 | 🚧 | 现代 LLM 关键组件 | `chapter_10_llm_modern_components/` |

## 章节运行命令 🧪
### Chapter 01: 计算图与自动微分
```bash
# 自动微分示例
python chapter_01_tensor_autograd/autograd.py

# 计算图可视化
python chapter_01_tensor_autograd/graph_visualization.py
```
> [!NOTE]
> 图可视化脚本可用于理解梯度传播路径

### Chapter 02: 万能逼近器（MLP）
```bash
# 标准拟合
python chapter_02_nn_basics_mlp/mlp.py --mode standard

# 2D 表面拟合
python chapter_02_nn_basics_mlp/mlp.py --mode 2d_surface

# 外推实验
python chapter_02_nn_basics_mlp/mlp.py --mode extrapolate
```
> [!NOTE]
> `extrapolate` 模式能直观看到模型泛化能力

### Chapter 03: 优化与正则化
```bash
# 优化器对比
python chapter_03_optimization_regularization/exp_optimization.py

# 正则化对比
python chapter_03_optimization_regularization/exp_regularization.py

# 标准化对比
python chapter_03_optimization_regularization/exp_normalization.py
```
> [!NOTE]
> 建议先跑 `exp_optimization.py` 获取直观对比曲线

### Chapter 04: 经典 CNN
```bash
# 卷积数学演示
python chapter_04_cnn_classic/demo_conv_math.py

# CIFAR-10 训练
python chapter_04_cnn_classic/train.py --epochs 100 --batch_size 128 --lr 0.001

# 推理与可视化
python chapter_04_cnn_classic/inference.py --img_dir ./chapter_04_cnn_classic/data/custom_imgs --model_path ./chapter_04_cnn_classic/results/best_model.pth
```
> [!NOTE]
> 推理脚本支持自定义图片目录，便于快速验证

### Chapter 05: ResNet 与现代 CNN
```bash
# ResNet-18 训练（STL-10）
python chapter_05_resnet_modern_cnn/src/model_train.py --epochs 100 --batch_size 64 --lr 0.1

# PlainNet-18 对比训练
python chapter_05_resnet_modern_cnn/src/model_train.py --model_type plainnet --epochs 100 --batch_size 64 --lr 0.1

# 推理
python chapter_05_resnet_modern_cnn/src/inference.py --model_path ./chapter_05_resnet_modern_cnn/checkpoints/resnet18_stl10.pth --image_path ./chapter_05_resnet_modern_cnn/images/airplane.png

# Grad-CAM 可视化
python chapter_05_resnet_modern_cnn/src/cam.py --model_path ./chapter_05_resnet_modern_cnn/checkpoints/resnet18_stl10.pth --image_path ./chapter_05_resnet_modern_cnn/images/airplane.png
```
> [!NOTE]
> Grad-CAM 能直观看到模型关注区域

### Chapter 06: RNN/LSTM
```bash
# 维度推演示例
python chapter_06_rnn_lstm_seq/examples/easy_demo.py

# 情感分类示例
python chapter_06_rnn_lstm_seq/examples/sentime_lstm_demo.py

# 完整训练流程
python chapter_06_rnn_lstm_seq/main.py --epochs 50 --batch_size 32 --lr 0.001
```
> [!NOTE]
> 先运行 `easy_demo.py` 熟悉维度变化再训练完整模型

### Chapter 07-10: 注意力 / Transformer / 高效注意力 / LLM 组件
```bash
# 开发中：代码结构已就绪，后续会补齐可运行脚本
```

## 贡献指南 🤝
1. `git checkout -b feature/YourFeature`
2. 开发与自测
3. `git commit -m 'Add: your feature'`
4. `git push origin feature/YourFeature`
5. 提交 PR

## 许可证 📄
Apache License 2.0，详见 `LICENSE`。

## 联系方式 📫
- Email: brenchchen.77@example.com
- Issues: https://github.com/BrenchCC/Deep-Learning-From-Scratch-PyTorch/issues
