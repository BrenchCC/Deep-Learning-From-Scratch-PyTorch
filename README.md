# Deep-Learning-From-Scratch-PyTorch
> 从零开始的深度学习教程：用 PyTorch 复现核心原理与现代方法（MLP / CNN / ResNet / RNN / Attention）

## 项目亮点
- 从零实现关键组件：自动微分、优化器、CNN/ResNet、RNN 等
- 理论 + 代码 + 可视化：可运行脚本与图表并重
- 渐进式学习路径：基础到序列建模逐层递进
- 包含 `sutskever-implementations/` 特别实验区，用于复现经典论文机制

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
└── sutskever-implementations/
```

## 快速开始
```bash
# 1) 克隆项目
git clone https://github.com/BrenchCC/Deep-Learning-From-Scratch-PyTorch.git
cd Deep-Learning-From-Scratch-PyTorch

# 2) 安装依赖
pip install -r requirements.txt

# 3) 运行第一个示例
python chapter_01_tensor_autograd/autograd.py
```

## 小贴士
> [!NOTE]
> 建议按章节顺序学习，避免知识断层。训练类脚本耗时较长，可先用较小 `epochs` 快速验证。

## 章节概览
| 章节 | 状态 | 重点 | 入口示例 |
|------|------|------|----------|
| 01 自动微分 | ✅ | 计算图 / VJP | `chapter_01_tensor_autograd/autograd.py` |
| 02 MLP | ✅ | 万能逼近 | `chapter_02_nn_basics_mlp/mlp.py` |
| 03 优化与正则化 | ✅ | 优化器 / 正则化 | `chapter_03_optimization_regularization/exp_optimization.py` |
| 04 经典 CNN | ✅ | 卷积 / 可视化 | `chapter_04_cnn_classic/train.py` |
| 05 ResNet | ✅ | 残差连接 | `chapter_05_resnet_modern_cnn/src/model_train.py` |
| 06 RNN/LSTM | ✅ | 序列建模 | `chapter_06_rnn_lstm_seq/main.py` |
| 07 注意力机制 | 🚧 | 注意力基础 | `chapter_07_attention_mechanism/` |

## Sutskever Implementations
`sutskever-implementations/` 是论文机制复现专区，当前包含以下内容：

| 实验 | 状态 | 主题 | 入口 |
|------|------|------|------|
| 01 Complexity Dynamics | ✅ | 元胞自动机复杂性增长 / 熵增 / 不可逆性 | `sutskever-implementations/01_complexity_dynamics/README.md` |
| 02 Char RNN (Karpathy) | ✅ | Vanilla RNN 字符级建模 / BPTT / 梯度裁剪 | `sutskever-implementations/02_char_rnn_karpathy/README.md` |
| 03 Understanding LSTM | ✅ | 门控记忆 / 状态可视化 / 梯度流对比 | `sutskever-implementations/03_lstm_understanding/README.md` |

运行示例：
```bash
# 01 复杂动力学
python sutskever-implementations/01_complexity_dynamics/complexity_dynamics.py

# 02 字符级 RNN
python sutskever-implementations/02_char_rnn_karpathy/02_char_rnn_karpathy.py

# 03 LSTM 机制理解
python sutskever-implementations/03_lstm_understanding/03_lstm_understanding.py
```

更多说明见：`sutskever-implementations/README.md`

## 贡献指南
1. `git checkout -b feature/YourFeature`
2. 开发与自测
3. `git commit -m 'Add: your feature'`
4. `git push origin feature/YourFeature`
5. 提交 PR

## 许可证
Apache License 2.0，详见 `LICENSE`。

## 联系方式
- Email: brenchchen.77@example.com
- Issues: https://github.com/BrenchCC/Deep-Learning-From-Scratch-PyTorch/issues
