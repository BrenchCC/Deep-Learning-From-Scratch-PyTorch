# Deep-Learning-From-Scratch-PyTorch

> 从零开始的深度学习学习仓库：用 PyTorch 贯通自动微分、MLP、CNN、ResNet、RNN/LSTM 与 Attention，并配套可运行脚本与可视化产物。

## TL;DR（3分钟上手）

```bash
git clone https://github.com/BrenchCC/Deep-Learning-From-Scratch-PyTorch.git
cd Deep-Learning-From-Scratch-PyTorch

pip install -r requirements.txt

python chapter_01_tensor_autograd/autograd.py
```

成功判定标准：
- 命令可正常结束，无报错退出。
- 可在终端看到自动微分相关输出日志。
- 仓库中的 `chapter_01_tensor_autograd/four_layer_autograd_graph.png` 可作为本章配套图示参考。

## 项目适合谁

- 想系统理解深度学习核心机制（不仅调用 API）。
- 想从“公式/概念”过渡到“可运行 PyTorch 代码”。
- 想按章节渐进学习并保留实验产物用于复盘。

## 学习路径（推荐顺序）

新手建议节奏：
- 先完成 `01 -> 02` 打基础（自动微分 + MLP）。
- 然后按兴趣分支：
  - 视觉方向：`03 -> 04 -> 05`
  - 序列方向：`03 -> 06 -> 07 -> 08 -> 09`

| 阶段 | 章节 | 你会学到什么 | 入口命令 | 预计耗时 |
|---|---|---|---|---|
| 基础 1 | [Chapter 01: Tensor & Autograd](chapter_01_tensor_autograd/README.md) | 计算图、链式法则、反向传播 | `python chapter_01_tensor_autograd/autograd.py` | 30-60 分钟 |
| 基础 2 | [Chapter 02: MLP Basics](chapter_02_nn_basics_mlp/README.md) | 多层感知机、激活函数、损失函数 | `python chapter_02_nn_basics_mlp/mlp.py --input_dim 1 --mode standard` | 45-90 分钟 |
| 共同前置 | [Chapter 03: Optimization & Regularization](chapter_03_optimization_regularization/README.md) | SGD/Adam、归一化、正则化 | `python chapter_03_optimization_regularization/exp_optimization.py` | 45-90 分钟 |
| 视觉分支 1 | [Chapter 04: Classic CNN](chapter_04_cnn_classic/README.md) | 卷积网络训练与推理 | `python chapter_04_cnn_classic/train.py --epochs 1 --batch_size 64` | 60-120 分钟 |
| 视觉分支 2 | [Chapter 05: ResNet](chapter_05_resnet_modern_cnn/README.md) | 残差连接与现代 CNN 训练 | `python chapter_05_resnet_modern_cnn/src/model_train.py --mode compare --epochs 1 --batch_size 64` | 60-120 分钟 |
| 序列分支 1 | [Chapter 06: RNN/LSTM/GRU](chapter_06_rnn_lstm_seq/README.md) | 序列建模与门控机制 | `python chapter_06_rnn_lstm_seq/main.py --model_type lstm --epochs 1 --data_size 2000` | 60-120 分钟 |
| 序列分支 2 | [Chapter 07: Attention Mechanism](chapter_07_attention_mechanism/README.md) | 注意力机制核心与掩码复制任务 | `python chapter_07_attention_mechanism/train.py --epochs 1 --num_samples 2000` | 45-90 分钟 |
| 序列分支 3 | [Chapter 08: Vanilla Transformer](chapter_08_transformer_vanilla/README.md) | 手搓标准 Transformer（排序+翻译） | `python chapter_08_transformer_vanilla/train.py --task sort --epochs 1 --num_samples 2000` | 60-120 分钟 |
| 序列分支 4 | [Chapter 09: Efficient Attention](chapter_09_efficient_attention/README.md) | MQA/GQA/MLA 与 KV Cache 优化 | `python chapter_09_efficient_attention/demo.py` | 60-120 分钟 |

## 快速运行导航（按目标）

### 我只想先看自动微分

```bash
python chapter_01_tensor_autograd/autograd.py
```

你会看到什么：
- 计算图与梯度传播的示例输出。

### 我只想快速跑通 CNN

```bash
python chapter_04_cnn_classic/train.py --epochs 1 --batch_size 64
```

你会看到什么：
- 1 个 epoch 的训练日志与基础结果产物（见章节目录下 `results/`）。

### 我只想跑 ResNet 对比实验

```bash
python chapter_05_resnet_modern_cnn/src/model_train.py --mode compare --epochs 1 --batch_size 64
```

你会看到什么：
- 残差结构相关训练对比过程，结果写入章节目录下 `results/` 与 `checkpoints/`。

### 我只想体验序列模型（LSTM）

```bash
python chapter_06_rnn_lstm_seq/main.py --model_type lstm --epochs 1 --data_size 2000
```

你会看到什么：
- 序列任务的训练日志与结果文件（见 `chapter_06_rnn_lstm_seq/results/`）。

### 我只想跑注意力机制最小实验

```bash
python chapter_07_attention_mechanism/demo.py
python chapter_07_attention_mechanism/train.py --epochs 1 --num_samples 2000
```

你会看到什么：
- `chapter_07_attention_mechanism/results/` 下的 `attention_demo.json`、`metrics.json`、`predictions.json` 等产物，以及 `checkpoints/` 模型文件。

### 我想直接跑简化版 Transformer

```bash
python chapter_08_transformer_vanilla/train.py --task sort --epochs 1 --num_samples 2000
python chapter_08_transformer_vanilla/train.py --task translate --epochs 1
```

你会看到什么：
- `chapter_08_transformer_vanilla/results/` 下的 `sort_metrics.json` 或 `translate_metrics.json`，以及对应预测文件和 checkpoint。

### 我想比较高效注意力变体（MQA/GQA/MLA）

```bash
python chapter_09_efficient_attention/demo.py
python chapter_09_efficient_attention/train.py --variant all --epochs 1 --num_samples 2000
```

你会看到什么：
- `chapter_09_efficient_attention/results/` 下的 `attention_compare.json`、`metrics_*.json`、`predictions_*.json`，以及 `checkpoints/` 模型文件。

## 章节总览

| 章节 | 核心主题 | 关键脚本 | 输入数据 | 输出产物 | 进阶阅读 |
|---|---|---|---|---|---|
| [01](chapter_01_tensor_autograd/README.md) | 自动微分与计算图 | `chapter_01_tensor_autograd/autograd.py` | 代码内构造示例 | 图示与日志 | [CODE_LOGIC_README](chapter_01_tensor_autograd/CODE_LOGIC_README.md) |
| [02](chapter_02_nn_basics_mlp/README.md) | MLP 与万能逼近 | `chapter_02_nn_basics_mlp/mlp.py` | `chapter_02_nn_basics_mlp/data/` | `images/`、训练结果 | [CODE_LOGIC_README](chapter_02_nn_basics_mlp/CODE_LOGIC_README.md) |
| [03](chapter_03_optimization_regularization/README.md) | 优化器与正则化 | `exp_optimization.py` / `exp_regularization.py` | 实验配置与合成数据 | `images/` | [CODE_LOGIC_README](chapter_03_optimization_regularization/CODE_LOGIC_README.md) |
| [04](chapter_04_cnn_classic/README.md) | 经典 CNN | `chapter_04_cnn_classic/train.py` | `chapter_04_cnn_classic/data/` | `results/`、可视化图 | [CODE_LOGIC_README](chapter_04_cnn_classic/CODE_LOGIC_README.md) |
| [05](chapter_05_resnet_modern_cnn/README.md) | ResNet 与现代 CNN | `src/model_train.py` | `data/` + `images/` | `results/`、`checkpoints/` | [CODE_LOGIC_README](chapter_05_resnet_modern_cnn/CODE_LOGIC_README.md) |
| [06](chapter_06_rnn_lstm_seq/README.md) | RNN/LSTM/GRU | `chapter_06_rnn_lstm_seq/main.py` | `chapter_06_rnn_lstm_seq/data/` | `results/`、`checkpoints/` | [CODE_LOGIC_README](chapter_06_rnn_lstm_seq/CODE_LOGIC_README.md) |
| [07](chapter_07_attention_mechanism/README.md) | Attention Mechanism Core | `chapter_07_attention_mechanism/demo.py` / `train.py` | 随机向量 + masked copy 数据 | `results/`、`checkpoints/` | [CODE_LOGIC_README](chapter_07_attention_mechanism/CODE_LOGIC_README.md) |
| [08](chapter_08_transformer_vanilla/README.md) | Vanilla Transformer | `chapter_08_transformer_vanilla/train.py` | sort 合成数据 + toy 翻译对 | `results/`、`checkpoints/` | [CODE_LOGIC_README](chapter_08_transformer_vanilla/CODE_LOGIC_README.md) |
| [09](chapter_09_efficient_attention/README.md) | Efficient Attention Variants | `chapter_09_efficient_attention/demo.py` / `train.py` | 合成 hidden states + toy next-token 数据 | `results/`、`checkpoints/` | [CODE_LOGIC_README](chapter_09_efficient_attention/CODE_LOGIC_README.md) |
| [Sutskever 实验区](sutskever-implementations/README.md) | 论文机制复现 | 各子目录入口脚本 | 各实验自带数据 | `images/`、`results/`、`checkpoints/` | [总览 README](sutskever-implementations/README.md) |

## 术语跳转索引

- 自动微分 / 计算图：见 [Chapter 01](chapter_01_tensor_autograd/README.md)
- MLP / 激活函数 / 万能逼近：见 [Chapter 02](chapter_02_nn_basics_mlp/README.md)
- 优化器 / 正则化 / 归一化：见 [Chapter 03](chapter_03_optimization_regularization/README.md)
- 卷积神经网络（CNN）：见 [Chapter 04](chapter_04_cnn_classic/README.md)
- 残差网络（ResNet）：见 [Chapter 05](chapter_05_resnet_modern_cnn/README.md)
- RNN / LSTM / GRU：见 [Chapter 06](chapter_06_rnn_lstm_seq/README.md)
- Attention Mechanism Core：见 [Chapter 07](chapter_07_attention_mechanism/README.md)
- Vanilla Transformer：见 [Chapter 08](chapter_08_transformer_vanilla/README.md)
- Efficient Attention Variants：见 [Chapter 09](chapter_09_efficient_attention/README.md)

## 仓库地图（精简）

```ascii
Deep-Learning-From-Scratch-PyTorch/
├── README.md                               # 项目总览与学习入口
├── requirements.txt                        # 依赖列表
├── utils/                                  # 通用工具（日志、设备、计时、IO）
├── chapter_01_tensor_autograd/             # 自动微分与计算图
├── chapter_02_nn_basics_mlp/               # MLP 基础
├── chapter_03_optimization_regularization/ # 优化与正则化
├── chapter_04_cnn_classic/                 # 经典 CNN
├── chapter_05_resnet_modern_cnn/           # ResNet 与现代 CNN
├── chapter_06_rnn_lstm_seq/                # 序列模型
├── chapter_07_attention_mechanism/         # 注意力机制核心
├── chapter_08_transformer_vanilla/         # 手搓标准 Transformer
├── chapter_09_efficient_attention/          # 高效注意力变体（MQA/GQA/MLA）
└── sutskever-implementations/              # 论文机制复现实验区
```

## 环境与依赖

```bash
pip install -r requirements.txt
```

建议：
- Python 版本建议与 PyTorch 常用版本保持兼容（建议 3.9+）。
- 首次运行训练脚本可使用较小参数（如 `--epochs 1`）先做冒烟验证。

## 常见问题（FAQ / Troubleshooting）

### Q1. 依赖安装失败（如 `ModuleNotFoundError`）

- 现象：运行脚本时报缺少包。
- 原因：依赖未完整安装或安装到错误环境。
- 一步解决命令：

```bash
pip install -r requirements.txt
```

### Q2. 脚本路径错误（如 `No such file or directory`）

- 现象：执行命令后提示找不到脚本或资源。
- 原因：不在仓库根目录执行命令。
- 一步解决命令：

```bash
cd Deep-Learning-From-Scratch-PyTorch
```

然后再运行对应 `python chapter_xxx/...py` 命令。

### Q3. 训练很慢，想先验证流程

- 现象：一次完整训练耗时较长。
- 原因：默认参数针对学习完整流程，不是最短时间配置。
- 一步解决命令（示例）：

```bash
python chapter_04_cnn_classic/train.py --epochs 1 --batch_size 64
```

### Q4. 显存不足（CUDA OOM）

- 现象：训练中出现 CUDA out of memory。
- 原因：batch size 过大或模型配置偏大。
- 一步解决命令（示例）：

```bash
python chapter_05_resnet_modern_cnn/src/model_train.py --mode compare --epochs 1 --batch_size 32
```

### Q5. 找不到结果图或模型文件

- 现象：运行后不确定产物保存位置。
- 原因：各章节将产物写到各自目录，而不是仓库统一目录。
- 一步定位命令：

```bash
find chapter_* sutskever-implementations -type d \( -name results -o -name images -o -name checkpoints \)
```

## English Summary

This project is a from-scratch PyTorch learning repository for core deep learning topics:
- Autograd and computation graph
- MLP fundamentals
- Optimization and regularization
- CNN and ResNet
- RNN/LSTM/GRU
- Attention mechanism core
- Vanilla transformer (sorting + toy translation)

Quick start:

```bash
pip install -r requirements.txt
python chapter_01_tensor_autograd/autograd.py
```

Recommended path:
1. Chapter 01 -> Chapter 02
2. Then choose either vision track (03 -> 04 -> 05) or sequence track (03 -> 06 -> 07 -> 08)

For paper-inspired standalone experiments, see:
- [sutskever-implementations/README.md](sutskever-implementations/README.md)

## 贡献指南

1. `git checkout -b feature/YourFeature`
2. 开发与自测
3. `git commit -m "feat: your change"`
4. `git push origin feature/YourFeature`
5. 提交 PR（附运行命令与关键结果）

## 许可证

Apache License 2.0，详见 [LICENSE](LICENSE)。

## 联系方式

- Email: brenchchen.77@example.com
- Issues: <https://github.com/BrenchCC/Deep-Learning-From-Scratch-PyTorch/issues>
