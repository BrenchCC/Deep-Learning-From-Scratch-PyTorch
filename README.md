# Deep-Learning-From-Scratch-PyTorch
> 建立一个面向回顾与教学的 GitHub 项目，从传统神经网络开始，逐步覆盖 CNN/ResNet、RNN 系列、各种注意力机制、到手写（从零实现）Transformer。每个目录包含：理论速览（README）、PyTorch 演示代码（模块化）、可运行训练脚本、示例数据／数据加载

## 项目目录结构
```ascii
Deep-Learning-From-Scratch-PyTorch/
├── README.md
├── requirements.txt
├── utils/                  # 通用工具箱
│   ├── __init__.py
│   ├── device.py           # TorchDevice工具
│   ├── file_io_util.py     # 文件IO工具
│   ├── model_summary.py    # 模型参数查看工具
│   ├── seed.py             # 随机种子固定工具
│   └── timer.py             # 时间测量工具
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