# Sutskever Implementations

对 Ilya Sutskever 相关论文与经典讲解内容做小规模、可运行、可视化的机制复现。
本目录重点不是 SOTA 指标，而是把核心思想变成可观察、可调参、可对比的工程实验。

## 学习路径 (Recommended Path)
建议按以下顺序学习：
1. `01_complexity_dynamics`：先建立“复杂性增长与熵增”的系统直觉。
2. `02_char_rnn_karpathy`：再看循环网络如何在时间上积累上下文信息。
3. `03_lstm_understanding`：最后理解门控记忆如何缓解长依赖与梯度衰减。

章节关系图式（从系统到模型）：
`复杂系统直觉 -> 序列建模机制 -> 长程记忆结构`

## 当前结构
```ascii
sutskever-implementations/
├── README.md
├── 01_complexity_dynamics/
│   ├── README.md
│   ├── complexity_dynamics.py
│   ├── 01_complexity_dynamics.ipynb
│   └── images/
├── 02_char_rnn_karpathy/
│   ├── README.md
│   ├── 02_char_rnn_karpathy.py
│   ├── 02_char_rnn_karpathy.ipynb
│   ├── checkpoints/models/
│   ├── images/
│   └── results/
└── 03_lstm_understanding/
    ├── README.md
    ├── 03_lstm_understanding.py
    ├── 03_lstm_understanding.ipynb
    ├── checkpoints/models/
    ├── images/
    └── results/
```

## 实验总览
| 编号 | 实验 | 核心主题 | 入口 |
|---|---|---|---|
| 01 | The First Law of Complexodynamics | 元胞自动机复杂性增长、熵增、不可逆混合 | `01_complexity_dynamics/README.md` |
| 02 | The Unreasonable Effectiveness of RNN | Character-level Vanilla RNN、BPTT、梯度裁剪、Adagrad | `02_char_rnn_karpathy/README.md` |
| 03 | Understanding LSTM Networks | 门控记忆、状态可视化、梯度流对比 | `03_lstm_understanding/README.md` |

## 每章最值得先看的结果
1. 01 章：`01_complexity_dynamics/images/entropy_complexity.png`（熵与复杂度联合趋势）。
2. 02 章：`02_char_rnn_karpathy/images/training_loss.png` + `results/training_samples.json`（损失与样本同步演化）。
3. 03 章：`03_lstm_understanding/images/gradient_flow_comparison.png`（RNN vs LSTM 梯度衰减差异）。

## 快速运行 (Conda)
建议在 Conda 环境中执行：

```bash
# 01: Complexity Dynamics
python sutskever-implementations/01_complexity_dynamics/complexity_dynamics.py

# 02: Char RNN (Karpathy)
python sutskever-implementations/02_char_rnn_karpathy/02_char_rnn_karpathy.py

# 03: LSTM Understanding
python sutskever-implementations/03_lstm_understanding/03_lstm_understanding.py
```

## 产物索引 (Artifacts)

### 01_complexity_dynamics
- 图像：
  - `01_complexity_dynamics/images/rule_30_evolution.png`
  - `01_complexity_dynamics/images/entropy_complexity.png`
  - `01_complexity_dynamics/images/coffee_mixing_grid.png`
  - `01_complexity_dynamics/images/coffee_mixing_entropy.png`

### 02_char_rnn_karpathy
- 图像：
  - `02_char_rnn_karpathy/images/training_loss.png`
  - `02_char_rnn_karpathy/images/hidden_state_activations.png`
- 结果：
  - `02_char_rnn_karpathy/results/training_losses.json`
  - `02_char_rnn_karpathy/results/training_samples.json`
  - `02_char_rnn_karpathy/results/generated_samples.json`
  - `02_char_rnn_karpathy/results/summary.json`
- 参数快照：
  - `02_char_rnn_karpathy/checkpoints/models/vanilla_rnn_weights.json`

### 03_lstm_understanding
- 图像：
  - `03_lstm_understanding/images/lstm_gate_visualization.png`
  - `03_lstm_understanding/images/lstm_vs_vanilla_rnn_states.png`
  - `03_lstm_understanding/images/gradient_flow_comparison.png`
- 结果：
  - `03_lstm_understanding/results/lstm_understanding_summary.json`
- 参数快照：
  - `03_lstm_understanding/checkpoints/models/lstm_init_weights.npz`
  - `03_lstm_understanding/checkpoints/models/vanilla_rnn_init_weights.npz`

## 设计原则
1. 机制优先：先理解“为什么有效”，再追求“更高性能”。
2. 可复现优先：脚本参数明确，输出路径固定。
3. 可解释优先：尽量提供曲线、样本和中间状态可视化。

## 后续扩展方向
1. 增加更多 Sutskever 相关论文的最小实现。
2. 统一实验配置模板（参数、日志、结果摘要）。
3. 增加跨实验对比文档（训练动力学、稳定性、可解释性）。
