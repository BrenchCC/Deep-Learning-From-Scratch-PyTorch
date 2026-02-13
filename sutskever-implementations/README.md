# Sutskever Implementations

对 Ilya Sutskever 相关论文与推荐内容做小规模、可运行、可视化的机制复现。
目标不是追求 SOTA 指标，而是把核心思想变成可观察、可调参、可对比的实验。

## 当前结构
```ascii
sutskever-implementations/
├── README.md
├── 01_complexity_dynamics/
│   ├── README.md
│   ├── complexity_dynamics.py
│   ├── 01_complexity_dynamics.ipynb
│   └── images/
└── 02_char_rnn_karpathy/
    ├── README.md
    ├── 02_char_rnn_karpathy.py
    ├── 02_char_rnn_karpathy.ipynb
    ├── checkpoints/models/
    ├── images/
    └── results/
```

## 实验总览
| 编号 | 实验 | 核心主题 | 入口 |
|------|------|----------|------|
| 01 | The First Law of Complexodynamics | 元胞自动机复杂性增长、熵增、不可逆性 | `01_complexity_dynamics/README.md` |
| 02 | The Unreasonable Effectiveness of RNN | Character-level Vanilla RNN、BPTT、梯度裁剪、采样 | `02_char_rnn_karpathy/README.md` |

## 快速运行
建议在 Conda 环境中执行：

```bash
# Paper 1: Complexity Dynamics
conda run -n <ENV_NAME> python sutskever-implementations/01_complexity_dynamics/complexity_dynamics.py

# Paper 2: Char RNN (Karpathy)
conda run -n <ENV_NAME> python sutskever-implementations/02_char_rnn_karpathy/02_char_rnn_karpathy.py
```

## 输出产物说明
### 01_complexity_dynamics
- `images/rule_30_evolution.png`：元胞自动机演化图
- `images/entropy_complexity.png`：复杂度与熵趋势图
- `images/coffee_mixing_grid.png`：扩散过程网格图
- `images/coffee_mixing_entropy.png`：扩散熵随时间曲线

### 02_char_rnn_karpathy
- `images/training_loss.png`：训练损失曲线
- `images/hidden_state_activations.png`：隐藏状态激活热图
- `results/training_losses.json`：平滑损失序列
- `results/training_samples.json`：训练期采样文本
- `results/generated_samples.json`：最终生成样本
- `results/summary.json`：关键统计摘要
- `checkpoints/models/vanilla_rnn_weights.json`：模型参数快照

## 设计原则
- 机制优先：先理解“为什么有效”，再追求“更高性能”。
- 可复现优先：脚本参数清晰，产物结构固定。
- 可解释优先：尽量提供曲线、样本与中间状态可视化。

## 后续计划
- 增加更多 Sutskever 相关论文的最小实现
- 补充统一实验配置模板（参数、日志、结果汇总）
- 增加跨实验的对比文档（训练动力学、稳定性、可解释性）
