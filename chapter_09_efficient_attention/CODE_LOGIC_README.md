# Chapter 09 Code Logic README

## 1. 训练主流程
`train.py` 使用统一入口支持四变体单跑或全跑：
1. `parse_args()` 解析配置。
2. 构建 `ToyNextTokenDataset` 和 dataloader。
3. 根据 `--variant` 选择 `mha/mqa/gqa/mla`。
4. 训练与验证循环，保存每个变体最佳 checkpoint。
5. 导出 `metrics_*.json`、`predictions_*.json`，并汇总 `attention_compare.json`。

---

## 2. Mermaid 图 1：训练流程总览
```mermaid
flowchart TD
    A[parse_args] --> B[build toy dataset]
    B --> C[train/val DataLoader]
    C --> D{variant all?}
    D -- yes --> E[loop mha mqa gqa mla]
    D -- no --> F[single variant]
    E --> G[build EfficientAttentionLM]
    F --> G
    G --> H[run_epoch train]
    H --> I[run_epoch eval]
    I --> J[save best checkpoint]
    J --> K[collect predictions]
    K --> L[save metrics and predictions]
    L --> M[save attention_compare.json]
```

---

## 3. Mermaid 图 2：四变体前向数据流
```mermaid
flowchart LR
    X[hidden_states BxSxD] --> Q[query projection]

    X --> K1[MHA: key projection per head]
    X --> K2[MQA: shared single key/value]
    X --> K3[GQA: grouped key/value]
    X --> K4[MLA: low-rank compress -> expand]

    Q --> A1[scaled dot-product attention]
    K1 --> A1

    Q --> A2[scaled dot-product attention]
    K2 --> A2

    Q --> A3[scaled dot-product attention]
    K3 --> A3

    Q --> A4[scaled dot-product attention]
    K4 --> A4

    A1 --> O1[output projection]
    A2 --> O2[output projection]
    A3 --> O3[output projection]
    A4 --> O4[output projection]
```

---

## 4. 文件角色表
| 文件 | 角色 | 重点 |
|---|---|---|
| `common.py` | 公共配置与复杂度工具 | `AttentionConfig`, `variant_kv_channels` |
| `mha.py` | 标准 MHA | 独立 Q/K/V 多头 |
| `mqa.py` | MQA | Query 多头 + K/V 共享 |
| `gqa.py` | GQA | 分组共享 K/V |
| `mla.py` | MLA | `compress -> expand` 低秩路径 |
| `dataset.py` | 数据构造 | `ToyNextTokenDataset` |
| `model.py` | 模型壳与变体工厂 | `build_attention_block`, `EfficientAttentionLM` |
| `demo.py` | 机制对比 | 输出统计、相似度、缓存对比 |
| `benchmark.py` | 性能基准 | 延迟与缓存曲线 |
| `train.py` | 训练与落盘 | `--variant all` 聚合对比 |

---

## 5. 最小验收命令
```bash
python chapter_09_efficient_attention/demo.py
python chapter_09_efficient_attention/benchmark.py --seq_lens 128,256,512
python chapter_09_efficient_attention/train.py --variant all --epochs 1 --num_samples 2000
```

验收标准：
1. `results/attention_compare.json` 存在且含四变体条目。
2. `results/metrics_*.json` 与 `results/predictions_*.json` 四组文件存在。
3. `checkpoints/*_best.pth` 四个文件存在。
4. `images/` 目录下有 demo 与 benchmark 图像文件。
