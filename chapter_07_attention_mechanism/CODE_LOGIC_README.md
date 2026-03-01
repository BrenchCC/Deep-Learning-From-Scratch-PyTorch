# Chapter 07 Code Logic README

## 1. 总体流程
`train.py` 的主流程分为 6 步：
1. 解析参数并设置随机种子。
2. 构建 `ToyCopyDataset` 与 `DataLoader`。
3. 初始化 `Seq2SeqTransformer`。
4. 先执行 shape/mask 检查，确保结构正确。
5. 进入训练与验证循环，记录指标并保存最佳权重。
6. 导出 `metrics.json`、`run_config.json`、`predictions.json`。

---

## 2. 数据流（dataset.py）

### 2.1 `ToyCopyDataset`
- 每个样本是一段随机 token 序列（不含 BOS/EOS）。
- token 范围固定在 `[3, vocab_size - 1]`，避免与特殊 token 冲突。

### 2.2 `ToyCopyCollator`
对 batch 内每个样本构造：
- `src = tokens + [EOS]`
- `tgt_input = [BOS] + tokens`
- `tgt_output = tokens + [EOS]`

然后按 batch 最大长度做右侧 padding。

---

## 3. 模型流（已拆分为多文件）

当前结构如下：
- `masks.py`：mask 构造函数
- `attention.py`：`ScaledDotProductAttention` 与 `MultiHeadAttention`
- `positional_encoding.py`：`SinusoidalPositionalEncoding`
- `feed_forward.py`：`PositionwiseFeedForward`
- `encoder.py`：`TransformerEncoderLayer`
- `decoder.py`：`TransformerDecoderLayer`
- `transformer.py`：`Seq2SeqTransformer`
- `model.py`：兼容导出层（保持旧导入路径）

### 3.1 Mask 工具
- `build_padding_mask(tokens, pad_token_id)`
  - 输出 `[batch, 1, 1, seq_len]`
  - `True` 表示该 key 位置应被屏蔽。
- `build_causal_mask(seq_len, device)`
  - 输出 `[1, 1, seq_len, seq_len]`
  - 上三角（未来位置）为 `True`。

### 3.2 注意力模块
- `ScaledDotProductAttention`
  - 计算 `scores = QK^T / sqrt(d_k)`
  - 对 mask 位置做极小值填充
  - softmax 后与 `V` 相乘得到上下文向量

- `MultiHeadAttention`
  - 线性投影得到 Q/K/V
  - 拆头 `[batch, heads, seq_len, head_dim]`
  - 调用 `ScaledDotProductAttention`
  - 合并 heads 并输出最终投影

### 3.3 Transformer Block
- `TransformerEncoderLayer`
  - Self-Attention -> Residual+LayerNorm
  - FFN -> Residual+LayerNorm

- `TransformerDecoderLayer`
  - Masked Self-Attention -> Residual+LayerNorm
  - Cross-Attention -> Residual+LayerNorm
  - FFN -> Residual+LayerNorm

### 3.4 整体模型 `Seq2SeqTransformer`
- `forward(src_tokens, tgt_input_tokens, return_attn = False)`
  1. 构建 `src_mask`、`tgt_padding_mask`、`tgt_causal_mask`
  2. `tgt_mask = tgt_padding_mask | tgt_causal_mask`
  3. Encoder 编码 source
  4. Decoder 基于 `tgt_input` 与 memory 解码
  5. 线性层映射到词表 logits

- `greedy_decode`
  - 从 BOS 开始逐步生成，直到达到 EOS 或 `max_len`。

---

## 4. 训练细节（train.py）

### 4.1 `run_shape_and_mask_checks`
在正式训练前执行，检查：
- logits 形状是否符合 `[batch, tgt_len, vocab_size]`
- padding/causal mask 的 dtype 和逻辑
- attention map 数量是否与层数一致

### 4.2 `run_copy_epoch`
- `stage = train`：前向、反向、梯度裁剪、参数更新
- `stage = eval`：仅前向统计
- loss 与 token_acc 都按非 padding token 聚合

### 4.3 模型保存策略
- 以验证集 `val_loss` 最优为准保存：
  - `checkpoints/transformer_copy_best.pth`

### 4.4 结果导出
- `metrics.json`：每个 epoch 的 train/val loss 与 token_acc
- `run_config.json`：完整参数快照 + best checkpoint 信息
- `predictions.json`：固定样本的 greedy decode 对比

---

## 5. 关键可调参数
- 结构：`d_model`, `num_heads`, `num_encoder_layers`, `num_decoder_layers`, `ffn_hidden_dim`
- 数据：`num_samples`, `vocab_size`, `min_seq_len`, `max_seq_len`
- 训练：`epochs`, `batch_size`, `lr`, `weight_decay`, `max_grad_norm`

---

## 6. 最小验收命令
```bash
python chapter_07_attention_mechanism/train.py --epochs 1 --num_samples 2000
```

验收标准：
1. 命令正常结束。
2. 生成 `checkpoints/transformer_copy_best.pth`。
3. 生成 `results/metrics.json`、`results/run_config.json`、`results/predictions.json`。
