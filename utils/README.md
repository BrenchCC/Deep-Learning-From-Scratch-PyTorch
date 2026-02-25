# utils 使用说明

`utils/` 是本项目的通用工具层，用于统一设备选择、日志配置、随机种子、训练循环、模型统计和文件保存逻辑，避免在各章节重复造轮子。

## 快速导入

推荐直接从 `utils` 顶层导入（`utils/__init__.py` 已统一导出常用接口）：

```python
from utils import (
    configure_logging,
    setup_seed,
    get_device,
    run_classification_epoch,
    save_json
)
```

## 模块总览

| 模块 | 主要接口 | 用途 |
| --- | --- | --- |
| `logging_util.py` | `configure_logging` | 统一日志格式与等级 |
| `seed.py` | `device_optimize`, `setup_seed`, `set_seed` | 可复现与线程配置 |
| `device.py` | `get_device` | 自动或手动选择 `cuda/mps/cpu` |
| `timer.py` | `Timer` | 代码块耗时统计（上下文管理器） |
| `model_summary.py` | `count_parameters`, `estimate_model_size`, `log_model_info`, `log_model_info_from_path` | 参数量、模型体积与结构日志 |
| `file_io_util.py` | `save_json`, `load_json`, `save_pickle` | 常用实验结果存取 |
| `data_presets.py` | `CIFAR10_STATS`, `CIFAR10_CLASSES`, `STL10_STATS`, `STL10_CLASSES` | 数据集标准化参数与类别名 |
| `train_loop.py` | `run_classification_epoch`, `default_batch_adapter` | 通用分类训练/验证 epoch 循环 |

## 推荐调用顺序（脚本入口）

在 `main()` 中可按下列顺序初始化：

```python
from utils import configure_logging, device_optimize, setup_seed, get_device


def bootstrap(seed = 42, device_mode = "auto"):
    configure_logging()
    device_optimize(threads = 1)
    setup_seed(seed = seed)
    device = get_device(mode = device_mode)
    return device
```

## 训练循环工具

### `run_classification_epoch(...)`

适用于标准分类任务，一个函数覆盖训练和验证。

- 训练阶段：`stage = "train"`，必须传 `optimizer`
- 验证阶段：`stage = "eval"`，不需要 `optimizer`
- 返回值：`{"loss": float, "acc": float, "num_samples": int}`，其中 `acc` 为百分比

示例：

```python
train_metrics = run_classification_epoch(
    model = model,
    dataloader = train_loader,
    criterion = criterion,
    device = device,
    stage = "train",
    optimizer = optimizer,
    epoch_idx = epoch
)

val_metrics = run_classification_epoch(
    model = model,
    dataloader = val_loader,
    criterion = criterion,
    device = device,
    stage = "eval",
    epoch_idx = epoch
)
```

### 自定义 `batch_adapter`

当 `DataLoader` 返回的 batch 不是 `(inputs, targets)` 时，提供 `batch_adapter` 即可复用同一训练循环。

```python
import torch


def sequence_batch_adapter(batch, device):
    inputs, lengths, targets = batch
    inputs = inputs.to(device)
    lengths = lengths.to(torch.device("cpu"))
    targets = targets.to(device)
    return (inputs, lengths), targets
```

调用时传入：

```python
run_classification_epoch(
    model = model,
    dataloader = train_loader,
    criterion = criterion,
    device = device,
    stage = "train",
    optimizer = optimizer,
    batch_adapter = sequence_batch_adapter
)
```

## 常用工具说明

### 1) 设备与可复现

- `get_device(mode = "auto")`
- `mode = "auto"` 时优先级为 `cuda > mps > cpu`
- `mode = "cuda"` 或 `mode = "mps"` 在设备不可用时会抛 `RuntimeError`
- `setup_seed(seed = 42)` 同时设置 `random / numpy / torch` 随机种子
- `set_seed` 是 `setup_seed` 的兼容别名
- `device_optimize(threads = 1)` 设置常见 BLAS/OpenMP 线程环境变量

### 2) 日志与计时

- `configure_logging(...)`：统一根日志器输出格式
- `Timer(name = "Task")`：记录代码块开始、结束和耗时

```python
from utils import Timer

with Timer("Forward Pass"):
    outputs = model(inputs)
```

### 3) 模型统计

- `count_parameters(model, only_trainable = True)`：统计参数量
- `estimate_model_size(model)`：按 fp32 估算模型大小（MB）
- `log_model_info(model)`：打印结构、总参数量、可训练参数量、估算大小
- `log_model_info_from_path(path, model)`：加载权重后输出模型信息

### 4) 文件读写

- `save_json(data, file_path, indent = 4)`：自动创建目录
- `load_json(file_path)`：读取失败时返回 `{}` 并记录日志
- `save_pickle(obj, file_path)`：保存任意 Python 对象

### 5) 数据集预设

用于 torchvision 标准化和推理可视化标签映射：

```python
from torchvision import transforms
from utils import CIFAR10_STATS, CIFAR10_CLASSES

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(*CIFAR10_STATS)
    ]
)

pred_name = CIFAR10_CLASSES[pred_idx]
```

## 顶层导出清单

`from utils import ...` 可直接导入以下对象：

- `CIFAR10_MEAN`, `CIFAR10_STD`, `CIFAR10_STATS`, `CIFAR10_CLASSES`
- `STL10_MEAN`, `STL10_STD`, `STL10_STATS`, `STL10_CLASSES`
- `get_device`
- `save_json`, `load_json`, `save_pickle`
- `configure_logging`
- `count_parameters`, `estimate_model_size`, `log_model_info`, `log_model_info_from_path`
- `device_optimize`, `setup_seed`, `set_seed`
- `Timer`
- `default_batch_adapter`, `run_classification_epoch`
