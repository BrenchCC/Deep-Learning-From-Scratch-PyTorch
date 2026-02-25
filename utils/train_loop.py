"""
Reusable training and evaluation loop helpers for classification tasks.
"""

from typing import Any
from typing import Callable
from typing import Optional
from typing import Tuple

import torch
from tqdm import tqdm

ModelInputs = Tuple[torch.Tensor, ...]
BatchAdapter = Callable[[Any, torch.device], Tuple[ModelInputs, torch.Tensor]]


def default_batch_adapter(batch: Any, device: torch.device) -> Tuple[ModelInputs, torch.Tensor]:
    """
    Convert a standard classification batch to model inputs and targets.

    Args:
        batch (Any): A batch from DataLoader, expected as ``(inputs, targets)``.
        device (torch.device): Runtime device for tensors.

    Returns:
        Tuple[ModelInputs, torch.Tensor]: ``((inputs,), targets)`` moved to ``device``.
    """
    inputs, targets = batch
    inputs = inputs.to(device)
    targets = targets.to(device)
    return (inputs,), targets


def run_classification_epoch(
    model: torch.nn.Module,
    dataloader,
    criterion: torch.nn.Module,
    device: torch.device,
    stage: str = "train",
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch_idx: Optional[int] = None,
    batch_adapter: Optional[BatchAdapter] = None,
    max_grad_norm: Optional[float] = None
) -> dict:
    """
    Run a single classification epoch for training or evaluation.

    Args:
        model (torch.nn.Module): Classification model.
        dataloader: DataLoader that yields batches.
        criterion (torch.nn.Module): Loss function.
        device (torch.device): Runtime device.
        stage (str): ``"train"`` or ``"eval"``.
        optimizer (Optional[torch.optim.Optimizer]): Optimizer required for training.
        epoch_idx (Optional[int]): Epoch index used in progress bar label.
        batch_adapter (Optional[BatchAdapter]): Adapter that maps raw batch to
            ``(model_inputs, targets)``.
        max_grad_norm (Optional[float]): Gradient clipping max norm for training.

    Returns:
        Dict[str, float]: Epoch metrics ``{"loss", "acc", "num_samples"}``.
    """
    normalized_stage = stage.lower().strip()
    if normalized_stage not in {"train", "eval"}:
        raise ValueError(f"Invalid stage: {stage}. Use 'train' or 'eval'.")

    is_train = normalized_stage == "train"
    if is_train and optimizer is None:
        raise ValueError("optimizer must be provided when stage = 'train'.")

    if batch_adapter is None:
        batch_adapter = default_batch_adapter

    if is_train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    if epoch_idx is None:
        loop_desc = normalized_stage.upper()
    else:
        loop_desc = f"Epoch {epoch_idx} [{normalized_stage.upper()}]"
    loop = tqdm(dataloader, desc = loop_desc, leave = False)

    with torch.set_grad_enabled(is_train):
        for batch in loop:
            model_inputs, targets = batch_adapter(batch, device)
            if not isinstance(model_inputs, tuple):
                model_inputs = (model_inputs,)

            outputs = model(*model_inputs)
            loss = criterion(outputs, targets)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = max_grad_norm)
                optimizer.step()

            batch_size = int(targets.size(0))
            running_loss += loss.item() * batch_size
            predicted = outputs.argmax(dim = 1)
            correct += predicted.eq(targets).sum().item()
            total += batch_size

            current_acc = 100.0 * correct / max(total, 1)
            loop.set_postfix(loss = f"{loss.item():.4f}", acc = f"{current_acc:.2f}%")

    avg_loss = running_loss / max(total, 1)
    avg_acc = 100.0 * correct / max(total, 1)
    return {"loss": avg_loss, "acc": avg_acc, "num_samples": total}
