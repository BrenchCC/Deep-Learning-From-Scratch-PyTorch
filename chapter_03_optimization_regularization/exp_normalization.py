import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.append(os.getcwd())

from chapter_03_optimization_regularization.exp_common import setup_experiment, log_section, ensure_parent_dir
from chapter_03_optimization_regularization.normalization import BatchNormalization, LayerNormalization, RMSNormalization

def test_norm_stability(norm_layer, inputs, layer_name, logger):
    """
    Run one normalization layer and collect global output statistics.

    Args:
        norm_layer (nn.Module): Normalization layer instance.
        inputs (torch.Tensor): Input tensor.
        layer_name (str): Display name for logs and plotting.
        logger (logging.Logger): Logger instance.

    Returns:
        dict: Result dictionary with name, mean, std, and ok flag.
    """
    result = {"name": layer_name, "mean": np.nan, "std": np.nan, "ok": False}
    try:
        norm_layer.train()
        output = norm_layer(inputs)
        result["mean"] = output.mean().item()
        result["std"] = output.std().item()
        result["ok"] = True
        logger.info(f"[{layer_name}] Output Mean: {result['mean']:.4f}, Std: {result['std']:.4f}")
    except Exception as err:
        logger.error(f"[{layer_name}] Failed: {err}")
    return result

def plot_normalization_stats(records, save_path: str):
    """
    Plot collected mean/std statistics for all normalization runs.

    Args:
        records (list[dict]): Statistics records.
        save_path (str): Output figure path.
    """
    labels = [item["name"] for item in records]
    means = [item["mean"] for item in records]
    stds = [item["std"] for item in records]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(2, 1, figsize = (14, 10), sharex = True)
    axes[0].bar(x, means, color = 'steelblue', alpha = 0.8)
    axes[0].axhline(0.0, color = 'gray', linestyle = '--', linewidth = 1)
    axes[0].set_ylabel("Global Mean")
    axes[0].set_title("Normalization Stability: Output Mean and Std Across Experiments")
    axes[0].grid(axis = 'y', alpha = 0.3)

    axes[1].bar(x, stds, color = 'darkorange', alpha = 0.8)
    axes[1].axhline(1.0, color = 'gray', linestyle = '--', linewidth = 1, label = 'Target Std ~ 1')
    axes[1].set_ylabel("Global Std")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation = 35, ha = 'right')
    axes[1].grid(axis = 'y', alpha = 0.3)
    axes[1].legend()

    ensure_parent_dir(save_path)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    logger = setup_experiment(logger_name = "Exp_Normalization", seed = 42)
    output_path = "chapter_03_optimization_regularization/images/normalization_stability_comparison.png"
    records = []

    dim = 64
    seq_len = 10

    log_section(logger, "Step 1/4: Standard Batch Size (N = 32)")
    data = torch.randn(32, seq_len, dim) * 5.0 + 10.0
    data_permuted = data.permute(0, 2, 1)

    records.append(test_norm_stability(nn.BatchNorm1d(dim), data_permuted, "PyTorch_BN_BS32", logger))
    records.append(test_norm_stability(BatchNormalization(num_features = dim), data_permuted, "Manual_BN_BS32", logger))
    records.append(test_norm_stability(nn.LayerNorm(normalized_shape = dim), data, "PyTorch_LN_BS32", logger))
    records.append(test_norm_stability(LayerNormalization(normalized_shape = dim), data, "Manual_LN_BS32", logger))
    records.append(test_norm_stability(RMSNormalization(dim = dim), data, "Manual_RMS_BS32", logger))

    log_section(logger, "Step 2/4: Single Batch (N = 1)")
    data_single = torch.randn(1, seq_len, dim) * 5.0 + 10.0
    data_single_permuted = data_single.permute(0, 2, 1)

    records.append(test_norm_stability(nn.BatchNorm1d(dim), data_single_permuted, "PyTorch_BN_BS1", logger))
    records.append(test_norm_stability(BatchNormalization(num_features = dim), data_single_permuted, "Manual_BN_BS1", logger))
    records.append(test_norm_stability(nn.LayerNorm(normalized_shape = dim), data_single, "PyTorch_LN_BS1", logger))
    records.append(test_norm_stability(LayerNormalization(normalized_shape = dim), data_single, "Manual_LN_BS1", logger))
    records.append(test_norm_stability(RMSNormalization(dim = dim), data_single, "Manual_RMS_BS1", logger))

    log_section(logger, "Step 3/4: Plot and Save Figure")
    plot_normalization_stats(records = records, save_path = output_path)
    logger.info(f"Saved figure: {output_path}")

    log_section(logger, "Step 4/4: Summary")
    logger.info("Conclusion: LN and RMSNorm are batch-size independent and stable for BS = 1.")

if __name__ == "__main__":
    main()
