"""
Implementation of Grad-CAM for model interpretability.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAM:
    """
    Grad-CAM helper class that captures activations and gradients of a target layer.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Initialize Grad-CAM hooks.

        Args:
            model (nn.Module): Classification model.
            target_layer (nn.Module): Layer to visualize.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._handles = []
        self._register_hooks()

    def _register_hooks(self):
        """
        Register forward/backward hooks on target layer.

        Args:
            None

        Returns:
            None
        """
        self._handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self._handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, module_input, module_output):
        """
        Save forward activations.

        Args:
            module (nn.Module): Hooked module.
            module_input (tuple): Input of hooked module.
            module_output (torch.Tensor): Output of hooked module.

        Returns:
            None
        """
        del module
        del module_input
        self.activations = module_output

    def _save_gradient(self, module, grad_input, grad_output):
        """
        Save backward gradients.

        Args:
            module (nn.Module): Hooked module.
            grad_input (tuple): Gradient wrt module input.
            grad_output (tuple): Gradient wrt module output.

        Returns:
            None
        """
        del module
        del grad_input
        self.gradients = grad_output[0]

    def close(self):
        """
        Remove all registered hooks.

        Args:
            None

        Returns:
            None
        """
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __call__(self, x, class_idx = None):
        """
        Generate Grad-CAM heatmap.

        Args:
            x (torch.Tensor): Input tensor [1, C, H, W].
            class_idx (int | None): Target class index. If None, use predicted class.

        Returns:
            tuple: (heatmap, class_idx)
        """
        outputs = self.model(x)
        if class_idx is None:
            class_idx = outputs.argmax(dim = 1).item()

        self.model.zero_grad(set_to_none = True)
        one_hot_target = torch.zeros_like(outputs)
        one_hot_target[0, class_idx] = 1
        outputs.backward(gradient = one_hot_target, retain_graph = True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("GradCAM hooks did not capture gradients or activations")

        pooled_gradients = torch.mean(self.gradients, dim = (0, 2, 3))
        activations = self.activations.detach().clone()

        for channel_idx in range(activations.shape[1]):
            activations[:, channel_idx, :, :] *= pooled_gradients[channel_idx]

        heatmap = torch.mean(activations, dim = 1).squeeze()
        heatmap = F.relu(heatmap)
        heatmap = heatmap.cpu().numpy()

        max_value = float(np.max(heatmap))
        if max_value > 0.0:
            heatmap = heatmap / max_value
        else:
            heatmap = np.zeros_like(heatmap)

        return heatmap, class_idx

    def __del__(self):
        """
        Ensure hooks are removed when object is deleted.

        Args:
            None

        Returns:
            None
        """
        self.close()


def show_cam_on_image(img_path, heatmap, save_path, alpha = 0.5):
    """
    Overlay Grad-CAM heatmap on original image.

    Args:
        img_path (str): Input image path.
        heatmap (np.ndarray): Heatmap in [0, 1].
        save_path (str): Output image path.
        alpha (float): Overlay alpha factor.

    Returns:
        None
    """
    image = cv2.imread(img_path)
    image = cv2.resize(image, (96, 96))

    resized_heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    resized_heatmap = np.uint8(255 * resized_heatmap)
    colored_heatmap = cv2.applyColorMap(resized_heatmap, cv2.COLORMAP_JET)

    superimposed_image = colored_heatmap * alpha + image
    superimposed_image = np.clip(superimposed_image, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, superimposed_image)
