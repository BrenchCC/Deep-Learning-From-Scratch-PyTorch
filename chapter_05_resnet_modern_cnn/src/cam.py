"""
src/cam.py
Implementation of Grad-CAM (Gradient-weighted Class Activation Mapping).
Allows visualizing which parts of the image contributed most to the model's prediction.
"""
import os
import sys
import cv2
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

sys.path.append(os.getcwd())

from chapter_05_resnet_modern_cnn.src.model import ResNet

class GradCAM:
    """
    Grad-CAM helper class.
    
    Args:
        model (nn.Module): The CNN model (e.g., ResNet).
        target_layer (nn.Module): The specific layer to visualize (usually the last Conv layer).
    """
    def __init__(self, model: ResNet, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to capture data during forward/backward pass
        # 1. Capture the feature map (Forward)
        target_layer.register_forward_hook(self.save_activation)
        # 2. Capture the gradients (Backward)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """
        Hook callback to save forward feature maps.
        """
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        Hook callback to save backward gradients.
        """
        self.gradients = grad_output[0]
    
    def __call__(self, x, class_idx = None):
            """
            Generates the heatmap for a specific input and target class.
            
            Args:
                x (Tensor): Input image tensor [1, C, H, W].
                class_idx (int, optional): The target class index. 
                                        If None, uses the highest predicted class.
            
            Returns:
                heatmap (np.array): The raw heatmap (0-1).
                result_class (int): The predicted or target class index.
            """
            # 1. Forward Pass
            output = self.model(x)
            
            if class_idx is None:
                # Get the index of the max log-probability
                class_idx = output.argmax(dim = 1).item()
            
            # 2. Backward Pass (Clear previous grads first)
            self.model.zero_grad()
            
            # Create a one-hot target tensor to backpropagate specific class
            one_hot_target = torch.zeros_like(output)
            one_hot_target[0][class_idx] = 1
            
            # Trigger backward to compute gradients w.r.t features
            output.backward(gradient = one_hot_target, retain_graph = True)
            
            # 3. Generate CAM
            # Gradients shape: [1, 512, H, W] -> Pooled Weight: [1, 512, 1, 1]
            pooled_gradients = torch.mean(self.gradients, dim = [0, 2, 3])
            
            # Activations shape: [1, 512, H, W]
            activations = self.activations.detach()
            
            # Weight the channels (batteries included broadcasting)
            # We loop to avoid complex broadcasting dimensions for readability
            for i in range(activations.shape[1]):
                activations[:, i, :, :] *= pooled_gradients[i]
                
            # Average the channels to get 2D heatmap: [1, H, W]
            heatmap = torch.mean(activations, dim = 1).squeeze()
            
            # Apply ReLU (We only care about positive influence)
            heatmap = F.relu(heatmap)
            
            # Normalize to 0-1
            heatmap = heatmap.cpu().numpy()
            heatmap /= np.max(heatmap) + 1e-8 # Avoid div by zero
            
            return heatmap, class_idx


def show_cam_on_image(img_path, heatmap, save_path, alpha = 0.5):
    """
    Overlay the heatmap on the original image.
    
    Args:
        img_path (str): Path to original image.
        heatmap (np.array): The 2D heatmap.
        save_path (str): Path to save the result.
        alpha (float): Transparency factor.
    """
    # Load original image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (96, 96)) # Resize to match model input for visualization
    
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Convert to RGB heatmap (JET colormap is standard for heatmaps)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Overlay
    superimposed_img = heatmap * alpha + img
    cv2.imwrite(save_path, superimposed_img)