"""
Defines the ResNet architecture from scratch with a toggle for residual connections.
This allows strictly comparing 'Plain Net' vs 'ResNet' to visualize the degradation problem.
"""
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("Model Architecture")
# 1. Basic Building Block
class BasicBlock(nn.Module):
    """
    Standard ResNet Basic Block: [Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN]
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride size for the first convolution (for downsampling).
        use_residual (bool): If True, adds the shortcut connection (F(x) + x).
                             If False, acts as a plain VGG-style block.
    """
    expansion = 1 # BasicBlock does not expand channel depth (unlike Bottleneck)

    def __init__(
        self,
        in_channels,
        out_channels,
        stride = 1,
        use_residual = True
    ):
        super().__init__()
        self.use_residual = use_residual

        # First convolutional layer
        # NOTE: If stride > 1, this layer handles the spatial downsampling
        self.conv1 = nn.Conv2d(
            in_channels = in_channels, 
            out_channels = out_channels, 
            kernel_size = 3, 
            stride = stride, 
            padding = 1, 
            bias = False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels = out_channels, 
            out_channels = out_channels, 
            kernel_size = 3, 
            stride = 1, 
            padding = 1, 
            bias = False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut (Identity Mapping) handling
        # We only need to adjust the shortcut if:
        # 1. Stride is not 1 (Feature map size changes)
        # 2. Input channels != Output channels (Channel depth changes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels = in_channels, 
                    out_channels = self.expansion * out_channels, 
                    kernel_size = 1, 
                    stride = stride, 
                    bias = False
                ),
                nn.BatchNorm2d(self.expansion * out_channels)
            )
    
    def forward(self, x): 
        """
        Forward pass logic.
        Implements: Output = ReLU(F(x) + x) if residual else ReLU(F(x))
        """
        # --- F(x) Path ---
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # --- Residual Connection ---
        if self.use_residual:
            out = out + self.shortcut(x)
        
        # Final Activation
        out = F.relu(out)
        
        return out
    
# 2. ResNet Architecture    
class ResNet(nn.Module):
    """
    ResNet generic implementation.
    
    Args:
        block (nn.Module): The block type (BasicBlock or Bottleneck).
        num_blocks (list): List of integers defining layers per stage (e.g., [2, 2, 2, 2] for ResNet18).
        num_classes (int): Number of output classes (10 for STL-10).
        use_residual (bool): Global toggle for residual connections.
    """

    def __init__(
        self,
        block,
        num_blocks,
        num_classes = 10,
        use_residual = True
    ):
        super().__init__()
        self.use_residual = use_residual

    # TODO: Implement the rest of the ResNet architecture here.
    # This includes the initial convolutional layer, the four stages of blocks,
if __name__ == "__main__":

    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )

    # Test BasicBlock
    dummy_input = torch.randn(2, 3, 96, 96) # Example input tensor

    basic_block = BasicBlock(
        in_channels = 3,
        out_channels = 64,
        stride = 1,
        use_residual = True
    )
    out = basic_block(dummy_input)
    logger.info(f"Input shape: {dummy_input.shape}")
    logger.info(f"Output shape: {out.shape}")
    logger.info(f"Input data: {dummy_input}")
    logger.info(f"Output data: {out}")

    
