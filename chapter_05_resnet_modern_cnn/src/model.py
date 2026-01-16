"""
Defines the ResNet architecture from scratch with a toggle for residual connections.
This allows strictly comparing 'Plain Net' vs 'ResNet' to visualize the degradation problem.
"""
import os
import sys
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
    expansion = 1  # BasicBlock does not expand channel depth (unlike Bottleneck)

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        use_residual=True
    ):
        super().__init__()
        self.use_residual = use_residual

        # First convolutional layer
        # NOTE: If stride > 1, this layer handles the spatial downsampling
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
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
                    in_channels=in_channels,
                    out_channels=self.expansion * out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
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
        num_classes=10,
        use_residual=True
    ):
        super().__init__()
        self.use_residual = use_residual
        self.in_channels = 64  # Initial number of channels after the stem

        # --- Initial Stem (Input Processing) ---
        # For STL-10 (96x96), standard ResNet stem (7x7 conv, stride 2) is fine.
        # It reduces 96 -> 48 immediately.
        self.conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = 64,
            kernel_size = 7,
            stride = 2,
            padding = 3,
            bias = False
        )
        self.bn1 = nn.BatchNorm2d(64)
        # MaxPool reduces 48 -> 24
        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1
        )

        # --- Residual Layers (Stages 1-4) ---
        # Layer 1: 64 channels, output 24x24
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # Layer 2: 128 channels, output 12x12
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # Layer 3: 256 channels, output 6x6
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # Layer 4: 512 channels, output 3x3
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # --- Classification Head ---
        # Global Average Pooling will transform any (C, H, W) to (C, 1, 1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Apply Zero-Gamma Initialization for better convergence
        self._initialize_weights()

    def _make_layer(
        self,
        block: BasicBlock,
        out_channels: int,
        blocks: int,
        stride: int = 1
    ):
        """
        Constructs a sequence of blocks (a ResNet stage).
        """
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(
                block(self.in_channels, out_channels, s, self.use_residual))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """
        Kaiming Initialization + Zero Gamma Trick for last BN in each residual block.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero Gamma Trick:
        # Initialize the last BN in each residual block to zero.
        # This makes the block act as an identity mapping initially.
        # if self.use_residual:
        #     for m in self.modules():
        #         if isinstance(m, BasicBlock):
        #             nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x):
        """
        Full network forward pass.
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        # Layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Head
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    # 3. Model Builders
def resnet18(num_classes = 10, use_residual = True):
    """
    Constructs a ResNet-18 model.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes = num_classes, use_residual = use_residual)

def resnet34(num_classes = 10, use_residual = True):
    """
    Constructs a ResNet-34 model.
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes = num_classes, use_residual = use_residual)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    # Dummy input for STL-10 size
    dummy_input = torch.randn(2, 3, 96, 96)  # Example input tensor

    # Test BasicBlock
    basic_block = BasicBlock(
        in_channels=3,
        out_channels=64,
        stride=1,
        use_residual=True
    )
    out = basic_block(dummy_input)
    logger.info(f"Input shape: {dummy_input.shape}")
    logger.info(f"Output shape: {out.shape}")
    logger.info(f"Input data: {dummy_input}")
    logger.info(f"Output data: {out}")


    # Test ResNet
    model_res = resnet18(use_residual = True)
    out_res = model_res(dummy_input)
    logger.info(f"ResNet18 Output Shape: {out_res.shape}") # Should be [2, 10]
    
    # Test PlainNet
    model_plain = resnet18(use_residual = False)
    out_plain = model_plain(dummy_input)
    logger.info(f"PlainNet18 Output Shape: {out_plain.shape}")
    
    # Count params
    sys.path.append(os.getcwd())  # Ensure utils can be imported
    from utils import count_parameters
    logger.info(f"Params: {count_parameters(model_res)}")
