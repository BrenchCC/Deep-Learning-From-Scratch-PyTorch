import logging

import torch
import torch.nn as nn

logger = logging.getLogger("SimpleCNN")


class SimpleCNN(nn.Module):
    """
    A classic CNN for CIFAR-10 sized inputs (32 x 32).

    Architecture:
        Block1: Conv -> BN -> ReLU -> MaxPool
        Block2: Conv -> BN -> ReLU -> MaxPool
        Block3: Conv -> BN -> ReLU -> MaxPool
        Head: Flatten -> Linear -> ReLU -> Dropout -> Linear
    """

    def __init__(self, num_classes = 10):
        """
        Initialize model layers.

        Args:
            num_classes (int): Number of classification categories.
        """
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 32, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features = 128 * 4 * 4, out_features = 512),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features = 512, out_features = num_classes)
        )

        self._initialize_weights()

    def _forward_features(self, x):
        """
        Compute convolutional features before classifier.

        Args:
            x (torch.Tensor): Input tensor with shape [batch, 3, 32, 32].

        Returns:
            torch.Tensor: Flattened features with shape [batch, 2048].
        """
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        return torch.flatten(out, 1)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor with shape [batch, 3, 32, 32].

        Returns:
            torch.Tensor: Logits with shape [batch, num_classes].
        """
        features = self._forward_features(x)
        return self.classifier(features)

    def _initialize_weights(self):
        """
        Initialize model parameters with common CNN defaults.

        Args:
            None

        Returns:
            None
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode = "fan_out", nonlinearity = "relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )

    model = SimpleCNN(num_classes = 10)
    dummy_input = torch.randn(1, 3, 32, 32)
    output = model(dummy_input)

    logger.info(f"Output shape: {output.shape}")
    output_probs = nn.Softmax(dim = 1)(output)
    logger.info(f"Output probabilities: {output_probs}")
    output_classes = torch.argmax(output_probs, dim = 1)
    logger.info(f"Predicted class: {output_classes.item()}")
