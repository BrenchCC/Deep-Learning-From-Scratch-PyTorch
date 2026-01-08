import logging
import torch
import torch.nn as nn

logger = logging.getLogger("CNN_Demo_Model")

class SimpleCNN(nn.Module):
    """
    A classic CNN architecture for CIFAR-10 sized images (32x32).
    Structure:
    - Block 1: Conv(32) -> BN -> ReLU -> MaxPool
    - Block 2: Conv(64) -> BN -> ReLU -> MaxPool
    - Block 3: Conv(128) -> BN -> ReLU -> MaxPool
    - Classifier: Flatten -> Linear -> ReLU -> Dropout -> Linear
    """
    
    def __init__(self, num_classes = 10):
        """
        Initialize the SimpleCNN model.
        
        Args:
            num_classes (int): Number of output classes. Default is 10 for CIFAR-10.
        """
        super(SimpleCNN, self).__init__()
        
        # Feature Extraction Block 1
        # Input: (Batch, 3, 32, 32) -> Output: (Batch, 32, 16, 16)
        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 3, 
                out_channels = 32, 
                kernel_size = 3, 
                padding = 1, 
                bias = False
            ),
            nn.BatchNorm2d(num_features = 32),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        # Feature Extraction Block 2
        # Input: (Batch, 32, 16, 16) -> Output: (Batch, 64, 8, 8)
        self.block2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 32, 
                out_channels = 64, 
                kernel_size = 3, 
                padding = 1, 
                bias = False
            ),
            nn.BatchNorm2d(num_features = 64),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        # Feature Extraction Block 3
        # Input: (Batch, 64, 8, 8) -> Output: (Batch, 128, 4, 4)
        self.block3 = nn.Sequential(
            nn.Conv2d(
                in_channels = 64, 
                out_channels = 128, 
                kernel_size = 3, 
                padding = 1, 
                bias = False
            ),
            nn.BatchNorm2d(num_features = 128),
            nn.ReLU(inplace = True),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        
        # Classifier Head
        # Flatten size: 128 channels * 4 * 4 = 2048
        self.classifier = nn.Sequential(
            nn.Linear(in_features = 128 * 4 * 4, out_features = 512),
            nn.ReLU(inplace = True),
            nn.Dropout(p = 0.5),
            nn.Linear(in_features = 512, out_features = num_classes)
        )
        
        # Weight Initialization (Optional but good practice)
        self._initialize_weights()

    def forward(self, x):
        """
        Forward pass logic.
        
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, 32, 32)
            
        Returns:
            torch.Tensor: Logits of shape (Batch, num_classes)
        """
        # Feature Extraction
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        
        # Flatten: (Batch, 128, 4, 4) -> (Batch, 2048)
        out = torch.flatten(out, 1)
        
        # Classification
        out = self.classifier(out)
        
        return out

    def _initialize_weights(self):
        """
        Initialize weights using Kaiming Normal for Conv and Linear layers.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    logging.basicConfig(
            level = logging.INFO,
            format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers = [logging.StreamHandler()]
        )
    # Test the model with a dummy input
    model = SimpleCNN(num_classes = 10)
    dummy_input = torch.randn(1, 3, 32, 32)  # Batch size of 1, 3 channels, 32x32 image
    output = model(dummy_input)
    logger.info(f"Output shape: {output.shape}")  # Should be (1, 10) for 10 classes
    output_probs = nn.Softmax(dim = 1)(output)
    logger.info(f"Output probabilities: {output_probs}")
    output_classes = torch.argmax(output_probs, dim = 1)
    logger.info(f"Predicted class: {output_classes.item()}")

    