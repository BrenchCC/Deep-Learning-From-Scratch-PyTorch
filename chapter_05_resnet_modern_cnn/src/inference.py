import os
import sys
import logging
import argparse

import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.append(os.getcwd())

from chapter_05_resnet_modern_cnn.src.model import resnet18
from chapter_05_resnet_modern_cnn.src.cam import GradCAM, show_cam_on_image

from utils import get_device, setup_seed, Timer, save_json, count_parameters


# STL-10 Classes mapping
CLASSES = [
    'airplane', 'bird', 'car', 'cat', 'deer', 
    'dog', 'horse', 'monkey', 'ship', 'truck'
]

# Global logger
logger = logging.getLogger("Model_Inference")

def preprocess_image(img_path, device):
    """
    Read and preprocess image for the model.
    """
    # STL-10 Mean/Std
    mean = [0.4467, 0.4398, 0.4066]
    std  = [0.2603, 0.2566, 0.2713]
    
    transform = transforms.Compose([
        transforms.Resize((96, 96)), # Force resize to model input
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    image = Image.open(img_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0) # Add batch dimension [1, 3, 96, 96]
    
    return input_tensor.to(device)

def args_parse():
    parser = argparse.ArgumentParser(description="ResNet Inference with Grad-CAM")
    parser.add_argument('--img_path', type = str, default = 'chapter_05_resnet_modern_cnn/images/airplane.png', 
                        help = 'Path to the input image')
    parser.add_argument('--model_path', type = str, default = 'chapter_05_resnet_modern_cnn/checkpoints/resnet18_stl10.pth',
                        help = 'Path to the trained model weights')
    args = parser.parse_args()
    
    return args

def model_inference():
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler(sys.stdout)]
    )

    setup_seed(42)
    device = get_device()
    logger.info(f"Using device for inference: {device}")

    args = args_parse()

    # Load model
    # NOTE: Must match the structure used in training!
    model = resnet18(num_classes = 10, use_residual = True)

    if not os.path.exists(args.model_path):
        logger.error(f"Checkpoint not found at {args.model_path}. Please train first.")
        return
    
    # Load weights (map_location handles loading GPU weights on CPU/MPS if needed)
    state_dict = torch.load(args.model_path, map_location = device, weights_only = True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval() # Set to evaluation mode (disable Dropout/BN updates)
    
    logger.info(f"Model loaded from {args.model_path}")
    
    # Setup Grad-CAM
    # We want to look at the last layer of the last residual block
    # In our implementation: model.layer4 is the last sequence
    # model.layer4[-1] is the last BasicBlock
    # model.layer4[-1].conv2 is the last convolution
    target_layer = model.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer)

    # Process Image
    if not os.path.exists(args.img_path):
        logger.error(f"Image not found at {args.img_path}")
        return
    
    input_tensor = preprocess_image(args.img_path, device)
    
    # Run Inference & Generation
    heatmap, class_idx = grad_cam(input_tensor)
    
    predicted_label = CLASSES[class_idx]
    logger.info(f"Predicted Class: {predicted_label} (Index: {class_idx})")

    # Save Visualization
    image_name = os.path.basename(args.img_path)
    save_dir = "chapter_05_resnet_modern_cnn/results/cam_vis"
    os.makedirs(save_dir, exist_ok = True)
    save_path = os.path.join(save_dir, f"cam_{predicted_label}_{image_name}")
    
    show_cam_on_image(args.img_path, heatmap, save_path)
    logger.info(f"Grad-CAM saved to {save_path}")

if __name__ == "__main__":
    model_inference()
