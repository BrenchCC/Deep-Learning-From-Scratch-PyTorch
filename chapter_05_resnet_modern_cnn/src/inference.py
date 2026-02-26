import os
import sys
import logging
import argparse

import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.append(os.getcwd())

from chapter_05_resnet_modern_cnn.src.cam import GradCAM, show_cam_on_image
from chapter_05_resnet_modern_cnn.src.model import resnet18

from utils import get_device
from utils import setup_seed
from utils import STL10_STATS
from utils import STL10_CLASSES

logger = logging.getLogger("ResNetInference")


def parse_args():
    """
    Parse command-line arguments for chapter 05 inference.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed inference configuration.
    """
    parser = argparse.ArgumentParser(description = "ResNet inference with Grad-CAM")
    parser.add_argument("--img_path", type = str, default = "chapter_05_resnet_modern_cnn/images/airplane.png", help = "Path to input image")
    parser.add_argument("--model_path", type = str, default = "chapter_05_resnet_modern_cnn/checkpoints/resnet18_stl10.pth", help = "Path to model checkpoint")
    return parser.parse_args()


def args_parse():
    """
    Backward-compatible alias for parse_args.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed inference configuration.
    """
    return parse_args()


def preprocess_image(img_path, device):
    """
    Read and preprocess a single input image.

    Args:
        img_path (str): Input image path.
        device (torch.device): Runtime device.

    Returns:
        torch.Tensor: Input tensor with shape [1, 3, 96, 96].
    """
    transform = transforms.Compose([
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(*STL10_STATS)
    ])

    image = Image.open(img_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)
    return input_tensor.to(device)


def load_state_dict(model_path, device):
    """
    Load checkpoint with backward compatibility for torch versions.

    Args:
        model_path (str): Checkpoint path.
        device (torch.device): Runtime device.

    Returns:
        dict: Loaded state dictionary.
    """
    try:
        return torch.load(model_path, map_location = device, weights_only = True)
    except TypeError:
        return torch.load(model_path, map_location = device)


def main():
    """
    Main inference entry for chapter 05.

    Args:
        None

    Returns:
        None
    """
    setup_seed(42)
    device = get_device()
    args = parse_args()

    logger.info(f"Using device: {device}")
    logger.info(f"Loading model from: {args.model_path}")

    if not os.path.exists(args.model_path):
        logger.error(f"Checkpoint not found: {args.model_path}")
        return
    if not os.path.exists(args.img_path):
        logger.error(f"Image not found: {args.img_path}")
        return

    model = resnet18(num_classes = len(STL10_CLASSES), use_residual = True)
    state_dict = load_state_dict(args.model_path, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    input_tensor = preprocess_image(args.img_path, device)
    target_layer = model.layer4[-1].conv2
    grad_cam = GradCAM(model, target_layer)

    try:
        heatmap, class_idx = grad_cam(input_tensor)
    finally:
        grad_cam.close()

    predicted_label = STL10_CLASSES[class_idx]
    logger.info(f"Predicted class: {predicted_label} (index: {class_idx})")

    image_name = os.path.basename(args.img_path)
    save_dir = "chapter_05_resnet_modern_cnn/results/cam_vis"
    os.makedirs(save_dir, exist_ok = True)
    save_path = os.path.join(save_dir, f"cam_{predicted_label}_{image_name}")

    show_cam_on_image(args.img_path, heatmap, save_path)
    logger.info(f"Grad-CAM saved to {save_path}")


def model_inference():
    """
    Backward-compatible alias for main.

    Args:
        None

    Returns:
        None
    """
    main()


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler(sys.stdout)]
    )
    main()
