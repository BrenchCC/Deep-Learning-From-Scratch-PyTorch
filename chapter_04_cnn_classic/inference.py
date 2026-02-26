import os
import sys
import logging
import argparse

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

sys.path.append(os.getcwd())

from chapter_04_cnn_classic.model import SimpleCNN

from utils import get_device
from utils import CIFAR10_STATS
from utils import CIFAR10_CLASSES

logger = logging.getLogger("CNNInference")


def parse_args():
    """
    Parse command-line arguments for chapter 04 inference.

    Args:
        None

    Returns:
        argparse.Namespace: Parsed inference configuration.
    """
    parser = argparse.ArgumentParser(description = "Inference on custom images for chapter 04")
    parser.add_argument("--img_dir", type = str, default = "./chapter_04_cnn_classic/data/custom_imgs", help = "Folder with images")
    parser.add_argument("--model_path", type = str, default = "./chapter_04_cnn_classic/results/best_model.pth", help = "Path to trained .pth")
    parser.add_argument("--output_dir", type = str, default = "./chapter_04_cnn_classic/images", help = "Where to save visualizations")
    return parser.parse_args()


def load_image(image_path, device):
    """
    Load and preprocess one input image.

    Args:
        image_path (str): Image path.
        device (torch.device): Runtime device.

    Returns:
        tuple: (pil_image, input_tensor)
    """
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(*CIFAR10_STATS)
    ])

    try:
        pil_image = Image.open(image_path).convert("RGB")
        img_tensor = transform(pil_image).unsqueeze(0).to(device)
        return pil_image, img_tensor
    except Exception as exc:
        logger.error(f"Failed to load image {image_path}: {exc}")
        return None, None


def visualize_feature_maps(model, img_tensor, save_path):
    """
    Visualize feature maps from the first convolution layer.

    Args:
        model (SimpleCNN): Trained model.
        img_tensor (torch.Tensor): Input tensor with shape [1, 3, 32, 32].
        save_path (str): Output path for figure.

    Returns:
        None
    """
    feature_maps = []

    def hook_fn(module, module_input, module_output):
        """
        Save first-layer feature maps from forward pass.

        Args:
            module (torch.nn.Module): Hooked layer.
            module_input (tuple): Layer input.
            module_output (torch.Tensor): Layer output.

        Returns:
            None
        """
        del module
        del module_input
        feature_maps.append(module_output)

    handle = model.block1[0].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(img_tensor)
    handle.remove()

    fmap = feature_maps[0].squeeze(0).cpu()
    fig, axes = plt.subplots(4, 4, figsize = (10, 10))
    fig.suptitle("Feature Maps (Layer 1 - First 16 Channels)", fontsize = 16)

    for idx in range(16):
        ax = axes[idx // 4, idx % 4]
        ax.imshow(fmap[idx], cmap = "viridis")
        ax.axis("off")
        ax.set_title(f"Channel {idx}")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Feature maps saved to {save_path}")


def main():
    """
    Main inference entry for chapter 04.

    Args:
        None

    Returns:
        None
    """
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok = True)

    device = get_device()
    logger.info(f"Using device: {device}")

    if not os.path.exists(args.model_path):
        logger.error(f"Model file not found: {args.model_path}. Please run training first.")
        return

    model = SimpleCNN(num_classes = len(CIFAR10_CLASSES)).to(device)
    state_dict = torch.load(args.model_path, map_location = device)
    model.load_state_dict(state_dict)
    model.eval()
    logger.info(f"Loaded model from {args.model_path}")

    if not os.path.exists(args.img_dir):
        logger.error(f"Image directory does not exist: {args.img_dir}")
        return

    image_files = [file_name for file_name in os.listdir(args.img_dir) if file_name.lower().endswith((".png", ".jpg", ".jpeg"))]
    if len(image_files) == 0:
        logger.warning(f"No images found in {args.img_dir}")
        return

    logger.info(f"Found {len(image_files)} images.")
    for image_file in image_files:
        image_path = os.path.join(args.img_dir, image_file)
        _, image_tensor = load_image(image_path, device)
        if image_tensor is None:
            continue

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.nn.functional.softmax(outputs, dim = 1)
            confidence, prediction = torch.max(probs, dim = 1)

        pred_label = CIFAR10_CLASSES[prediction.item()]
        pred_confidence = confidence.item() * 100.0
        logger.info(f"Image: {image_file} | Prediction: {pred_label} ({pred_confidence:.2f}%)")

        save_name = os.path.splitext(image_file)[0] + "_feature_map.png"
        save_path = os.path.join(args.output_dir, save_name)
        visualize_feature_maps(model, image_tensor, save_path)


if __name__ == "__main__":
    logging.basicConfig(
        level = logging.INFO,
        format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers = [logging.StreamHandler()]
    )
    main()
