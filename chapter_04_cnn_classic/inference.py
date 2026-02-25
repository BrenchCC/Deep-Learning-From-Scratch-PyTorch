import os
import sys
import logging
import argparse

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

# Add root path
sys.path.append(os.getcwd())

from utils import get_device

from chapter_04_cnn_classic.model import SimpleCNN

logger = logging.getLogger("Inference")

def load_image(image_path, device):
    """
    Load and preprocess a single image.
    Resizes the image to 32x32 to match CIFAR-10 model input.
    """
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # Force resize
        transforms.ToTensor(),
        transforms.Normalize(*stats)
    ])
    
    try:
        img_pil = Image.open(image_path).convert("RGB")
        img_tensor = transform(img_pil).unsqueeze(0) # Add batch dim: (1, 3, 32, 32)
        return img_pil, img_tensor.to(device)
    except Exception as e:
        logger.error(f"Failed to load image {image_path}: {e}")
        return None, None

def visualize_feature_maps(model, img_tensor, save_path):
    """
    Visualize the output of the first convolutional layer.
    """
    # Hook to capture feature maps
    feature_maps = []
    
    def hook_fn(module, input, output):
        feature_maps.append(output)
    
    # Register hook on the first conv layer of block1
    # model.block1[0] is the first Conv2d
    handle = model.block1[0].register_forward_hook(hook_fn)
    
    # Forward pass
    with torch.no_grad():
        model(img_tensor)
        
    handle.remove()
    
    # Processing maps: (1, 32, 32, 32) -> (32, 32, 32)
    fmap = feature_maps[0].squeeze(0).cpu()
    
    # Plotting first 16 channels
    fig, axes = plt.subplots(4, 4, figsize = (10, 10))
    fig.suptitle("Feature Maps (Layer 1 - First 16 Channels)", fontsize = 16)
    
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.imshow(fmap[i], cmap = "viridis")
        ax.axis("off")
        ax.set_title(f"Channel {i}")
        
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Feature maps saved to {save_path}")

def main():
    # 1. Argparse
    parser = argparse.ArgumentParser(description = "Inference on Custom Images")
    parser.add_argument("--img_dir", type = str, default = "./chapter_04_cnn_classic/data/custom_imgs", help = "Folder with images")
    parser.add_argument("--model_path", type = str, default = "./chapter_04_cnn_classic/results/best_model.pth", help = "Path to .pth file")
    parser.add_argument("--output_dir", type = str, default = "./chapter_04_cnn_classic/images", help = "Where to save viz")
    
    args = parser.parse_args()
    
    # 2. Logging
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    logging.basicConfig(
        level = logging.INFO,
        format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers = [logging.StreamHandler()]
    )
    
    device = get_device()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 3. Load Model
    logger.info(f"Loading model from {args.model_path}...")
    model = SimpleCNN(num_classes = 10).to(device)
    
    if os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location = device))
        model.eval()
    else:
        logger.error("Model file not found! Please run train_cifar.py first.")
        return

    # 4. Process Images
    if not os.path.exists(args.img_dir):
        logger.error(f"Image directory {args.img_dir} does not exist.")
        return

    image_files = [f for f in os.listdir(args.img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    logger.info(f"Found {len(image_files)} images.")

    for img_file in image_files:
        img_path = os.path.join(args.img_dir, img_file)
        pil_img, img_tensor = load_image(img_path, device)
        
        if img_tensor is None: continue
        
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim = 1)
            conf, pred = torch.max(probs, 1)
            
        pred_class = classes[pred.item()]
        confidence = conf.item() * 100
        
        logger.info(f"Image: {img_file} | Prediction: {pred_class} ({confidence:.2f}%)")
        
        # Visualize Feature Maps
        save_name = os.path.splitext(img_file)[0] + "_feature_map.png"
        save_path = os.path.join(args.output_dir, save_name)
        visualize_feature_maps(model, img_tensor, save_path)

if __name__ == "__main__":
    main()
