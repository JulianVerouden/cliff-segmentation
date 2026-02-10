import os
import torch
import argparse
import torchvision.transforms as T
from PIL import Image
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unet_model import UNetResNet50

# -------------------------
# Parse CLI arguments
# -------------------------
parser = argparse.ArgumentParser(description="Run inference on a folder of images.")
parser.add_argument("--input", type=str, required=True, help="Input image folder")
parser.add_argument("--output", type=str, required=True, help="Output mask folder")
parser.add_argument("--model", type=str, default=r"data\checkpoints\unet_resnet50_final.pth",
                    help="Path to trained model weights")

args = parser.parse_args()

IMAGE_FOLDER = args.input
OUTPUT_FOLDER = args.output
MODEL_PATH = args.model

# -------------------------
# Setup
# -------------------------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = UNetResNet50(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Loaded model from {MODEL_PATH}")

# Transforms
inference_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def predict_probability(image_path):
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size

    x = inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)      # (1,1,H,W)
        pred = pred.squeeze().cpu().numpy()  # (H,W)

    # Convert to 0–255 grayscale heatmap
    prob_img = Image.fromarray((pred * 255).astype(np.uint8))
    prob_img = prob_img.resize(orig_size, Image.NEAREST)

    return prob_img


def predict_mask(image_path):
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size

    x = inference_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x)
        pred = torch.sigmoid(pred)
        pred = pred.squeeze().cpu().numpy()

    mask = (pred > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)
    mask_img = mask_img.resize(orig_size, Image.NEAREST)

    return mask_img


# Process images
image_files = [f for f in os.listdir(IMAGE_FOLDER)
               if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))]

print(f"Found {len(image_files)} images.")

for fname in image_files:
    in_path = os.path.join(IMAGE_FOLDER, fname)
    out_path = os.path.join(OUTPUT_FOLDER, fname)

    mask = predict_mask(in_path)
    mask.save(out_path)
    print(f"Saved mask → {out_path}")

    # Save probability heatmap
    prob = predict_probability(in_path)  # new function
    prob_path = os.path.join(OUTPUT_FOLDER, f"{os.path.splitext(fname)[0]}_prob.png")
    prob.save(prob_path)
    print(f"Saved prob heatmap → {prob_path}")

print("Inference complete.")
