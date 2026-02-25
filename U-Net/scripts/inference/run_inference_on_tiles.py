import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import InferenceConfig
import os
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as T


# Device & Transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inference_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def run_inference_on_tiles(cfg: InferenceConfig, model):
    os.makedirs(cfg.prob_dir, exist_ok=True)
    os.makedirs(cfg.mask_dir, exist_ok=True)

    tiles = [f for f in os.listdir(cfg.tile_dir) if f.endswith(".png")]

    for t in tiles:
        tile_path = os.path.join(cfg.tile_dir, t)
        img = Image.open(tile_path).convert("RGB")

        # Run model
        x = inference_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()

        # probability output (float 0–1)
        prob = pred

        # binary mask
        mask = (prob >= cfg.segmentation_threshold).astype(np.uint8) * 255

        # Save prob & mask
        base = t.replace(".png", "")
        Image.fromarray((prob * 255).astype(np.uint8)).save(
            os.path.join(cfg.prob_dir, f"{base}_prob.png")
        )
        Image.fromarray(mask).save(
            os.path.join(cfg.mask_dir, f"{base}_mask.png")
        )
