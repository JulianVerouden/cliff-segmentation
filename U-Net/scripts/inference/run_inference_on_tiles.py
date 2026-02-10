import os
from PIL import Image
import torch
import numpy as np
import torchvision.transforms as T

# -------------------------
# Device & Transform
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

inference_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# -------------------------
# Main function
# -------------------------
def run_inference_on_tiles(tile_dir, out_prob_dir, out_mask_dir, model):
    os.makedirs(out_prob_dir, exist_ok=True)
    os.makedirs(out_mask_dir, exist_ok=True)

    tiles = [f for f in os.listdir(tile_dir) if f.endswith(".png")]

    for t in tiles:
        tile_path = os.path.join(tile_dir, t)
        img = Image.open(tile_path).convert("RGB")

        # ----- Run model -----
        x = inference_transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(x)
            pred = torch.sigmoid(pred).squeeze().cpu().numpy()

        # probability output (float 0–1)
        prob = pred

        # binary mask
        mask = (prob >= 0.5).astype(np.uint8) * 255

        # Save prob & mask
        base = t.replace(".png", "")
        Image.fromarray((prob * 255).astype(np.uint8)).save(
            os.path.join(out_prob_dir, f"{base}_prob.png")
        )
        Image.fromarray(mask).save(
            os.path.join(out_mask_dir, f"{base}_mask.png")
        )
