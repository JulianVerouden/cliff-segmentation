import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import InferenceConfig

import os
import numpy as np
from PIL import Image
from pathlib import Path

def stitch_image(
    cfg: InferenceConfig,
    name
):
    meta_path = os.path.join(cfg.meta_dir, f"{name}_meta.npy")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    meta = np.load(meta_path, allow_pickle=True).item()
    W, H = meta["W"], meta["H"]
    tiles = meta["tiles"]

    # Initialize accumulators
    prob_acc = np.zeros((H, W), dtype=np.float32)
    prob_count = np.zeros((H, W), dtype=np.float32)
    mask_acc = np.zeros((H, W), dtype=np.float32)
    mask_count = np.zeros((H, W), dtype=np.float32)

    for t in tiles:
        x, y = t["x"], t["y"]
        base = t["tile"].replace(".png", "")

        # Load tile probability and mask
        prob_tile = np.array(Image.open(os.path.join(cfg.prob_dir, f"{base}_prob.png"))) / 255.0
        mask_tile = np.array(Image.open(os.path.join(cfg.mask_dir, f"{base}_mask.png"))) / 255.0

        h_tile, w_tile = prob_tile.shape

        # Accumulate into the correct location
        prob_acc[y:y+h_tile, x:x+w_tile] += prob_tile[:h_tile, :w_tile]
        prob_count[y:y+h_tile, x:x+w_tile] += 1

        mask_acc[y:y+h_tile, x:x+w_tile] += mask_tile[:h_tile, :w_tile]
        mask_count[y:y+h_tile, x:x+w_tile] += 1

    # Avoid division by zero
    prob_count[prob_count == 0] = 1
    mask_count[mask_count == 0] = 1

    prob_final = prob_acc / prob_count
    mask_final = (mask_acc / mask_count) > 0.5

    # Save stitched images to output_path
    os.makedirs(Path(cfg.output_dir, "probs"), exist_ok=True)
    os.makedirs(Path(cfg.output_dir, "masks"), exist_ok=True)

    prob_path = os.path.join(cfg.output_dir, "probs", f"{name}_prob_full.png")
    mask_path = os.path.join(cfg.output_dir, "masks", f"{name}_mask_full.png") 

    Image.fromarray((prob_final * 255).astype(np.uint8)).save(prob_path)
    Image.fromarray(mask_final.astype(np.uint8) * 255).save(mask_path)

    print(f"Stitched full image saved: {prob_path} and {mask_path}")
