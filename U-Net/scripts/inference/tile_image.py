import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import InferenceConfig

import os
import numpy as np
from PIL import Image
import math
from pathlib import Path

def tile_image(
    cfg: InferenceConfig,
    image_path
):
    os.makedirs(cfg.tile_dir, exist_ok=True)
    os.makedirs(cfg.meta_dir, exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    W, H = img.size
    name = os.path.splitext(os.path.basename(image_path))[0]

    # Helper to compute coordinates (ensure full coverage)
    def compute_coords(size, tile_size, grid):
        coords = np.linspace(0, max(size - tile_size, 0), grid, dtype=int)
        if coords[-1] != size - tile_size:
            coords = np.append(coords, size - tile_size)
        return coords

    grid_x = math.ceil(W / cfg.tile_w)
    grid_y = math.ceil(H / cfg.tile_h)

    x_coords = compute_coords(W, cfg.tile_w, grid_x)
    y_coords = compute_coords(H, cfg.tile_h, grid_y)

    tiles = []

    for yi, y in enumerate(y_coords):
        for xi, x in enumerate(x_coords):
            # Clip tile to image edges
            x_end = min(x + cfg.tile_w, W)
            y_end = min(y + cfg.tile_h, H)
            img_tile = img.crop((x, y, x_end, y_end))

            # Pad tile if smaller than tile_w x tile_h
            tile_array = np.array(img_tile)
            if tile_array.shape[0] != cfg.tile_h or tile_array.shape[1] != cfg.tile_w:
                padded = np.zeros((cfg.tile_h, cfg.tile_w, 3), dtype=tile_array.dtype)
                padded[:tile_array.shape[0], :tile_array.shape[1], :] = tile_array
                img_tile = Image.fromarray(padded)

            tile_name = f"{name}_{yi}_{xi}.png"
            img_tile.save(os.path.join(cfg.tile_dir, tile_name))

            tiles.append({
                "tile": tile_name,
                "x": x,
                "y": y,
                "w": x_end - x,
                "h": y_end - y,
                "yi": yi,
                "xi": xi
            })

    # Save metadata
    meta_path = os.path.join(cfg.meta_dir, f"{name}_meta.npy")
    meta = {
        "W": W,
        "H": H,
        "tiles": tiles,
        "tile_w": cfg.tile_w,
        "tile_h": cfg.tile_h,
    }

    np.save(meta_path, np.array(meta, dtype=object))

    print(f"Tiled {name}: {len(tiles)} tiles saved to {cfg.tile_dir}")
    return meta_path
