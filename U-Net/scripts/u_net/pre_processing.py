import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import make_train_config, TrainConfig

import os
import csv
import math
import random
import argparse
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from scripts.helper_scripts.next_available_path import next_available_path

# Create input and output folders for the tiling of the images
def structure_data_folder(parent_dir, dir_input, dir_output):
    dir_input.mkdir(parents=True, exist_ok=True)
    dir_output.mkdir(parents=True, exist_ok=True)

    # Move images to input folder 
    for item in parent_dir.iterdir():
        if item.is_file():
            shutil.move(str(item), dir_input / item.name)


# Create tiles of a specified size from data folder
def tile_images(cfg: TrainConfig) -> None:
    structure_data_folder(cfg.base_img_dir, cfg.image_dir, cfg.tiles_img_dir)
    structure_data_folder(cfg.base_mask_dir, cfg.mask_dir, cfg.tiles_mask_dir)

    image_files = [p for p in cfg.image_dir.iterdir() if p.is_file()]
    mask_files = {p.stem: p for p in cfg.mask_dir.iterdir() if p.is_file()}

    total_kept = 0
    total_kept_plants = 0
    total_skipped = 0
    stats = []

    for img_path in image_files:
        base_name = img_path.stem
        mask_path = mask_files.get(base_name)

        if mask_path is None:
            continue

        # Skip if already tiled
        already_tiled = any(
            f.startswith(f"{base_name}_")
            for f in os.listdir(cfg.tiles_img_dir)
        )
        if already_tiled:
            print(f"Skipping {img_path}, already tiled")
            continue

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        W, H = image.size

        grid_x = math.ceil(W / cfg.tile_w)
        grid_y = math.ceil(H / cfg.tile_h)

        x_coords = np.linspace(0, W - cfg.tile_w, grid_x, dtype=int)
        y_coords = np.linspace(0, H - cfg.tile_h, grid_y, dtype=int)

        kept_tiles = 0
        kept_tiles_plant = 0
        skipped_tiles = 0

        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                img_tile = image.crop((x, y, x + cfg.tile_w, y + cfg.tile_h))
                mask_tile = mask.crop((x, y, x + cfg.tile_w, y + cfg.tile_h))

                mask_arr = np.array(mask_tile)
                mask_bin = (mask_arr >= cfg.tile_threshold).astype(np.uint8) * 255

                if mask_bin.sum() == 0:
                    if random.random() < 0.9:
                        skipped_tiles += 1
                        continue
                else:
                    kept_tiles_plant += 1

                mask_tile = Image.fromarray(mask_bin, mode="L")

                tile_img_name = f"{base_name}_{i}_{j}.png"

                img_tile.save(cfg.tiles_img_dir / tile_img_name)
                mask_tile.save(cfg.tiles_mask_dir / tile_img_name)

                kept_tiles += 1

        total_kept += kept_tiles
        total_kept_plants += kept_tiles_plant
        total_skipped += skipped_tiles
        stats.append([img_path, kept_tiles, kept_tiles_plant, skipped_tiles])

        print(f"{img_path}: kept {kept_tiles}, skipped {skipped_tiles}")

    if cfg.stats_file is not None:
        with open(next_available_path(cfg.stats_file), mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["filename", "kept_tiles", "kept_tiles_plants", "skipped_tiles"]
            )
            writer.writerows(stats)
            writer.writerow(["TOTAL", total_kept, total_kept_plants, total_skipped])

        print("=" * 40)
        print(f"Total across dataset: kept {total_kept}, skipped {total_skipped}")
        print(f"Stats saved to {cfg.stats_file}")
        print("=" * 40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tile image/mask datasets.")
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Name of the dataset subfolder (e.g. meia_velha)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = make_train_config(args.dataset_name)
    tile_images(cfg)


if __name__ == "__main__":
    main()
