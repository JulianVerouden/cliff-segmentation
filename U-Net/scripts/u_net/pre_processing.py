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

def balance_dataset(contains_plant, does_not_contain_plant, cfg):
    keep_percentage = min(1, len(contains_plant) / len(does_not_contain_plant))

    print(f"Keeping {keep_percentage} of tiles without target.")

    skip_tiles = []
    total_tiles_len = len(contains_plant) + len(does_not_contain_plant)

    # Randomly select tiles without target to be discarded. The goal is to get the same amount of tiles with and without the target.
    for i, tile in enumerate(does_not_contain_plant):
        if random.random() > keep_percentage:
            skip_tiles.append(tile)

    # Remove selected tile masks and images
    for tile_name in skip_tiles:
        img_path = cfg.tiles_img_dir / tile_name
        mask_path = cfg.tiles_mask_dir / tile_name

        try:
            img_path.unlink()
            mask_path.unlink()
        except FileNotFoundError:
            pass

    # Log dataset stats
    total_kept = total_tiles_len - len(skip_tiles)

    print(f"Total across dataset: kept {total_kept}, skipped {len(skip_tiles)}")
    print(f"Of the {total_kept} kept, {len(contains_plant)} contain target.")

# Create tiles of a specified size from data folder
def tile_images(cfg: TrainConfig) -> None:
    structure_data_folder(cfg.base_img_dir, cfg.image_dir, cfg.tiles_img_dir)
    structure_data_folder(cfg.base_mask_dir, cfg.mask_dir, cfg.tiles_mask_dir)

    image_files = [p for p in cfg.image_dir.iterdir() if p.is_file()]
    mask_files = {p.stem: p for p in cfg.mask_dir.iterdir() if p.is_file()}

    contains_plant = []
    does_not_contain_plant = []

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

        # List of coordinates evenly spread evenly across the image
        x_coords = np.linspace(0, W - cfg.tile_w, grid_x, dtype=int)
        y_coords = np.linspace(0, H - cfg.tile_h, grid_y, dtype=int)

        # Loop through all tiles in an image
        for i, y in enumerate(y_coords):
            for j, x in enumerate(x_coords):
                # Crop to tile
                img_tile = image.crop((x, y, x + cfg.tile_w, y + cfg.tile_h))
                mask_tile = mask.crop((x, y, x + cfg.tile_w, y + cfg.tile_h))

                # Bin image mask
                mask_arr = np.array(mask_tile)
                mask_bin = (mask_arr >= cfg.tile_threshold).astype(np.uint8) * 255

                mask_tile = Image.fromarray(mask_bin, mode="L")

                tile_img_name = f"{base_name}_{i}_{j}.png"

                # Save img and mask tile
                img_tile.save(cfg.tiles_img_dir / tile_img_name)
                mask_tile.save(cfg.tiles_mask_dir / tile_img_name)

                # Separate tiles that contain none of the target
                if mask_bin.sum() == 0:
                    does_not_contain_plant.append(tile_img_name)
                else:
                    contains_plant.append(tile_img_name)

    if cfg.balance_dataset:
        balance_dataset(contains_plant, does_not_contain_plant, cfg)
    else:
        print(f"Saved a total of {len(contains_plant) + len(does_not_contain_plant)} tiles, of which {len(contains_plant)} contain target.")


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
