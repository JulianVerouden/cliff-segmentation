import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import csv
from config import TrainConfig, GenerateTestSplit

import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
import numpy as np

def perform_spatial_split(df, eps_meters, train_fraction):
    # Remove images without GPS
    df = df.dropna(subset=["latitude", "longitude"])
    coords = df[["latitude", "longitude"]].values

    # Cluster by spatial proximity
    eps_degrees = eps_meters / 111_000
    db = DBSCAN(eps=eps_degrees, min_samples=1).fit(coords)
    df["cluster"] = db.labels_

    # Compute tile counts per cluster
    cluster_sums = df.groupby("cluster")["tile_count"].sum().reset_index().sort_values("tile_count", ascending=False)
    total_tiles = cluster_sums["tile_count"].sum()

    # Select clusters for training set (~train_fraction of tiles)
    target_tiles = train_fraction * total_tiles
    train_clusters = []
    cum_sum = 0

    print("target tiles: ", target_tiles)
    for _, row in cluster_sums.iterrows():
        print("cum_sum: ", cum_sum)
        if cum_sum < target_tiles:
            cum_sum_new = cum_sum + row["tile_count"]
            print("cum_sum_new: ", cum_sum_new)
            
            # Check if new cumulative sum exceeds the target tiles, if so: append the cluster based on whether adding it is closer or bigger than the wanted tile count.
            if (cum_sum_new > target_tiles):
                dif = abs(target_tiles - cum_sum)
                dif_new = abs(target_tiles - cum_sum_new)

                print("dif: ", dif)
                print("new_dif: ", dif_new)
                if (dif < dif_new):
                    break
                else:
                    train_clusters.append(row["cluster"])
            else:
                train_clusters.append(row["cluster"])
                cum_sum += row["tile_count"]
        else:
            break

    df["set"] = df["cluster"].apply(lambda c: "train" if c in train_clusters else "test")
    df.drop(columns=["cluster"], inplace=True)

    return df

def create_spatial_split_csv(
    spatial_df: pd.DataFrame,
    tiles_dir: Path,
    output_csv: Path,
):
    # --- Safety checks ---
    required_cols = {"filename", "set"}

    if not required_cols.issubset(spatial_df.columns):
        raise ValueError(f"spatial_df must contain columns {required_cols}")

    # Build image -> set lookup dictionary
    image_to_set = dict(
        zip(spatial_df["filename"], spatial_df["set"])
    )

    tile_rows = []

    # Iterate over all tile files
    for tile_path in tiles_dir.iterdir():
        if not tile_path.is_file():
            continue

        tile_name = tile_path.name

        # Extract parent image name
        # Assumes format: image_a_0_0.png
        stem = tile_path.stem  # image_a_0_0
        parts = stem.split("_")

        if len(parts) < 3:
            continue  # skip unexpected filenames

        image_name = "_".join(parts[:-2])  # remove row_col

        if image_name not in image_to_set:
            continue  # tile from image not in spatial split

        tile_rows.append({
            "filename": tile_name,
            "set": image_to_set[image_name]
        })

    # --- Write CSV ---
    with output_csv.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=["filename", "set"])
        writer.writeheader()
        writer.writerows(tile_rows)

    print(f"Saved tile split CSV to: {output_csv}")

def random_tile_split(csv_dir, image_folder, mask_folder, train_fraction=0.85, seed=42):
    np.random.seed(seed)

    # List only files, skip folders
    all_image_tiles = sorted([f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))])
    all_mask_tiles = sorted([f for f in os.listdir(mask_folder) if os.path.isfile(os.path.join(mask_folder, f))])

    # Build mapping from image -> mask based on naming convention
    # Adjust this if your masks have a different pattern
    image_to_mask = {}
    for img in all_image_tiles:
        mask_name = img
        if mask_name in all_mask_tiles:
            image_to_mask[img] = mask_name
        else:
            raise FileNotFoundError(f"Mask not found for image {img}: expected {mask_name}")

    # Shuffle images
    shuffled_indices = np.random.permutation(len(all_image_tiles))
    n_train = int(len(all_image_tiles) * train_fraction)
    train_indices = shuffled_indices[:n_train]
    test_indices  = shuffled_indices[n_train:]

    train_images = [all_image_tiles[i] for i in train_indices]
    test_images  = [all_image_tiles[i] for i in test_indices]

    # Create csv with test/train field per tile
    with csv_dir.open("w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["filename", "set"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for train_img in train_images:
            row = {"filename": train_img, "set": "train"}
            writer.writerow(row)
        for test_img in test_images:
            row = {"filename": test_img, "set": "test"}
            writer.writerow(row)

    print(f"Random tile split done: {len(train_images)} tiles in training, {len(test_images)} tiles in test.")

# Plot spatial coordinates of images based on the train and test split
def plot_split(df):
    plt.scatter(
        df["longitude"],
        df["latitude"],
        c=df["set"].map({"train": "red", "test": "blue"}),
        s=10,
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Spatial Split — Training Set (red) vs Test Set B (blue)")
    plt.show()

def main(cfg: TrainConfig):
    if (cfg.test_split_method == GenerateTestSplit.SPATIAL):
        print(f"Creating a spatial tile split")
        csv_path = Path(cfg.metadata_file)
        df = pd.read_csv(csv_path)
        df = perform_spatial_split(df, cfg.eps_meters, cfg.train_fraction)
        create_spatial_split_csv(df, cfg.tiles_img_dir, cfg.test_split)
        
        # Plot and show results
        total_tiles = df["tile_count"].sum()
        tiles_a = df.loc[df["set"] == "train", "tile_count"].sum()
        tiles_b = df.loc[df["set"] == "test", "tile_count"].sum()

        print(f"Set A (training): {tiles_a} tiles ({tiles_a / total_tiles:.1%})")
        print(f"Set B (test): {tiles_b} tiles ({tiles_b / total_tiles:.1%})")
        plot_split(df)
    elif(cfg.test_split_method == GenerateTestSplit.RANDOM):
        print("Creating a random tile split")
        random_tile_split(
            csv_dir=cfg.test_split,
            image_folder=cfg.tiles_img_dir,
            mask_folder=cfg.tiles_mask_dir,
            train_fraction=cfg.train_fraction,
            seed=cfg.random_split_seed
        )
