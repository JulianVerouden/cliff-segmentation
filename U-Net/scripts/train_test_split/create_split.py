import pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np
import argparse

# ---------- CONFIG ----------
csv_path = r"data\train_test_split_meia_velha.csv"
eps_meters = 60          # distance threshold for clustering (adjust as needed)
train_fraction = 0.85    # fraction of tiles for training set
random_seed = 42          # for reproducibility

# ---------- FUNCTIONS ----------
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
    for _, row in cluster_sums.iterrows():
        if cum_sum < target_tiles:
            train_clusters.append(row["cluster"])
            cum_sum += row["tile_count"]
        else:
            break

    df["set"] = df["cluster"].apply(lambda c: "A" if c in train_clusters else "B")
    df.drop(columns=["cluster"], inplace=True)
    return df

def random_tile_split(image_folder, mask_folder, train_fraction=0.85, seed=42):
    import os
    import shutil
    import numpy as np

    np.random.seed(seed)

    # List only files, skip folders
    all_image_tiles = sorted([f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))])
    all_mask_tiles = sorted([f for f in os.listdir(mask_folder) if os.path.isfile(os.path.join(mask_folder, f))])

    # Build mapping from image -> mask based on naming convention
    # Adjust this if your masks have a different pattern
    image_to_mask = {}
    for img in all_image_tiles:
        mask_name = img.replace(".png", "_mask.png")  # adjust if needed
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

    # Get corresponding masks
    train_masks = [image_to_mask[f] for f in train_images]
    test_masks  = [image_to_mask[f] for f in test_images]

    # Ensure folders exist
    def make_and_move(files, src, dst):
        os.makedirs(dst, exist_ok=True)
        for f in files:
            shutil.move(os.path.join(src, f), os.path.join(dst, f))

    make_and_move(train_images, image_folder, os.path.join(image_folder, "training"))
    make_and_move(test_images,  image_folder, os.path.join(image_folder, "test"))
    make_and_move(train_masks,  mask_folder,  os.path.join(mask_folder, "training"))
    make_and_move(test_masks,   mask_folder,  os.path.join(mask_folder, "test"))

    print(f"Random tile split done: {len(train_images)} tiles in training, {len(test_images)} tiles in test.")

def plot_split(df):
    plt.scatter(
        df["longitude"],
        df["latitude"],
        c=df["set"].map({"A": "red", "B": "blue"}),
        s=10,
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Spatial Split — Set A (red) vs Set B (blue)")
    plt.show()

def move_tiles(df, name=None):
    if name:
        image_folder = os.path.join("data", "images", name, "tiles")
        mask_folder = os.path.join("data", "masks", name, "tiles")
    else:
        image_folder = os.path.join("data", "images", "tiles")
        mask_folder = os.path.join("data", "masks", "tiles")

    files_train = df.loc[df['set'] == "A", "filename"].tolist()
    files_test = df.loc[df['set'] == "B", "filename"].tolist()

    all_tiles_images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    all_tiles_masks = [f for f in os.listdir(mask_folder) if os.path.isfile(os.path.join(mask_folder, f))]

    tiles_images_train = [tile for tile in all_tiles_images if any(name in tile for name in files_train)]
    tiles_images_test = [tile for tile in all_tiles_images if any(name in tile for name in files_test)]
    tiles_masks_train = [tile for tile in all_tiles_masks if any(name in tile for name in files_train)]
    tiles_masks_test = [tile for tile in all_tiles_masks if any(name in tile for name in files_test)]

    def _move(tiles, src, dst):
        os.makedirs(dst, exist_ok=True)
        for tile in tiles:
            shutil.move(os.path.join(src, tile), os.path.join(dst, tile))

    _move(tiles_images_train, image_folder, os.path.join(image_folder, "training"))
    _move(tiles_images_test, image_folder, os.path.join(image_folder, "test"))
    _move(tiles_masks_train, mask_folder, os.path.join(mask_folder, "training"))
    _move(tiles_masks_test, mask_folder, os.path.join(mask_folder, "test"))

# ---------- MAIN ----------
def main(spatial_split=True, seed=random_seed, name=None):
    if spatial_split:
        df = pd.read_csv(csv_path)
        df = perform_spatial_split(df, eps_meters, train_fraction)
        df.to_csv(csv_path, index=False)
        move_tiles(df, name)
        
        # Sanity check & plot
        total_tiles = df["tile_count"].sum()
        tiles_a = df.loc[df["set"] == "A", "tile_count"].sum()
        tiles_b = df.loc[df["set"] == "B", "tile_count"].sum()
        print(f"Set A (training): {tiles_a} tiles ({tiles_a / total_tiles:.1%})")
        print(f"Set B (test): {tiles_b} tiles ({tiles_b / total_tiles:.1%})")
        plot_split(df)
    else:
        # Determine folders
        if name:
            image_folder = os.path.join("data", "images", name, "tiles")
            mask_folder = os.path.join("data", "masks", name, "tiles")
        else:
            image_folder = os.path.join("data", "images", "tiles")
            mask_folder = os.path.join("data", "masks", "tiles")
        
        print("Creating a random tile split")
        random_tile_split(
            image_folder=image_folder,
            mask_folder=mask_folder,
            train_fraction=train_fraction,
            seed=seed
        )

# ---------- RUN ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset spatially or randomly.")
    parser.add_argument("--spatial_split", type=int, default=1, help="1 for spatial split, 0 for random tile split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--name", type=str, default=None, help="Dataset name (e.g., meia_velha). Optional")
    args = parser.parse_args()

    main(spatial_split=bool(args.spatial_split), seed=args.seed, name=args.name)
