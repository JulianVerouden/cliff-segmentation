import sys
from pathlib import Path
import os

from pycolmap import Reconstruction # type: ignore
import pandas as pd
import shutil

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import ReconstructionConfig
import scripts.colmap.colmap as cm
import scripts.combine_point_clouds as combine_pc

def filter_mvs_cfgs(cfg: ReconstructionConfig, current_cluster: int):
    patch_cfg_path = Path(cfg.colmap_data_path, "patch-match.cfg")
    fusion_cfg_path = Path(cfg.colmap_data_path, "fusion.cfg")

    patch_cfg_path_image = Path(cfg.dense_path, "stereo", "patch-match.cfg")
    patch_cfg_path_mask = Path(cfg.dense_path_masks, "stereo", "patch-match.cfg")
    fusion_cfg_path_image = Path(cfg.dense_path, "stereo", "fusion.cfg")
    fusion_cfg_path_mask = Path(cfg.dense_path_masks, "stereo", "fusion.cfg")

    if not patch_cfg_path.exists():
        raise FileNotFoundError(f"Missing {patch_cfg_path}")
    if not fusion_cfg_path.exists():
        raise FileNotFoundError(f"Missing {fusion_cfg_path}")

    # Load cluster assignment
    metadata = pd.read_csv(cfg.metadata_csv)[["filename", "cluster"]]
    cluster_images = set(
        Path(f).name for f in metadata.loc[
            metadata["cluster"] == current_cluster,
            "filename"
        ].values
    )

    # Filter patch-match.cfg
    with open(patch_cfg_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    filtered_patch = []
    i = 0
    kept_patch = 0

    while i < len(lines) - 1:
      img = lines[i]
      params = lines[i + 1]

      if img in cluster_images:
          filtered_patch.append(img)
          filtered_patch.append(params)
          kept_patch += 1

      i += 2

    with open(patch_cfg_path_image, "w") as f:
      for l in filtered_patch:
          f.write(l + "\n")
    
    with open(patch_cfg_path_mask, "w") as f:
      for l in filtered_patch:
          f.write(l + "\n")

    # Filter fusion.cfg
    with open(fusion_cfg_path, "r") as f:
        fusion_images = [l.strip() for l in f if l.strip()]

    filtered_fusion = [img for img in fusion_images if img in cluster_images]

    if len(filtered_fusion) == 0:
        raise ValueError("fusion.cfg has 0 images after filtering")

    with open(fusion_cfg_path_image, "w") as f:
        for img in filtered_fusion:
            f.write(img + "\n")

    with open(fusion_cfg_path_mask, "w") as f:
        for img in filtered_fusion:
            f.write(img + "\n")

    print(f"[cluster {current_cluster}]")
    print(f"  patch-match.cfg: {kept_patch} images")
    print(f"  fusion.cfg: {len(filtered_fusion)} images")

def move_dataset_to_colmap(cfg: ReconstructionConfig):
  img_dir = Path(cfg.dataset_img_dir)
  mask_dir = Path(cfg.dataset_mask_dir)

  for file_name in os.listdir(img_dir):
    stem, img_ext = os.path.splitext(Path(file_name))

    # Move images (any extension)
    for img_path in img_dir.glob(f"{stem}.*"):
        if img_path.is_file():
            shutil.move(str(img_path), cfg.colmap_image_path)

    # Move masks (any extension)
    for mask_path in mask_dir.glob(f"{stem}.*"):
        if mask_path.is_file():
          root, _ = os.path.splitext(mask_path)
          new_path = root + img_ext

          os.rename(str(mask_path), new_path)

          shutil.move(new_path, cfg.colmap_mask_path)

def move_masks_to_dataset(cfg: ReconstructionConfig):
  mask_dir = Path(cfg.dataset_mask_dir)

  cluster_mask_dir = cfg.colmap_image_path

  masks = os.listdir(cluster_mask_dir)

  for mask in masks:
     path = Path(cluster_mask_dir, mask)
     shutil.move(path, mask_dir)

  os.rename(f"{cfg.dense_path}/fused.ply", f"{cfg.dense_path}/fused_mask.ply")
  os.rename(f"{cfg.dense_path}/fused.ply.vis", f"{cfg.dense_path}/fused_mask.ply.vis")

def move_masks_for_reconstruction(cfg: ReconstructionConfig):
  # Colmap uses sparse/images for undistortion, rename masks to images
  os.rename(cfg.colmap_image_path, cfg.colmap_temp_image_path)
  os.rename(cfg.colmap_mask_path, cfg.colmap_image_path)

  shutil.copy(f"{cfg.dense_path}/fused.ply", f"{cfg.dense_path}/fused_rgb.ply")
  shutil.copy(f"{cfg.dense_path}/fused.ply.vis", f"{cfg.dense_path}/fused_rgb.ply.vis")

  # Remove undistorted images so they can be replaced with masks
  shutil.rmtree(cfg.undistorted_path)

def move_maps_to_dense_masks(cfg: ReconstructionConfig):
  shutil.move(Path(cfg.dense_path, "stereo", "depth_maps"), Path(cfg.dense_path_masks, "stereo"))
  shutil.move(Path(cfg.dense_path, "stereo", "normal_maps"), Path(cfg.dense_path_masks, "stereo"))

def reset_stereo(cfg: ReconstructionConfig):
  shutil.rmtree(Path(cfg.dense_path_masks, "stereo"))
  shutil.rmtree(Path(cfg.dense_path, "stereo"))

  os.makedirs(Path(cfg.dense_path, "stereo", "depth_maps"))
  os.makedirs(Path(cfg.dense_path, "stereo", "normal_maps"))
  os.makedirs(Path(cfg.dense_path, "stereo", "consistency_graphs"))
  
  os.makedirs(Path(cfg.dense_path_masks, "stereo", "consistency_graphs"))

def reconstruction(cfg: ReconstructionConfig, num_clusters: int):
  colmap = cm.Colmap(cfg)

  os.makedirs(cfg.colmap_image_path, exist_ok=True)
  os.makedirs(cfg.colmap_mask_path, exist_ok=True)
  os.makedirs(Path(f"{cfg.sparse_aligned_path}", "0"), exist_ok=True)

  # Move all images in dataset to colmap_data folder for sparse reconstruction
  move_dataset_to_colmap(cfg)
  all_images = os.listdir(cfg.dataset_img_dir)

  for img in all_images:
    shutil.move(Path(cfg.dataset_img_dir, img), cfg.colmap_image_path)

  # Global sparse reconstruction
  colmap.extract_features()
  colmap.match_features_exhaustive()
  colmap.sparse_reconstruction(cfg.sparse_path)
  colmap.model_aligner()

  # Global undistortion
  colmap.image_undistortion(cfg.colmap_image_path, Path(f"{cfg.sparse_path}", "0"), cfg.dense_path)
  colmap.image_undistortion(cfg.colmap_mask_path, Path(f"{cfg.sparse_path}", "0"), cfg.dense_path_masks)
  shutil.move(Path(cfg.dense_path, "stereo", "fusion.cfg"), cfg.colmap_data_path)
  shutil.move(Path(cfg.dense_path, "stereo", "patch-match.cfg"), cfg.colmap_data_path)

  # Loop through clusters for dense reconstruction
  for cluster_index in range(num_clusters):
    filter_mvs_cfgs(cfg, cluster_index)

    # Dense reconstruction RGB
    colmap.dense_stereo(cfg.dense_path)
    colmap.stereo_fusion(cfg.dense_path, f"{cfg.dense_path}/fused.ply")

    # Dense reconstruction Masks
    # Move masks
    move_maps_to_dense_masks(cfg)
    colmap.stereo_fusion(cfg.dense_path_masks, f"{cfg.dense_path_masks}/fused.ply")

    # Combine rgb and mask point clouds
    combine_pc.add_mask_channel(cfg, f"masked_point_cloud_{cluster_index}")
    reset_stereo(cfg)

    # TODO: Add logging/keep track of where you are in the point cloud creation
    # TODO: Add cluster as point cloud scalar field
    # TODO: symlink instead of moving files
    # TODO: Optional verbosity
    # TODO: Make clustering optional
  combine_pc.combine_clusters(cfg)

def main(cfg: ReconstructionConfig, num_clusters: int):
  reconstruction(cfg, num_clusters)