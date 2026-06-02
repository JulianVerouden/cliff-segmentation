from pathlib import Path

from dataclasses import dataclass

@dataclass
class ReconstructionConfig:
  dataset_name: str = ""

  dataset_dir: Path = Path()
  dataset_img_dir: Path = Path()
  dataset_mask_dir: Path = Path()
  metadata_csv: Path = Path()
  metadata_txt: Path = Path()
  dataset_output: Path = Path()
  cluster_plot: Path = Path()
  cluster_point_clouds: Path = Path()

  # Dataset clustering
  force_clustering: bool = False
  plot_clusters: bool = True
  cluster_size: int = 300 # cluster_size of 0 to skip clustering

  # Colmap
  colmap_bin: Path = Path() 
  colmap_data_path: Path = Path("colmap_data")
  database_path: Path = Path(colmap_data_path, r"database.db")
  colmap_image_path: Path = Path(colmap_data_path, r"images")
  colmap_temp_image_path: Path = Path(colmap_data_path, r"images_temp")
  colmap_mask_path: Path = Path(colmap_data_path, r"masks")
  sparse_path: Path = Path(colmap_data_path, r"sparse")
  sparse_aligned_path: Path = Path(colmap_data_path, r"sparse_aligned")
  dense_path: Path = Path(colmap_data_path, r"dense")
  dense_path_masks: Path = Path(colmap_data_path, r"dense_masks")
  undistorted_path: Path = Path(dense_path, r"images")

  # extract_features
  max_img_size_feature_extraction: int = 1600
  max_num_features: int = 2048

  # match_features_sequential
  sequence_overlap: int = 6

  # match_features_spatial
  max_num_neighbors: int = 20

  # Dense reconstruction
  cache_size: int = 8
  max_img_size_dense: int = 1600

  # patch_match_stereo
  geom_consistency: bool = False
  num_iterations: int = 3

  # stereo_fusion
  num_threads: int = 4
  use_cache: bool = True
  max_num_pixels: int = 1500000
  

def make_config(dataset_name: str, colmap_bin: str) -> ReconstructionConfig:
  base = Path("data")
  output = Path("output")

  return ReconstructionConfig(
    colmap_bin        =Path(colmap_bin),
    dataset_name      =dataset_name,
    dataset_dir       =base / dataset_name,
    dataset_img_dir   =base / dataset_name / "images",
    dataset_mask_dir  =base / dataset_name / "masks",
    metadata_csv      =base / dataset_name / f"metadata_{dataset_name}.csv",
    metadata_txt      =base / dataset_name / f"metadata_{dataset_name}.txt",
    dataset_output    =output / dataset_name,
    cluster_plot      =output / dataset_name / f"cluster_plot_{dataset_name}.png",
    cluster_point_clouds =output / dataset_name / f"cluster_point_clouds",
  )