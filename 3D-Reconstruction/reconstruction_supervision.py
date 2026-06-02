import argparse

import scripts.split_dataset.create_clusters as create_clusters
import scripts.colmap.reconstruction as reconstruction
from config import ReconstructionConfig, make_config

def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="U-Net training loop.")
  parser.add_argument(
      "--dataset_name",
      required=True,
      help="Name of the dataset subfolder (e.g. meia_velha)",
  )

  parser.add_argument(
      "--colmap_bin",
      required=True,
      help="Location of colmap bin (.exe file)",
  )
  return parser.parse_args()

def main(cfg: ReconstructionConfig) -> None:
  num_clusters = create_clusters.main(cfg)
  reconstruction.main(cfg, num_clusters)

if __name__ == "__main__":
  args = parse_args()
  cfg = make_config(args.dataset_name, args.colmap_bin)

  main(cfg)
