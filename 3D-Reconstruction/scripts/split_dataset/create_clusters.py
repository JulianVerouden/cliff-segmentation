import sys
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from config import ReconstructionConfig
import scripts.split_dataset.collect_exif as collect_exif

# Create metadata_csv where the images are clustered based on their longitude and latitude
def create_clusters(cfg: ReconstructionConfig) -> int:
  metadata = pd.read_csv(cfg.metadata_csv)
  positions = metadata[["latitude", "longitude"]]

  num_chunks = max(1, len(positions) // cfg.cluster_size)
  kmeans = KMeans(n_clusters=num_chunks, random_state=42)
  labels = kmeans.fit_predict(positions)

  metadata['cluster'] = labels

  if (cfg.plot_clusters):
    plot_split(metadata, cfg)

  metadata.to_csv(cfg.metadata_csv)

  return num_chunks

# Create a metadata file where every image is in the same cluster
def no_clusters(cfg: ReconstructionConfig) -> int:
  metadata = pd.read_csv(cfg.metadata_csv)

  metadata['cluster'] = 0

  metadata.to_csv(cfg.metadata_csv)

  return 1

# Plot clusters
def plot_split(df, cfg: ReconstructionConfig):
    plt.scatter(
        df["longitude"],
        df["latitude"],
        c=df["cluster"],
        s=10,
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Clusters")
    
    os.makedirs(cfg.dataset_output, exist_ok=True)
    plt.savefig(cfg.cluster_plot)

def main(cfg: ReconstructionConfig) -> int:
  if(not os.path.exists(cfg.metadata_csv) or not os.path.exists(cfg.metadata_txt) or cfg.force_clustering):
    print("Collecting exif data")
    collect_exif.main(cfg)
  else:
    print("Metadata found, continuing with cluster creation")

  num_clusters = 0

  if cfg.cluster_size > 0:
    print("Creating clusters")
    num_clusters = create_clusters(cfg)
  elif cfg.cluster_size == 0:
    print("Continuing without clustering")
    num_clusters = no_clusters(cfg)


  return num_clusters
