import pandas as pd
import numpy as np

# Load CSV
df = pd.read_csv("test_per_image_metrics.csv")

# --------------------------------------------
# 1. Remove all Background entries
# --------------------------------------------
df_euph = df[df["class"] == "Euphorbia"].copy()

# --------------------------------------------
# 2. Remove rows with NaN in IoU (and optionally other metrics)
# --------------------------------------------
# Option A: Drop rows where IoU is NaN only:
df_euph_clean = df_euph.dropna(subset=["IoU"])

# Option B: Drop rows where ANY metric is NaN:
# df_euph_clean = df_euph.dropna(subset=["IoU", "F1", "Recall", "Precision", "OA"])

# --------------------------------------------
# 3. Compute top 5 and bottom 5 by IoU
# --------------------------------------------
df_sorted = df_euph_clean.sort_values("IoU")

bottom_5 = df_sorted.head(5)
top_5 = df_sorted.tail(5)

print("\nBOTTOM 5 (lowest IoU):")
print(bottom_5)

print("\nTOP 5 (highest IoU):")
print(top_5)