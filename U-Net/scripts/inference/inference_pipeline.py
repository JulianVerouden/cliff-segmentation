import sys
import os
import shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tile_image import tile_image
from run_inference_on_tiles import run_inference_on_tiles
from stitch_tiles import stitch_image

import torch
import torchvision.transforms as T
from unet_model import UNetResNet50

# -------------------------
# Settings
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = r"data\checkpoints\checkpoint0_150_ce_common_meia_velha.pth"
image_folder = r"data\images\test"
output_path = r"data\images\test"
# image_folder = r"F:\Files\Study\Thesis\DatasetsThesis\Meia Velha\Route 3\RGB\Main dataset"  # folder with all images
# output_path = r"F:\Files\Study\Thesis\OutputThesis\meia_velha"

# Temporary folders
tile_dir = r"temp\tiles"
meta_dir = r"temp\meta"
prob_dir = r"temp\prob_tiles"
mask_dir = r"temp\mask_tiles"

tile_size = 256

# -------------------------
# Load model
# -------------------------
model = UNetResNet50(n_channels=3, n_classes=1).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()
print(f"Loaded model from {model_path}")

# -------------------------
# Transforms
# -------------------------
inference_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# -------------------------
# Process all images
# -------------------------
image_files = [f for f in os.listdir(image_folder)
               if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]

print(f"Found {len(image_files)} images.")

for img_file in image_files:
    image_path = os.path.join(image_folder, img_file)
    name = os.path.splitext(img_file)[0]

    print(f"\nProcessing {img_file} ...")

    # Step 1: Tile
    tile_image(
        image_path,
        out_dir=tile_dir,
        meta_dir=meta_dir,
        tile_w=tile_size,
        tile_h=tile_size,
    )

    # Step 2: Inference on tiles
    run_inference_on_tiles(
        tile_dir=tile_dir,
        out_prob_dir=prob_dir,
        out_mask_dir=mask_dir,
        model=model
    )

    # Step 3: Stitch back
    stitch_image(
        name=name,
        meta_dir=meta_dir,
        prob_dir=prob_dir,
        mask_dir=mask_dir,
        output_path=output_path
    )

    print(f"Finished {img_file}")

    # -------------------------
    # Clear temporary folders
    # -------------------------
    for folder in [tile_dir, meta_dir, prob_dir, mask_dir]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
            os.makedirs(folder)

print("\nAll images processed. Pipeline complete.")
