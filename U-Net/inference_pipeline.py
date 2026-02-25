import sys
import os
import shutil
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.inference.tile_image import tile_image
from scripts.inference.run_inference_on_tiles import run_inference_on_tiles
from scripts.inference.stitch_tiles import stitch_image

import torch
import torchvision.transforms as T
from scripts.u_net.unet_model import UNetResNet50
from config import InferenceConfig, make_train_config

def run_inference(cfg: InferenceConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if (not cfg.checkpoint.exists()):
        raise FileNotFoundError(f"{cfg.checkpoint} not found. Please make sure it exists and check whether inference_checkpoint is correctly specified in config.py.")
    
    if(cfg.checkpoint.suffix != ".pth"):
        raise ValueError(f"{cfg.checkpoint} is not a .pth file. Make sure inference_checkpoint in config.py points to a valid .pth file.")
    
    image_folder = cfg.input_image_dir

    # Load model
    model = UNetResNet50(n_channels=cfg.num_channels, n_classes=cfg.num_classes).to(device)
    model.load_state_dict(torch.load(cfg.checkpoint, map_location=device))
    model.eval()

    print(f"Loaded model from {cfg.checkpoint}")

    # Process all images
    image_files = [f for f in os.listdir(image_folder)
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff"))]

    print(f"Found {len(image_files)} images.")

    for img_file in image_files:
        image_path = os.path.join(image_folder, img_file)
        name = os.path.splitext(img_file)[0]

        print(f"\nProcessing {img_file} ...")

        tile_image(cfg, image_path)
        run_inference_on_tiles(cfg, model)
        stitch_image(cfg, name)

        print(f"Finished {img_file}")

        # Clear temporary folders
        for folder in [cfg.tile_dir, cfg.meta_dir, cfg.prob_dir, cfg.mask_dir]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
                os.makedirs(folder)

    print("\nAll images processed. Pipeline complete.")

def main() -> None:
    cfg = InferenceConfig()
    
    run_inference(cfg)

if __name__ == "__main__":
    main()