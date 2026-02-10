import os
import numpy as np
from PIL import Image

def stitch_image(
    name,
    meta_dir="meta",
    prob_dir="prob_tiles",
    mask_dir="mask_tiles",
    output_path=""  # new parameter
):
    meta_path = os.path.join(meta_dir, f"{name}_meta.npy")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Meta file not found: {meta_path}")

    meta = np.load(meta_path, allow_pickle=True).item()
    W, H = meta["W"], meta["H"]
    tiles = meta["tiles"]

    # Ensure output folder exists
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    else:
        output_path = ""  # fallback to current folder

    # Initialize accumulators
    prob_acc = np.zeros((H, W), dtype=np.float32)
    prob_count = np.zeros((H, W), dtype=np.float32)
    mask_acc = np.zeros((H, W), dtype=np.float32)
    mask_count = np.zeros((H, W), dtype=np.float32)

    for t in tiles:
        x, y = t["x"], t["y"]
        base = t["tile"].replace(".png", "")

        # Load tile probability and mask
        prob_tile = np.array(Image.open(os.path.join(prob_dir, f"{base}_prob.png"))) / 255.0
        mask_tile = np.array(Image.open(os.path.join(mask_dir, f"{base}_mask.png"))) / 255.0

        h_tile, w_tile = prob_tile.shape

        # Accumulate into the correct location
        prob_acc[y:y+h_tile, x:x+w_tile] += prob_tile[:h_tile, :w_tile]
        prob_count[y:y+h_tile, x:x+w_tile] += 1

        mask_acc[y:y+h_tile, x:x+w_tile] += mask_tile[:h_tile, :w_tile]
        mask_count[y:y+h_tile, x:x+w_tile] += 1

    # Avoid division by zero
    prob_count[prob_count == 0] = 1
    mask_count[mask_count == 0] = 1

    prob_final = prob_acc / prob_count
    mask_final = (mask_acc / mask_count) > 0.5

    # Save stitched images to output_path
    prob_path = os.path.join(output_path, "probs", f"{name}_prob_full.png")
    mask_path = os.path.join(output_path, "masks", f"{name}_mask_full.png") 

    Image.fromarray((prob_final * 255).astype(np.uint8)).save(prob_path)
    Image.fromarray(mask_final.astype(np.uint8) * 255).save(mask_path)

    print(f"Stitched full image saved: {prob_path} and {mask_path}")
