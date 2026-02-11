import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os

from get_dataset import SegmentationDataset, get_train_transforms
from unet_model import UNetResNet50
from training_loop import compute_segmentation_metrics   # if defined in training_loop.py
from torchvision.utils import save_image

def compute_iou(pred, target, threshold):
    pred_bin = (pred > threshold).float()
    target_bin = (target > 0.5).float()

    intersection = (pred_bin * target_bin).sum().item()
    union = pred_bin.sum().item() + target_bin.sum().item() - intersection

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

# Settings
checkpoint_path = r"data\checkpoints\checkpoint0_150_yes_common_meia_velha.pth"
name = "150_yes_common_meia_velha_new"

test_images_path = "data/images/meia_velha/tiles/test"
test_masks_path  = "data/masks/meia_velha/tiles/test"

batch_size = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Dataset and Dataloader
# -----------------------------
# If you don't have a special test-transform, reuse validation transforms:
test_dataset = SegmentationDataset(
    test_images_path,
    test_masks_path,
    transform=get_train_transforms(augmentation=False),
    ordered=True
)


test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Load model
# -----------------------------
model = UNetResNet50(n_channels=3, n_classes=1, pretrained=False).to(device)
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print(f"Loaded checkpoint: {checkpoint_path}")

# -----------------------------
# Gather predictions
# -----------------------------
all_preds = []
all_masks = []
all_images = []
all_filenames = []

with torch.no_grad():
    for i, (images, masks) in enumerate(test_loader):
        images, masks = images.to(device), masks.to(device)

        preds = torch.sigmoid(model(images))

        all_preds.append(preds.cpu())
        all_masks.append(masks.cpu())
        all_images.append(images.cpu())
        all_filenames.append(test_dataset.images[i])

all_preds = torch.cat(all_preds, dim=0)
all_masks = torch.cat(all_masks, dim=0)
all_images = torch.cat(all_images, dim=0)

ious = []
for i in range(len(all_preds)):
    iou = compute_iou(all_preds[i], all_masks[i], threshold=0.5)
    ious.append((i, iou))

# FIND BOTTOM AND TOP PERFORMING TILES BASED ON IOU
# Identify images where model predicts at least 1 pixel as foreground
positive_predictions = []

for i in range(len(all_preds)):
    pred_binary = (all_preds[i] > 0.5).float()
    if pred_binary.sum().item() > 0:     # at least 1 predicted pixel
        # pair: (index, IoU)
        positive_predictions.append((i, ious[i][1]))

# Sort by IoU
ious_sorted = sorted(ious, key=lambda x: x[1])
bottom5 = ious_sorted[:5]
top5 = ious_sorted[-5:]

# Sort by IoU descending
positive_predictions_sorted = sorted(positive_predictions, key=lambda x: x[1], reverse=True)

top5_pred_fg = positive_predictions_sorted[:5]

print("\nTop-5 IoU:")
for idx, iou in top5:
    fname = os.path.basename(all_filenames[idx])
    print(f"{fname} — IoU={iou:.4f}")

print("\nBottom-5 IoU:")
for idx, iou in bottom5:
    fname = os.path.basename(all_filenames[idx])
    print(f"{fname} — IoU={iou:.4f}")

print("\nTop-5 images with at least 1 predicted positive pixel (sorted by IoU):")
for idx, iou in top5_pred_fg:
    fname = os.path.basename(all_filenames[idx])
    print(f"{fname} — IoU={iou:.4f}")

save_dir_top = "data/iou_rankings/top5"
save_dir_bottom = "data/iou_rankings/bottom5"
save_dir_top_fg = "data/iou_rankings/top5fg"

os.makedirs(save_dir_top, exist_ok=True)
os.makedirs(save_dir_bottom, exist_ok=True)
os.makedirs(save_dir_top_fg, exist_ok=True)

def save_triplet(idx, iou, folder):
    img = all_images[idx]
    mask = all_masks[idx]
    pred = (all_preds[idx] > 0.5).float()

    save_image(img, os.path.join(folder, f"img_{idx}_iou_{iou:.4f}.png"))
    save_image(mask, os.path.join(folder, f"img_{idx}_mask.png"))
    save_image(pred, os.path.join(folder, f"img_{idx}_pred.png"))

# Save top 5
for idx, iou in top5:
    save_triplet(idx, iou, save_dir_top)

# Save bottom 5
for idx, iou in bottom5:
    save_triplet(idx, iou, save_dir_bottom)

for idx, iou in top5_pred_fg:
    save_triplet(idx, iou, save_dir_top_fg)
    

print("\nSaved top-5 and bottom-5 IoU-ranked images.")

# -----------------------------
# Final metrics using best threshold
# -----------------------------
final_metrics = compute_segmentation_metrics(all_preds, all_masks, threshold=0.5)

print("\nTest Metrics:")
for k, v in final_metrics.items():
    print(f"{k}: {v:.4f}")

# -----------------------------
# Save metrics
# -----------------------------
os.makedirs("data/model_metrics", exist_ok=True)
save_path = f"data/model_metrics/model_metrics_test_{name}.txt"

with open(save_path, "w") as f:
    for k, v in final_metrics.items():
        f.write(f"{k}: {v:.4f}\n")

print(f"\nSaved metrics to: {save_path}")
