import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
from config import make_train_config, TrainConfig

import torch.optim as optim
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

from get_dataset import SegmentationDataset, get_train_transforms
from unet_model import UNetResNet50, UnifiedDiceCELoss
from scripts.helper_scripts.next_available_path import next_available_path

import numpy as np
import os

# Metric computation
def compute_segmentation_metrics(preds, masks, threshold=0.5, eps=1e-8):
    bin_preds = (preds > threshold).float()
    mask_preds = (masks > threshold).float()

    # Dice
    p_all = bin_preds.reshape(-1)
    g_all = mask_preds.reshape(-1)

    tp = (p_all * g_all).sum()
    fp = (p_all * (1 - g_all)).sum()
    fn = ((1 - p_all) * g_all).sum()

    intersect_all = tp
    denom_all = p_all.sum() + g_all.sum()
    global_dice = (2 * intersect_all / (denom_all + eps)).item()

    # IoU
    union_all = (p_all + g_all - p_all * g_all).sum()
    iou = (intersect_all / (union_all + eps)).item()

    # Accuracy
    accuracy = (p_all == g_all).float().mean().item()

    # Recall
    recall = (tp / (tp + fn + eps)).item()

    # Precision
    precision = (tp / (tp + fp + eps)).item()

    return {
        "dice": global_dice,
        "iou": iou,
        "accuracy": accuracy,
        "recall": recall,
        "precision": precision
    }

# Training pipeline
def train(cfg: TrainConfig):    
    # Generate file name suffixes -> <datasetname>_<epochs>_<augmentation_method>_<(Optional) pretraining>
    name_parts = [cfg.dataset_name, str(cfg.num_epochs), str(cfg.augmentation_method.value)]

    if cfg.pretraining != None:
        name_parts.append(cfg.pretraining.stem)
    elif cfg.load_IMAGENET1K_V1:
        name_parts.append("IMAGENET1K_V1")

    name = '_'.join(name_parts)

    # Dataset
    full_dataset = SegmentationDataset(
        cfg.tiles_img_dir,
        cfg.tiles_mask_dir,
        transform=get_train_transforms(augmentation=cfg.augmentation_method)
    )

    # Load model
    train_size = int(cfg.train_split * len(full_dataset))
    val_size   = len(full_dataset) - train_size
    torch.manual_seed(cfg.seed)

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=cfg.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNetResNet50(n_channels=cfg.num_channels, n_classes=cfg.num_classes, pretrained=cfg.load_IMAGENET1K_V1, dropout=cfg.dropout_rate).to(device)

    # Pretrain on specified .pth checkpoint
    if (cfg.pretraining != None):
        state_dict = torch.load(cfg.pretraining, map_location="cpu")
        model.load_state_dict(state_dict)

    model = model.to(device)

    # Loss and optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn   = UnifiedDiceCELoss(dice_weight=cfg.dice_weight, ce_weight=1-cfg.dice_weight)

    # Training
    train_losses, val_losses = train_loop(cfg, model, train_loader, val_loader, device, loss_fn, optimizer)

    # Save Model
    os.makedirs(f"output/checkpoints/{cfg.dataset_name}", exist_ok=True)
    model_path = next_available_path(Path(f"output/checkpoints/{cfg.dataset_name}/checkpoint_{name}.pth"))
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Plot Loss
    plos_loss_curves(cfg, train_losses, val_losses, name)

    # Final validation metrics and threshold sweep
    model_eval(cfg, model, val_loader, device, name)

# Training loop that returns train and validation losses
def train_loop(cfg, model, train_loader, val_loader, device, loss_fn, optimizer):
    train_losses = []
    val_losses = []

    for epoch in range(cfg.num_epochs):
        model.train()
        train_epoch_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            preds = model(images)
            loss  = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

        avg_train_loss = train_epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Loss
        model.eval()
        val_epoch_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                val_epoch_loss += loss_fn(preds, masks).item()

        avg_val_loss = val_epoch_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{cfg.num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

# Create and save validation and train loss curves
def plos_loss_curves(cfg: TrainConfig, train_losses, val_losses, name):
    loss_curve_path = next_available_path(Path(f"output/loss_curves/{cfg.dataset_name}/loss_curves_{name}.png", dpi=300))

    os.makedirs(f"output/loss_curves/{cfg.dataset_name}", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, cfg.num_epochs + 1), train_losses, label="Training Loss", marker='o')
    plt.plot(range(1, cfg.num_epochs + 1), val_losses, label="Validation Loss", marker='s')
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(loss_curve_path)
    plt.show()

def model_eval(cfg, model, val_loader, device, name):
    print("\nRunning final validation metrics...")

    all_preds = []
    all_masks = []

    model.eval()
    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)
            preds = torch.sigmoid(model(images))
            all_preds.append(preds.cpu())
            all_masks.append(masks.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Compute final metrics
    final_metrics = compute_segmentation_metrics(all_preds, all_masks, cfg.segmentation_threshold)

    print("\nFinal Validation Metrics:")
    for k, v in final_metrics.items():
        print(f"{k}: {v:.4f}")

    # Save metrics
    os.makedirs(f"output/model_metrics/{cfg.dataset_name}", exist_ok=True)
    model_metrics_path = next_available_path(Path(f"output/model_metrics/{cfg.dataset_name}/model_metrics_{name}.txt"))

    with open(model_metrics_path, "w") as f:
        for k, v in final_metrics.items():
            f.write(f"{k}: {v:.4f}\n")

    print(f"\nValidation metrics saved to {model_metrics_path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U-Net training loop.")
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Name of the dataset subfolder (e.g. meia_velha)",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    cfg = make_train_config(args.dataset_name)
    train(cfg)

if __name__ == "__main__":
    main()
