import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import os

def loss_curve_cm_unet(csv_path="metrics.csv"):
    # Load with no header
    df = pd.read_csv(csv_path, header=None)

    df.columns = [
    "epoch","step","test_Accuracy","test_F1","test_Precision","test_Recall","test_mIoU","train_F1","train_OA","train_loss_epoch","train_loss_step","train_mIoU","val_F1","val_OA","val_loss","val_mIoU"
    ]

    # -----------------------------------------
    # FIX: Replace empty strings with NaN
    # -----------------------------------------
    df = df.replace("", np.nan)

    # Convert all numeric columns to floats (safe)
    for col in df.columns:
        if col not in ["epoch", "step"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------------------
    # MERGE THE TWO ROWS PER EPOCH
    # -----------------------------------------
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")

    df_merged = df.groupby("epoch").max().reset_index()
    df_sorted = df_merged.sort_values("epoch")


    print(df_sorted)
    # -----------------------------------------
    # PLOT LOSS
    # -----------------------------------------
    plt.figure(figsize=(8,5))
    plt.plot(range(1, 152), df_sorted["train_loss_epoch"], marker='o', label="Training Loss")
    plt.plot(range(1, 152), df_sorted["val_loss"], marker='s', label="Validation Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def get_euphorbia_metrics(csv_path="test_per_image_metrics.csv"):
    df = pd.read_csv(csv_path)

    # Keep only Euphorbia rows
    df_euph = df[df["class"] == "Euphorbia"].copy()

    # Remove rows where IoU / Precision / Recall is NaN
    df_euph = df_euph.dropna(subset=["IoU", "Precision", "Recall", "OA"])

    mean_IoU = df_euph["IoU"].mean()
    mean_precision = df_euph["Precision"].mean()
    mean_recall = df_euph["Recall"].mean()
    mean_accuracy = df_euph["OA"].mean()

    print("Macro-Averaged Metrics for Euphorbia:")
    print(f"IoU:       {mean_IoU:.4f}")
    print(f"Precision:  {mean_precision:.4f}")
    print(f"Recall:     {mean_recall:.4f}")
    print(f"Accuracy:   {mean_accuracy:.4f}")

def load_mask(path):
    """Loads a binary mask and returns a boolean numpy array."""
    img = Image.open(path).convert("L")
    arr = np.array(img)
    # Treat nonzero as foreground
    return (arr > 0).astype(np.uint8)

def compute_segmentation_metrics_from_folders(
    gt_dir,
    pred_dir,
    eps=1e-8
):
    # ----------------------------------------------------------------------
    # 1. Match files by base name
    # ----------------------------------------------------------------------
    gt_files = sorted(glob(os.path.join(gt_dir, "*.tif")))
    results = []

    for gt_path in gt_files:
        base = os.path.splitext(os.path.basename(gt_path))[0]
        pred_path = os.path.join(pred_dir, base + ".png")

        if not os.path.exists(pred_path):
            print(f"WARNING: Missing prediction for {base}")
            continue

        gt = load_mask(gt_path)
        pred = load_mask(pred_path)

        if gt.shape != pred.shape:
            raise ValueError(f"Shape mismatch: {gt_path} vs {pred_path}")

        results.append((gt, pred))

    if not results:
        raise RuntimeError("No valid image pairs found.")

    # ----------------------------------------------------------------------
    # 2. Compute global pixel-level TP/FP/TN/FN
    # ----------------------------------------------------------------------
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_tn = 0

    empty_tile_correct = []
    fg_iou_list = []
    fg_dice_list = []

    for gt, pred in results:
        p = pred.flatten()
        g = gt.flatten()

        tp = np.sum(p * g)
        fp = np.sum(p * (1 - g))
        fn = np.sum((1 - p) * g)
        tn = np.sum((1 - p) * (1 - g))

        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_tn += tn

        # Foreground tiles only
        if np.sum(g) > 0:
            intersection = tp
            union = tp + fp + fn
            fg_iou_list.append((intersection + eps) / (union + eps))

            denom = np.sum(p) + np.sum(g)
            fg_dice_list.append((2 * intersection + eps) / (denom + eps))
        else:
            # Empty-tile accuracy: pred must be completely empty
            empty_tile_correct.append(1.0 if np.sum(p) == 0 else 0.0)

    # ----------------------------------------------------------------------
    # 3. Compute final global metrics
    # ----------------------------------------------------------------------
    global_intersection = all_tp
    global_union = all_tp + all_fp + all_fn
    global_denom = (all_tp + all_fp) + (all_tp + all_fn)

    global_iou = (global_intersection + eps) / (global_union + eps)
    global_dice = (2 * global_intersection + eps) / (global_denom + eps)
    global_precision = all_tp / (all_tp + all_fp + eps)
    global_recall = all_tp / (all_tp + all_fn + eps)
    global_accuracy = (all_tp + all_tn) / (all_tp + all_tn + all_fp + all_fn + eps)

    fg_iou = float(np.mean(fg_iou_list)) if fg_iou_list else float("nan")
    fg_dice = float(np.mean(fg_dice_list)) if fg_dice_list else float("nan")
    empty_accuracy = float(np.mean(empty_tile_correct)) if empty_tile_correct else float("nan")

    return {
        "global_iou": float(global_iou),
        "global_dice": float(global_dice),
        "global_precision": float(global_precision),
        "global_recall": float(global_recall),
        "global_accuracy": float(global_accuracy),
        "fg_iou": fg_iou,
        "fg_dice": fg_dice,
        "empty_tile_accuracy": empty_accuracy,
        "tp_total": int(all_tp),
        "fp_total": int(all_fp),
        "fn_total": int(all_fn),
        "tn_total": int(all_tn),
        "num_images": len(results)
    }


if __name__ == "__main__":
    gt_dir = r"D:\Documents\GitHub\CM-UNet-main\data\vaihingen\test_masks"
    pred_dir = r"D:\Documents\GitHub\CM-UNet-main\predictions\bin"

    metrics = compute_segmentation_metrics_from_folders(gt_dir, pred_dir)

    print("\n=== GLOBAL METRICS ===")
    for k, v in metrics.items():
        print(f"{k:25s}: {v}")