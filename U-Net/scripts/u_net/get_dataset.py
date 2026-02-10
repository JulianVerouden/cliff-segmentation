import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config import AugmentationMethod

import cv2
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import numpy as np


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, ordered=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.ordered = ordered

        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform

        # List all files in the image_dir
        self.images = [p for p in self.image_dir.iterdir() if p.is_file()]

        # Build a lookup for masks by stem
        self.masks = {p.stem: p for p in self.mask_dir.iterdir() if p.is_file()}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
            img_path: Path = Path(self.images[idx])
            stem = img_path.stem

            # Find corresponding mask
            mask_path: Path = self.masks.get(stem)
            if mask_path is None:
                raise FileNotFoundError(f"No mask found for image {img_path.name}")

            # Read image (BGR -> RGB)
            image = cv2.imread(str(img_path))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Read mask as grayscale
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

            # Apply augmentation / transforms
            if self.transform:
                transformed = self.transform(image=image, mask=mask)
                image = transformed["image"]
                mask = transformed["mask"]

            # Ensure mask is binary [1, H, W] float tensor
            mask = (mask > 0).float().unsqueeze(0) # [1, H, W]

            return image, mask
    
def get_train_transforms(augmentation: AugmentationMethod):
    transforms = []

    if (augmentation == AugmentationMethod.BADROUSS):
        transforms = transforms_badrouss()
    elif (augmentation == AugmentationMethod.MA):
        transforms = transforms_ma()
    elif (augmentation == AugmentationMethod.MUMUNI):
        transforms = transforms_mumuni()

    # Always normalize + convert to tensor
    transforms += [
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ]
    return A.Compose(transforms)

# Badrouss
def transforms_badrouss():
    transforms = [
        # Random brightness & contrast
        A.RandomBrightnessContrast(
            brightness_limit=(0.0, 0.5),
            contrast_limit=(0.0, 0.5),
            p=0.5
        ),

        # Horizontal shift (width shift range = 0.5)
        A.ShiftScaleRotate(
            shift_limit_x=0.5,
            shift_limit_y=0.0,
            scale_limit=0.0,
            rotate_limit=0,
            p=0.5
        ),

        # Vertical shift (height shift range = 0.5)
        A.ShiftScaleRotate(
            shift_limit_x=0.0,
            shift_limit_y=0.5,
            scale_limit=0.0,
            rotate_limit=0,
            p=0.5
        ),

        # Horizontal flip (50%)
        A.HorizontalFlip(p=0.5),

        # Vertical flip (50%)
        A.VerticalFlip(p=0.5),

        # Hue-Saturation-Value adjustment (20%)
        A.HueSaturationValue(p=0.2),

        # GaussNoise + OneOf distortions (probability = 0.8)
        A.GaussNoise(p=0.8),

        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50),
            A.GridDistortion(num_steps=5),
            A.OpticalDistortion(distort_limit=0.5),
            A.Transpose(),
        ], p=0.8),
    ]

    return transforms

# Mumuni
def transforms_mumuni():
    transforms = [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale=(0.9, 1.1),
            translate_percent=(0.05, 0.05),
            rotate=(-30, 30),
            p=0.5
        ),
        A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, p=0.3),
        A.GaussianBlur(blur_limit=(3,7), p=0.2),
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.ElasticTransform(alpha=1, sigma=50, p=0.2),
            A.GridDistortion(p=0.2)
        ], p=0.3),
    ]
   
    return transforms

# Ma
def transforms_ma():
    transforms = [
        # 1) Random cropping
        A.RandomResizedCrop(
            size=(256, 256),
            scale=(0.7, 1.0),
            ratio=(0.75, 1.33),
            p=1.0
        ),

        # 2) Local shift (Affine requires valid scale/rotate/shear)
        A.Affine(
            translate_percent={"x": 0.1, "y": 0.1},  # ±10% shift
            scale=1.0,              # NO SCALING, but valid
            rotate=0.0,             # NO ROTATION, but valid
            shear=0.0,              # NO SHEAR, but valid
            fit_output=False,
            p=0.8
        ),

        # 3) JPEG-like compression
        A.ImageCompression(
            quality_range=(40, 80),
            p=1.0
        ),
    ]

    return transforms