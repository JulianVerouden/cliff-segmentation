from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum

class AugmentationMethod(Enum):
    NONE = "none"
    MUMUNI = "mumuni"
    BADROUSS = "badrouss"
    MA = "ma"
    
@dataclass
class TrainConfig:
    # Base directories (dataset-specific)
    dataset_name: str
    base_img_dir: Path
    base_mask_dir: Path
    image_dir: Path
    mask_dir: Path
    tiles_img_dir: Path
    tiles_mask_dir: Path

    # Tiling parameters
    tile_w: int = 256
    tile_h: int = 256
    tile_threshold: int = 50

    # Stats
    stats_file: Optional[Path] = None

    # Training parameters
    segmentation_threshold: float = 0.5
    seed: int = 42
    augmentation_method: AugmentationMethod = AugmentationMethod.MUMUNI
    num_epochs: int = 150
    # Accepts file paths to .pth files. Example -> Path("U-Net\output\checkpoints\test_checkpoint.pth")
    pre_training: Optional[Path] = None
    train_split: float = 0.85
    batch_size: int = 8
    dropout_rate: float = 0.5
    dice_weight: float = 0.7
    learning_rate: float = 1e-4

def make_train_config(dataset_name: str) -> TrainConfig:
    """
    Build a TileConfig for a given dataset name.
    """
    base = Path("data")

    return TrainConfig(
        dataset_name = dataset_name,
        base_img_dir    =base / "images" / dataset_name,
        base_mask_dir   =base / "masks" / dataset_name, 
        image_dir       =base / "images" / dataset_name / "full" ,
        mask_dir        =base / "masks" / dataset_name / "full" ,
        tiles_img_dir   =base / "images" / dataset_name / "tiles" ,
        tiles_mask_dir  =base / "masks" / dataset_name / "tiles" ,
        stats_file      =base / f"tile_stats_{dataset_name}.csv",
    )
