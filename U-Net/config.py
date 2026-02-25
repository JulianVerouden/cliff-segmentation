from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from enum import Enum

class AugmentationMethod(Enum):
    NONE = "none"
    MUMUNI = "mumuni"
    BADROUSS = "badrouss"
    MA = "ma"
    
class GenerateTestSplit(Enum):
    RANDOM = "random"           # Create a train/test split by selecting random images for either set
    SPATIAL = "spatial"         # Create a train/test split by clustering images based on their XY coordinates. Split works better with larger datasets.
                                # Code in image_metadata.py might require changes based on how metadata is structured in your dataset.

class UseTestSplit(Enum):
    FORCE = "force"             # Force a new dataset split, even if one already exists for the current dataset.
    DIRECTORY = "directory"     # Use images in dataset folder for training/validation. Use images in a separate directory for 
    CSV = "CSV"                 # Use train/test split from csv, if none exists for the current dataset, create a new one.
    NONE = "none"               # Use all images for training

class DatasetMode(Enum):
    TRAIN = "train"
    TEST = "test"

@dataclass
class TrainConfig:
    # Base directories (dataset-specific)
    dataset_name: str
    dataset_dir: Path
    base_img_dir: Path
    base_mask_dir: Path
    image_dir: Path
    mask_dir: Path
    tiles_img_dir: Path
    tiles_mask_dir: Path
    metadata_file: Path
    test_split: Path

    # Tiling parameters
    tile_w: int = 256
    tile_h: int = 256
    tile_threshold: int = 50
    balance_dataset: bool = True # Keep an even amount of tiles with and without the segmentation target.

    # Stats
    stats_file: Optional[Path] = None

    # Training parameters
    segmentation_threshold: float = 0.5
    seed: int = 42
    augmentation_method: AugmentationMethod = AugmentationMethod.MUMUNI
    num_epochs: int = 1
    num_classes: int = 1
    num_channels: int = 3
    # Accepts file paths to .pth files. Example -> Path(r"output\checkpoints\test_checkpoint.pth"). Use when you want to load your own checkpoint weights.
    pretraining: Optional[Path] =  Path(r"output\checkpoints\test_checkpoint.pth")
    load_IMAGENET1K_V1: bool = True
    train_split: float = 0.85
    batch_size: int = 8
    dropout_rate: float = 0.5
    dice_weight: float = 0.7
    learning_rate: float = 1e-4

    # Test parameters
    test_split_method: GenerateTestSplit = GenerateTestSplit.RANDOM
    use_test_split: UseTestSplit = UseTestSplit.CSV
    test_dataset_dir: Optional[Path] = None # Required when test_split_method is DataSplit.DIRECTORY
    checkpoint_path: Optional[Path] = Path(r"output\checkpoints\example_data\checkpoint_example_data_1_mumuni_IMAGENET1K_V1.pth")
    random_split_seed: int = 42
    eps_meters: int = 60 # Max distance for clustering during spatial test split generation
    train_fraction: float = 0.85

# Build a TileConfig for a given dataset name.
def make_train_config(dataset_name: str) -> TrainConfig:
    base = Path("data")

    return TrainConfig(
        dataset_name = dataset_name,
        dataset_dir     =base / dataset_name,
        base_img_dir    =base / dataset_name / "images",
        base_mask_dir   =base / dataset_name / "masks", 
        image_dir       =base / dataset_name / "images" / "full" ,
        mask_dir        =base / dataset_name / "masks" / "full" ,
        tiles_img_dir   =base / dataset_name / "images" / "tiles" ,
        tiles_mask_dir  =base / dataset_name / "masks" / "tiles" ,
        stats_file      =base / dataset_name / f"tile_stats_{dataset_name}.csv",
        metadata_file   =base / dataset_name / f"metadata_{dataset_name}.csv",
        test_split      =base / dataset_name / f"test_split_{dataset_name}.csv",
    )
