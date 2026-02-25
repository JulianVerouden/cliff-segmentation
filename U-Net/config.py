from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from enum import Enum
from dataclasses import asdict

import json

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

def _serialize(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    return obj

def _load_train_config(checkpoint: Path) -> Tuple[Dict[str, Any], bool]:
    config_path = checkpoint.with_suffix(".json")

    if not config_path.exists():
        return {}, False

    with config_path.open("r") as f:
        return json.load(f), True
    
@dataclass
class BaseSegmentationConfig:
    # Shared model / tiling parameters
    tile_w: int = 256
    tile_h: int = 256
    num_channels: int = 3
    num_classes: int = 1
    segmentation_threshold: float = 0.5

@dataclass
class InferenceConfig(BaseSegmentationConfig):
    # Inference parameters
    checkpoint_name: str = field(init=False)
    checkpoint: Path = Path(r"output\example_training_data\checkpoints\checkpoint_example_training_data_1_mumuni_IMAGENET1K_V1.pth")
    input_image_dir: Path = Path(r"data\inference_images")
    output_dir: Path = field(init=False)

    tile_dir: Path = Path(r"data\temp\tiles")
    meta_dir: Path = Path(r"data\temp\meta")
    prob_dir: Path = Path(r"data\temp\prob_tiles")
    mask_dir: Path = Path(r"data\temp\mask_tiles")

    def __post_init__(self):
        self.checkpoint_name = self.checkpoint.stem
        self.output_dir = Path(r"output\inference", self.checkpoint_name)

        # Load training config
        train_cfg, found_cfg = _load_train_config(self.checkpoint)

        # Extract shared parameters
        if(found_cfg):
            self.tile_w = train_cfg["tile_w"]
            self.tile_h = train_cfg["tile_h"]
            self.num_channels = train_cfg["num_channels"]
            self.num_classes = train_cfg["num_classes"]
            self.segmentation_threshold = train_cfg["segmentation_threshold"]
        else:
            print("Did not find config file corresponding to checkpoint. Attempting to continue with default parameters from config.")


@dataclass
class TrainConfig(BaseSegmentationConfig):
    # Base directories (dataset-specific)
    # Dataset
    dataset_name: str = ""
    full_name: str = field(init=False)

    dataset_dir: Path       = Path()
    base_img_dir: Path      = Path()
    base_mask_dir: Path     = Path()
    image_dir: Path         = Path()
    mask_dir: Path          = Path()
    tiles_img_dir: Path     = Path()
    tiles_mask_dir: Path    = Path()
    metadata_file: Path     = Path()
    test_split: Path        = Path()

    # Output
    model_metrics_dir: Path = field(init=False)
    model_metrics: Path = field(init=False)
    model_metrics_test: Path = field(init=False)
    top_bottom_dir: Path = field(init=False)
    loss_curves_dir: Path = field(init=False)
    loss_curves: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    checkpoint: Path = field(init=False)

    # Tiling parameters
    tile_threshold: int = 50
    balance_dataset: bool = True # Keep an even amount of tiles with and without the segmentation target.

    # Stats
    stats_file: Optional[Path] = None

    # Train/Test split parameters
    test_split_method: GenerateTestSplit = GenerateTestSplit.RANDOM
    use_test_split: UseTestSplit = UseTestSplit.CSV
    test_dataset_dir: Optional[Path] = None # Required when test_split_method is DataSplit.DIRECTORY
    test_only_checkpoint: Optional[Path] = Path(r"output\example_data\checkpoints\checkpoint_example_data_1_mumuni_IMAGENET1K_V1.pth") # Required when only running test
    random_split_seed: int = 42
    eps_meters: int = 60 # Max distance for clustering during spatial test split generation
    train_fraction: float = 0.85

    # Training parameters
    seed: int = 42
    augmentation_method: AugmentationMethod = AugmentationMethod.MUMUNI
    num_epochs: int = 1
    # Accepts file paths to .pth files. Example -> Path(r"output\checkpoints\test_checkpoint.pth"). Use when you want to load your own checkpoint weights.
    pretraining: Optional[Path] =  None
    load_IMAGENET1K_V1: bool = True
    train_split: float = 0.85
    batch_size: int = 8
    dropout_rate: float = 0.5
    dice_weight: float = 0.7
    learning_rate: float = 1e-4

    # Test parameters
    iou_threshold: float = 0.5
    save_top_bottom: bool = True
    save_metrics: bool = True

    # Post init for file and directory names
    def __post_init__(self):
        name_parts = [
            self.dataset_name,
            str(self.num_epochs),
            str(self.augmentation_method.value),
        ]

        if self.pretraining is not None:
            name_parts.append(self.pretraining.stem)
        elif self.load_IMAGENET1K_V1:
            name_parts.append("IMAGENET1K_V1")

        self.full_name = "_".join(name_parts)
        
        output = Path("output")

        # Output
        self.loss_curves_dir     =output / self.dataset_name / "loss_curves"
        self.model_metrics_dir   =output / self.dataset_name / "model_metrics"
        self.top_bottom_dir      =output / self.dataset_name / "iou_rankings"
        self.checkpoint_dir      =output / self.dataset_name / "checkpoints"

        self.loss_curves         =self.loss_curves_dir / f"loss_curves_{self.full_name}.png"
        self.model_metrics       =self.model_metrics_dir / f"model_metrics_{self.full_name}"
        self.model_metrics_test  =self.model_metrics_dir / f"model_metrics_test_{self.full_name}"
        self.checkpoint          =self.checkpoint_dir / f"checkpoint_{self.full_name}.pth"

    def to_dict(self) -> dict:
        raw = asdict(self)
        return {k: _serialize(v) for k, v in raw.items()}

# Build a TileConfig for a given dataset name.
def make_train_config(dataset_name: str) -> TrainConfig:
    base = Path("data")
    
    return TrainConfig(
        # Dataset
        dataset_name    =dataset_name,
        dataset_dir     =base / dataset_name,
        base_img_dir    =base / dataset_name / "images",
        base_mask_dir   =base / dataset_name / "masks", 
        image_dir       =base / dataset_name / "images" / "full" ,
        mask_dir        =base / dataset_name / "masks" / "full" ,
        tiles_img_dir   =base / dataset_name / "images" / "tiles" ,
        tiles_mask_dir  =base / dataset_name / "masks" / "tiles" ,
        test_split      =base / dataset_name / f"test_split_{dataset_name}.csv",
        stats_file      =base / dataset_name / f"tile_stats_{dataset_name}.csv",
        metadata_file   =base / dataset_name / f"metadata_{dataset_name}.csv",
    )
