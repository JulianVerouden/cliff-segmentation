import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import argparse
from config import make_train_config, TrainConfig, UseTestSplit

import scripts.u_net.pre_processing as pre_processing
import scripts.train_test_split.image_metadata as image_metadata
import scripts.u_net.training_loop as training_loop
# import scripts.u_net.test_loop as test_loop
import scripts.helper_scripts.next_available_path as next_path

import scripts.train_test_split.create_split as create_split


from scripts.helper_scripts.next_available_path import next_available_path

# Check whether filenames match in two folders, and the amount of files is higher than 0.
def check_file_integrity(dir_a: Path, dir_b: Path) -> bool:
    stems_a = {
        p.stem for p in dir_a.iterdir() if p.is_file()
    }
    stems_b = {
        p.stem for p in dir_b.iterdir() if p.is_file()
    }

    return stems_a == stems_b and len(stems_a) > 0

# When to preprocess: base_dir exists (mask and image), but not the full and tile folders AND filenames match. Otherwise raise exception
# When NOT to preprocess: full and tile folders exist. Raise exception if filenames do not match.
def pre_process_dataset(cfg: TrainConfig):
    base_dir_exists: bool = cfg.base_img_dir.is_dir() and cfg.base_mask_dir.is_dir()
    tile_full_dirs_exist: bool = cfg.image_dir.is_dir() and cfg.mask_dir.is_dir() and cfg.tiles_img_dir.is_dir() and cfg.tiles_mask_dir.is_dir()
    
    if (base_dir_exists):
        if(tile_full_dirs_exist):
            if (not (check_file_integrity(cfg.image_dir, cfg.mask_dir) and check_file_integrity(cfg.tiles_img_dir, cfg.tiles_mask_dir))):
                raise FileNotFoundError(f"There is a mismatch between the mask and image file names, or one of the folders is empty")
            else:
                print("Dataset directory integrity correct.")
        else:
            if(check_file_integrity(cfg.base_img_dir, cfg.base_mask_dir)):
                pre_processing.main(cfg)
            else:
                raise FileNotFoundError(f"There is a mismatch between the mask and image file names, or one of the folders is empty")   
    else:
        raise NotADirectoryError(f"{cfg.base_img_dir} and/or {cfg.base_mask_dir} does not exist. Make sure the dataset is structured as explained in the readme.")
    
def handle_test_split(cfg: TrainConfig):
    # Check whether a valid test directory is given
    if (cfg.use_test_split == UseTestSplit.DIRECTORY):
        print("Use directory for train/test split.")

        if(cfg.test_dataset_dir == None):
          raise ValueError("test_dataset_dir must be provided when use_test_split is DIRECTORY")
        
        if not cfg.test_dataset_dir.exists():
            raise FileNotFoundError(f"Test dataset directory does not exist: {cfg.test_dataset_dir}")

        if not cfg.test_dataset_dir.is_dir():
            raise NotADirectoryError(f"Path is not a directory: {cfg.test_dataset_dir}")
    elif(cfg.use_test_split == UseTestSplit.CSV):
        print("Use csv for train/test split.")

        csv = next_path.find_latest_path(cfg.dataset_dir, f"metadata_{cfg.dataset_name}", "csv")

        if (csv == None):
            print("No CSV found for current dataset, generating new one.")
            image_metadata.main(cfg)
            create_split.main(cfg)
    elif(cfg.use_test_split == UseTestSplit.FORCE):
            print("Force generating new CSV for train/test split.")

            image_metadata.main(cfg)
            create_split.main(cfg)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="U-Net training loop.")
    parser.add_argument(
        "--dataset_name",
        required=True,
        help="Name of the dataset subfolder (e.g. meia_velha)",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        default=False,
        help="Specify whether the model should train from scratch"
    )

    parser.add_argument(
        "--test",
        action="store_true",
        default=False,
        help="Specify whether the model should train from scratch"
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        default=False,
        help="Reset dataset structuring"
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()
    cfg = make_train_config(args.dataset_name)
    
    if (args.reset):
        pre_processing.undo_pre_processing(cfg)
    else:
        pre_process_dataset(cfg)
        handle_test_split(cfg)
        
        if (args.train):
            training_loop.main(cfg)
        if(args.test):
            print("Handle testing")

if __name__ == "__main__":
    main()