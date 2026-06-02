#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "$1"

# Default train/test loop
python reconstruction_supervision.py --dataset_name "test_dataset" --colmap_bin "D:\Documents\GitHub\colmap\bin\colmap.exe"