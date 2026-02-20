#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "$1"

# Default train/test loop
python train_test_supervision.py --dataset_name example_data --train --test

# Reset dataset structuring
# python train_test_supervision.py --dataset_name example_data --reset 