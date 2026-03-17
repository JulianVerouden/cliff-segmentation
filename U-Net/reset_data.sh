#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "$1"

# Reset dataset structuring
python train_test_supervision.py --dataset_name example_training_data --reset 