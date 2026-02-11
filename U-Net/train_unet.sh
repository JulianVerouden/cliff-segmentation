#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

echo "$1"
python scripts/u_net/pre_processing.py --dataset_name example_data
# python scripts/u_net/training_loop.py --dataset_name example_data