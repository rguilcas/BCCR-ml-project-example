#!/usr/bin/env bash

# Exit on error
set -e

# Go to project root (important!)
cd ../

# Run training
python -m src.training.train \
    --config configs/experiments/default.yml