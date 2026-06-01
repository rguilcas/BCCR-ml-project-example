#!/usr/bin/env bash

# Exit on error
set -e

# Go to project root
cd ../

RUN_ID=5d1b623a

python -m src.evaluation.evaluate \
    --run-id "$RUN_ID" \
    --project-root "bccr-ml-course" \
    --checkpoint "best.ckpt" \
