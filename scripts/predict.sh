#!/usr/bin/env bash

# Exit on error
set -e

# Go to project root
cd ../

RUN_ID=iprziwi7

python -m src.evaluation.evaluate \
    --run-id "$RUN_ID" \
    --project-root "bccr-ml-course" \
    --checkpoint "best.ckpt" \
