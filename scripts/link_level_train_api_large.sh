#!/bin/bash

# Set the environment variables if needed
# export VARIABLE_NAME=value

# Set the visible GPU(s)
export CUDA_VISIBLE_DEVICES='7,6'

# Train the link-level model
echo "[INFO] Start training of the link-level model..."
python -m gnn.model.link_level_train \
    --dataset_type "combined" \
    --epochs 500 \
    --batch_size 1 \
    --val_ratio 0.1 \
    --lr 1e-4 \
    --embed_method "api_large" \
    --in_channels 3072
