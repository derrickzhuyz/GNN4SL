#!/bin/bash

# Set the environment variables if needed
# export VARIABLE_NAME=value

# Set the visible GPU(s)
export CUDA_VISIBLE_DEVICES='6,7'

# Train the link-level model
echo "[INFO] Start training of the link-level model..."
python -m gnn.model.link_level_train \
    --dataset_type "bird" \
    --epochs 500 \
    --batch_size 1 \
    --val_ratio 0.1 \
    --val_dataset_type "combined" \
    --lr 1e-3 \
    --embed_method "sentence_transformer" \
    --in_channels 384