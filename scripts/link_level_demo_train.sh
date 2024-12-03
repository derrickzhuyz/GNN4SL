#!/bin/bash

# Set the environment variables if needed
# export VARIABLE_NAME=value

# Set the visible GPU(s)
export CUDA_VISIBLE_DEVICES='6,7'

# Train the link-level model
echo "[INFO] Start training of the link-level model..."
python -m gnn.model.link_level_train \
    --dataset_type "bird" \
    --epochs 2 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "bird" \
    --lr 1e-3 \
    --embed_method "sentence_transformer" \
    --in_channels 384 \
    --prediction_method "dot_product" \
    --negative_sampling \
    --negative_sampling_ratio 1.0 \
    --negative_sampling_method "hard" \
    --metric "f1"
