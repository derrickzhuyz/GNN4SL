#!/bin/bash

# Set the visible GPU(s)
export CUDA_VISIBLE_DEVICES='7,6'

# embedding method and dimension: api_large, dimension: 3072
EMB_METHOD="api_large"
EMB_DIM=3072

# Set the model path
# train on BIRD training set for 500 epochs
MODEL_PATH_0="checkpoints/link_level_model/api_large/link_level_model_bird_20241129_135722.pt"

# train on combined training set for 500 epochs
MODEL_PATH_1="checkpoints/link_level_model/api_large/link_level_model_combined_20241129_133641.pt"

# Define a list of model paths to test
MODEL_PATHS=(
    # "${MODEL_PATH_0}"
    "${MODEL_PATH_1}"
)

# Test the link-level model on Spider dev set
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "[INFO] Start testing on Spider dev set with model: ${MODEL_PATH}..."
    python -m gnn.model.link_level_test \
        --model_path "${MODEL_PATH}" \
        --dataset_type "spider" \
        --batch_size 1 \
        --embed_method "${EMB_METHOD}" \
        --in_channels "${EMB_DIM}"
done

# Test the link-level model on BIRD dev set
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "[INFO] Start testing on BIRD dev set with model: ${MODEL_PATH}..."
    python -m gnn.model.link_level_test \
        --model_path "${MODEL_PATH}" \
        --dataset_type "bird" \
        --batch_size 1 \
        --embed_method "${EMB_METHOD}" \
        --in_channels "${EMB_DIM}"
done