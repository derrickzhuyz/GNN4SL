#!/bin/bash

# Set the visible GPU(s)
export CUDA_VISIBLE_DEVICES='5,6'

# embedding method and dimension: sentence_transformer, dimension: 384
EMB_METHOD="api_small"
EMB_DIM=1536

# Set the model path
# train on spider train with 1 epoch
# MODEL_PATH_0="checkpoints/link_level_gat_12.10/api_small/gat_train_bird_f1_200ep_no_neg_samp_dot_product_1207_0723.pt"
MODEL_PATH_1="checkpoints/link_level_gcn_12.10/api_small/gcn_train_bird_f1_200ep_no_neg_samp_dot_product_1207_0711.pt"

# Define a list of model paths to test
MODEL_PATHS=(
    # "${MODEL_PATH_0}"
    "${MODEL_PATH_1}"
)

# Test the link-level model on Spider dev set

# Test the link-level model on BIRD dev set
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "[INFO] Start testing on BIRD dev set with model: ${MODEL_PATH}..."
    python -m gnn.model.link_level_test \
        --model_type "gcn" \
        --model_path "${MODEL_PATH}" \
        --dataset_type "bird" \
        --batch_size 1 \
        --embed_method "${EMB_METHOD}" \
        --in_channels "${EMB_DIM}" \
        --prediction_method "dot_product"
done