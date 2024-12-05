#!/bin/bash

# Set the visible GPU(s)
export CUDA_VISIBLE_DEVICES='5,6'

# embedding method and dimension: sentence_transformer, dimension: 384
EMB_METHOD="sentence_transformer"
EMB_DIM=384

# Set the model path
# train on spider train with 1 epoch
MODEL_PATH_0="checkpoints/link_level_gat/sentence_transformer/gat_train_spider_f1_10ep_hard_neg_samp_2.0_dot_product_1205_1315.pt"

# Define a list of model paths to test
MODEL_PATHS=(
    "${MODEL_PATH_0}"
)

# Test the link-level model on Spider dev set
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "[INFO] Start testing on Spider dev set with model: ${MODEL_PATH}..."
    python -m gnn.model.link_level_test \
        --model_type "gat" \
        --model_path "${MODEL_PATH}" \
        --dataset_type "spider" \
        --batch_size 1 \
        --threshold 0.5 \
        --embed_method "${EMB_METHOD}" \
        --in_channels "${EMB_DIM}" \
        --prediction_method "dot_product"
done

# # Test the link-level model on BIRD dev set
# for MODEL_PATH in "${MODEL_PATHS[@]}"; do
#     echo "[INFO] Start testing on BIRD dev set with model: ${MODEL_PATH}..."
#     python -m gnn.model.link_level_test \
#         --model_path "${MODEL_PATH}" \
#         --dataset_type "bird" \
#         --batch_size 1 \
#         --embed_method "${EMB_METHOD}" \
#         --in_channels "${EMB_DIM}"
# done