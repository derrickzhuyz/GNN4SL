#!/bin/bash

# Set the visible GPU(s)
export CUDA_VISIBLE_DEVICES='0,1'

# Set the model path
# train on spider train
MODEL_PATH_0="checkpoints/link_level_model/link_level_model_best.pt" 
# demo
MODEL_PATH_1="checkpoints/link_level_model/sentence_transformer/link_level_model_20241125_072822.pt" 
# train on combined training set for 200 epochs
MODEL_PATH_2="checkpoints/link_level_model/sentence_transformer/link_level_model_combined_20241125_132315.pt" 
# resume training on combined training set for 200 epochs, 400 epochs in total
MODEL_PATH_3="checkpoints/link_level_model/sentence_transformer/link_level_model_combined_20241128_075541_resume_200ep.pt" 


# Define a list of model paths to test
MODEL_PATHS=(
    "${MODEL_PATH_2}"
    "${MODEL_PATH_3}"
)

# Test the link-level model on Spider dev set
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "[INFO] Start testing on Spider dev set with model: ${MODEL_PATH}..."
    python -m gnn.model.link_level_test \
        --model_path "${MODEL_PATH}" \
        --dataset_type "spider" \
        --batch_size 1 \
        --embed_method "sentence_transformer" \
        --in_channels 384
done

# Test the link-level model on BIRD dev set
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "[INFO] Start testing on BIRD dev set with model: ${MODEL_PATH}..."
    python -m gnn.model.link_level_test \
        --model_path "${MODEL_PATH}" \
        --dataset_type "bird" \
        --batch_size 1 \
        --embed_method "sentence_transformer" \
        --in_channels 384
done