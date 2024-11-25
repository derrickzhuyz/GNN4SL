#!/bin/bash

# Set the visible GPU(s)
export CUDA_VISIBLE_DEVICES='0,1'

# Set the model path
MODEL_PATH_0="checkpoints/link_level_model/link_level_model_best.pt" # train on spider train
MODEL_PATH_1="checkpoints/link_level_model/sentence_transformer/link_level_model_20241125_072822.pt" # demo

# Test the link-level model on Spider dev set
echo "[INFO] Start testing on Spider dev set..."
python -m gnn.model.link_level_test \
    --model_path "${MODEL_PATH_0}" \
    --dataset_type "spider" \
    --batch_size 16 \
    --embed_method "sentence_transformer"

# Test the link-level model on BIRD dev set
echo "[INFO] Start testing on BIRD dev set..."
python -m gnn.model.link_level_test \
    --model_path "${MODEL_PATH_0}" \
    --dataset_type "bird" \
    --batch_size 16 \
    --embed_method "sentence_transformer"