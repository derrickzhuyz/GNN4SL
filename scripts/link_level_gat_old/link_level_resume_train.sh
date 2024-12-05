#!/bin/bash

# Set the environment variables if needed
# export VARIABLE_NAME=value

# Set the visible GPU(s)
export CUDA_VISIBLE_DEVICES='3,7'

# Resume training of the link-level model
echo "[INFO] Start resuming training of the link-level model..."
python -m gnn.model.link_level_train \
--dataset_type "combined" \
    --epochs 100 \
    --batch_size 1 \
    --val_ratio 0.1 \
    --val_dataset_type "combined" \
    --lr 1e-4 \
    --embed_method "sentence_transformer" \
    --resume_from checkpoints/link_level_model/sentence_transformer/link_level_model_combined_20241128_075541_resume_200ep.pt