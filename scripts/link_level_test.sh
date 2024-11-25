#!/bin/bash

# Set the visible GPU(s)
export CUDA_VISIBLE_DEVICES='1,2'

# Set the model path
MODEL_PATH_0="checkpoints/link_level_model/link_level_model_best.pt" # train on spider train
MODEL_PATH_1="checkpoints/link_level_model/sentence_transformer/link_level_model_20241125_072822.pt" # demo

# Test the link-level model
echo "Start testing of the link-level model..."
python -m gnn.model.link_level_test \
    --model_path "${MODEL_PATH_0}"