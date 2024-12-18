#!/bin/bash

# Set the environment variables if needed
# export VARIABLE_NAME=value


# Run experiments: negative sampling, ratio=1.0

# 8. negative sampling: hard, ratio=1.0
(
export CUDA_VISIBLE_DEVICES='4'
python -m gnn.model.link_level_train \
    --model_type "gat" \
    --dataset_type "bird" \
    --epochs 200 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "bird" \
    --lr 1e-3 \
    --embed_method "api_small" \
    --in_channels 1536 \
    --prediction_method "concat_mlp" \
    --negative_sampling \
    --negative_sampling_ratio 1.0 \
    --negative_sampling_method "hard" \
    --metric "f1" 
) &

# 9. negative sampling: random, ratio=1.0
(
export CUDA_VISIBLE_DEVICES='4'
python -m gnn.model.link_level_train \
    --model_type "gat" \
    --dataset_type "bird" \
    --epochs 200 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "bird" \
    --lr 1e-3 \
    --embed_method "api_small" \
    --in_channels 1536 \
    --prediction_method "concat_mlp" \
    --negative_sampling \
    --negative_sampling_ratio 1.0 \
    --negative_sampling_method "random" \
    --metric "f1" 
) &


# Run experiments: negative sampling, ratio=2.0

# 10. negative sampling: hard, ratio=2.0
(
export CUDA_VISIBLE_DEVICES='5'
python -m gnn.model.link_level_train \
    --model_type "gat" \
    --dataset_type "bird" \
    --epochs 200 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "bird" \
    --lr 1e-3 \
    --embed_method "api_small" \
    --in_channels 1536 \
    --prediction_method "concat_mlp" \
    --negative_sampling \
    --negative_sampling_ratio 2.0 \
    --negative_sampling_method "hard" \
    --metric "f1" 
) &

# 11. negative sampling: random, ratio=2.0
(
export CUDA_VISIBLE_DEVICES='5'
python -m gnn.model.link_level_train \
    --model_type "gat" \
    --dataset_type "bird" \
    --epochs 200 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "bird" \
    --lr 1e-3 \
    --embed_method "api_small" \
    --in_channels 1536 \
    --prediction_method "concat_mlp" \
    --negative_sampling \
    --negative_sampling_ratio 2.0 \
    --negative_sampling_method "random" \
    --metric "f1" 
)


