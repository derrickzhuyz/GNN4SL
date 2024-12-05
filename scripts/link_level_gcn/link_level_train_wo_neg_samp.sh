#!/bin/bash

# Run experiments: vanilla, bird-only, mlp as prediction
(
export CUDA_VISIBLE_DEVICES='7'

# 1. vanilla
python -m gnn.model.link_level_train \
    --model_type "gcn" \
    --dataset_type "combined" \
    --epochs 200 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "bird" \
    --lr 1e-3 \
    --embed_method "api_small" \
    --in_channels 1536 \
    --prediction_method "concat_mlp" \
    --metric "f1" &

# 2. train on bird-only
python -m gnn.model.link_level_train \
    --model_type "gcn" \
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
    --metric "f1" &

# 6. dot product as prediction
python -m gnn.model.link_level_train \
    --model_type "gcn" \
    --dataset_type "combined" \
    --epochs 200 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "bird" \
    --lr 1e-3 \
    --embed_method "api_small" \
    --in_channels 1536 \
    --prediction_method "dot_product" \
    --metric "f1" &

# 7. GAT model
python -m gnn.model.link_level_train \
    --model_type "gat" \
    --dataset_type "combined" \
    --epochs 200 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "bird" \
    --lr 1e-3 \
    --embed_method "api_small" \
    --in_channels 1536 \
    --prediction_method "concat_mlp" \
    --metric "f1" &

wait
) &

# Run experiments: api-large, sentence-transformer, AUC metric
(
export CUDA_VISIBLE_DEVICES='6'

# 3. api-large
python -m gnn.model.link_level_train \
    --model_type "gcn" \
    --dataset_type "combined" \
    --epochs 200 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "bird" \
    --lr 1e-3 \
    --embed_method "api_large" \
    --in_channels 3072 \
    --prediction_method "concat_mlp" \
    --metric "f1" &

# 4. sentence-transformer
python -m gnn.model.link_level_train \
    --model_type "gcn" \
    --dataset_type "combined" \
    --epochs 200 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "bird" \
    --lr 1e-3 \
    --embed_method "api_small" \
    --in_channels 1536 \
    --prediction_method "concat_mlp" \
    --metric "f1" &

# 5. AUC metric
python -m gnn.model.link_level_train \
    --model_type "gcn" \
    --dataset_type "combined" \
    --epochs 200 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "bird" \
    --lr 1e-3 \
    --embed_method "api_small" \
    --in_channels 1536 \
    --prediction_method "concat_mlp" \
    --metric "auc" &

wait
) &

# Wait for all processes to complete
wait