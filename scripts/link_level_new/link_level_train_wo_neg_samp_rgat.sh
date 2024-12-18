#!/bin/bash


# 1. vanilla RGAT model (use bird dataset and f1)
(
export CUDA_VISIBLE_DEVICES='7'
python -m gnn.model.link_level_train \
    --model_type rgat \
    --num_layers 3 \
    --num_heads 4 \
    --num_relations 3 \
    --dataset_type "bird" \
    --epochs 200 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "bird" \
    --lr 1e-3 \
    --embed_method api_large \
    --in_channels 3072 \
    --prediction_method concat_mlp \
    --metric f1
) &


# 2. RGAT model (still use f1 metric, but use combined dataset)
(
export CUDA_VISIBLE_DEVICES='6'
python -m gnn.model.link_level_train \
    --model_type rgat \
    --num_layers 3 \
    --num_heads 4 \
    --num_relations 3 \
    --dataset_type "combined" \
    --epochs 200 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "combined" \
    --lr 1e-3 \
    --embed_method api_large \
    --in_channels 3072 \
    --prediction_method concat_mlp \
    --metric f1
) &


# 3. RGAT model (still use bird dataset, but use auc metric)
(
export CUDA_VISIBLE_DEVICES='5'
python -m gnn.model.link_level_train \
    --model_type rgat \
    --num_layers 3 \
    --num_heads 4 \
    --num_relations 3 \
    --dataset_type "bird" \
    --epochs 200 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "bird" \
    --lr 1e-3 \
    --embed_method api_large \
    --in_channels 3072 \
    --prediction_method concat_mlp \
    --metric auc
) &


# 4. GAT (not RGAT) model (use best setting)
(
export CUDA_VISIBLE_DEVICES='4'
python -m gnn.model.link_level_train \
    --model_type "gat" \
    --num_layers 2 \
    --dataset_type "combined" \
    --epochs 200 \
    --batch_size 1 \
    --threshold 0.5 \
    --val_ratio 0.1 \
    --val_dataset_type "combined" \
    --lr 1e-3 \
    --embed_method "api_large" \
    --in_channels 3072 \
    --prediction_method "concat_mlp" \
    --metric "auc"
)
