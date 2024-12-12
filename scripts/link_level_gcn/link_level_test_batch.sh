#!/bin/bash

# Set the visible GPU(s)
export CUDA_VISIBLE_DEVICES='5,6'

# embedding method, model type, dimension
EMB_METHOD="api_small"
MODEL_TYPE="gat"
SUFFIX="_12.10"


if [ "${EMB_METHOD}" == "api_small" ]; then
    EMB_DIM=1536
elif [ "${EMB_METHOD}" == "api_large" ]; then
    EMB_DIM=3072
elif [ "${EMB_METHOD}" == "sentence_transformer" ]; then
    EMB_DIM=384
elif [ "${EMB_METHOD}" == "bert" ]; then
    EMB_DIM=768
else
    echo "[ERROR] Invalid embedding method: ${EMB_METHOD}"
    exit 1
fi

# Set the model directory and test all models in the directory
MODEL_DIR="checkpoints/link_level_${MODEL_TYPE}${SUFFIX}/${EMB_METHOD}"
MODEL_PATHS=()

# Find all .pt files in the specified directory and add them to MODEL_PATHS
while IFS= read -r file; do
    MODEL_PATHS+=("$file")
done < <(find "${MODEL_DIR}" -name "*.pt")

# Check if any models were found
if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "[ERROR] No .pt files found in ${MODEL_DIR}"
    exit 1
fi

# Print found models
echo "[INFO] Found ${#MODEL_PATHS[@]} model(s):"
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "  - ${MODEL_PATH}"
done

# Test the link-level model on Spider dev set
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "[INFO] Start testing on Spider dev set with model: ${MODEL_PATH}..."
    python -m gnn.model.link_level_test \
        --model_type "${MODEL_TYPE}" \
        --model_path "${MODEL_PATH}" \
        --dataset_type "spider" \
        --batch_size 1 \
        --threshold 0.5 \
        --embed_method "${EMB_METHOD}" \
        --in_channels "${EMB_DIM}" \
        --prediction_method "concat_mlp"
done

# Test the link-level model on BIRD dev set
for MODEL_PATH in "${MODEL_PATHS[@]}"; do
    echo "[INFO] Start testing on BIRD dev set with model: ${MODEL_PATH}..."
    python -m gnn.model.link_level_test \
        --model_type "${MODEL_TYPE}" \
        --model_path "${MODEL_PATH}" \
        --dataset_type "bird" \
        --batch_size 1 \
        --embed_method "${EMB_METHOD}" \
        --in_channels "${EMB_DIM}" \
        --prediction_method "concat_mlp"
done