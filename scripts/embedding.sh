#!/bin/bash

# Natural language embedding 

# Run NL embedding processor
echo "[INFO] Running nl_embedding_processor.py..."


python gnn/graph_data/nl_embedding_processor.py \
    --vector_dim 384 \
    --embed_method sentence_transformer

python gnn/graph_data/nl_embedding_processor.py \
    --vector_dim 1536 \
    --embed_method api_small

python gnn/graph_data/nl_embedding_processor.py \
    --vector_dim 3072 \
    --embed_method api_large

# Embedding log
echo "[INFO] Embedding log: logs/embedding.log"

if [ $? -ne 0 ]; then
    echo "[! Error] nl_embedding_processor.py failed."
    exit 1
fi
