#!/bin/bash

# Graph dataset construction

# Run link-level graph dataset construction
echo "[INFO] Running link_level_graph_dataset.py..."
python gnn/graph_data/link_level_graph_dataset.py \
    --embed_method sentence_transformer

python gnn/graph_data/link_level_graph_dataset.py \
    --embed_method api_small

python gnn/graph_data/link_level_graph_dataset.py \
    --embed_method api_large

if [ $? -ne 0 ]; then
    echo "[! Error] link_level_graph_dataset.py failed."
    exit 1
fi

echo "[INFO] Link-level graph dataset construction completed!"


# Run node-level graph dataset construction
# echo "[INFO] Running node_level_graph_dataset.py..."
# python gnn/graph_data/node_level_graph_dataset.py
# if [ $? -ne 0 ]; then
#     echo "[! Error] node_level_graph_dataset.py failed."
#     exit 1
# fi

# echo "[INFO] Node-level graph dataset construction completed!"

# Schema to graph data log
echo "[INFO] Schema to graph data log:"
echo "    Link-level graph dataset log: logs/link_level_graph_dataset.log"
# echo "    Node-level graph dataset log: logs/node_level_graph_dataset.log"