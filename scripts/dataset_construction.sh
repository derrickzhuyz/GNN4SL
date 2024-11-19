#!/bin/bash

echo "[INFO] You are running the script: extract_schema.sh"
sleep 2

# Run preprocessing
echo "[INFO] Running preprocess.py..."
python extraction/preprocess.py
if [ $? -ne 0 ]; then
    echo "[! Error] preprocess.py failed."
    exit 1
fi

# Run DB schema extraction
echo "[INFO] Running extract_db_schema.py..."
python extraction/extract_db_schema.py
if [ $? -ne 0 ]; then
    echo "[! Error] extract_db_schema.py failed."
    exit 1
fi

# Run gold schema linking
echo "[INFO] Running gold_schema_linking.py..."
python extraction/gold_schema_linking.py
if [ $? -ne 0 ]; then
    echo "[! Error] gold_schema_linking.py failed."
    exit 1
fi

# Run postprocessing
echo "[INFO] Running postprocess.py..."
python extraction/postprocess.py
if [ $? -ne 0 ]; then
    echo "[! Error] postprocess.py failed."
    exit 1
fi


echo "[INFO] Schema extraction completed! Now you can run db_schema_stats.ipynb to get some statistics of the DB schemas."

# Schema extraction log
echo "[INFO] Schema extraction log: logs/extraction.log"


echo "[INFO] Running extract_labeled_dataset.py..."
python extraction/extract_labeled_dataset.py
if [ $? -ne 0 ]; then
    echo "[! Error] extract_labeled_dataset.py failed."
    exit 1
fi


# -------------------------------------------------------------------------------


# Run NL embedding processor
echo "[INFO] Running nl_embedding_processor.py..."
python gnn/graph_data/nl_embedding_processor.py
if [ $? -ne 0 ]; then
    echo "[! Error] embedding.py failed."
    exit 1
fi

# Embedding log
echo "[INFO] Embedding log: logs/embedding.log"


# -------------------------------------------------------------------------------


# Run node-level graph dataset construction
echo "[INFO] Running node_level_graph_dataset.py..."
python gnn/graph_data/node_level_graph_dataset.py
if [ $? -ne 0 ]; then
    echo "[! Error] node_level_graph_dataset.py failed."
    exit 1
fi

echo "[INFO] Node-level graph dataset construction completed!"


# Run link-level graph dataset construction
echo "[INFO] Running link_level_graph_dataset.py..."
python gnn/graph_data/link_level_graph_dataset.py
if [ $? -ne 0 ]; then
    echo "[! Error] link_level_graph_dataset.py failed."
    exit 1
fi

echo "[INFO] Link-level graph dataset construction completed!"


# Schema to graph data log
echo "[INFO] Schema to graph data log:"
echo "    Node-level graph dataset log: logs/node_level_graph_dataset.log"
echo "    Link-level graph dataset log: logs/link_level_graph_dataset.log"
