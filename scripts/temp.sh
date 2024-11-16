# Run NL embedding processor
echo "[INFO] Running nl_embedding_processor.py..."
python gnn/graph_data/nl_embedding_processor.py
if [ $? -ne 0 ]; then
    echo "[! Error] embedding.py failed."
    exit 1
fi


# Embedding log
echo "[INFO] Embedding log: logs/embedding.log"

# Run homo graph dataset construction
echo "[INFO] Running homo_graph_dataset.py..."
python gnn/graph_data/homo_graph_dataset.py
if [ $? -ne 0 ]; then
    echo "[! Error] homo_graph_dataset.py failed."
    exit 1
fi

# Schema to graph data log
echo "[INFO] Schema to graph data log: logs/schema2graph_dataset.log"



# -------------------------------------------------------------------------------

echo "[INFO] Running homo_train.py..."
sleep 2

python -m gnn.model.homo_train
if [ $? -ne 0 ]; then
    echo "[! Error] preprocess.py failed."
    exit 1
fi

echo "[INFO] homo_train.py completed!"


# -------------------------------------------------------------------------------

echo "[INFO] Running homo_test.py..."
sleep 2

python -m gnn.model.homo_test
if [ $? -ne 0 ]; then
    echo "[! Error] homo_test.py failed."
    exit 1
fi

echo "[INFO] homo_test.py completed!"