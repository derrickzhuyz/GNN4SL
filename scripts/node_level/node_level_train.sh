echo "[INFO] Running node_level_train.py..."
sleep 2

python -m gnn.model.node_level_train
if [ $? -ne 0 ]; then
    echo "[! Error] node_level_train.py failed."
    exit 1
fi

echo "[INFO] node_level_train.py completed!"
