echo "[INFO] Running node_level_train.py..."
sleep 2

python -m gnn.model.node_level_train
if [ $? -ne 0 ]; then
    echo "[! Error] node_level_train.py failed."
    exit 1
fi

echo "[INFO] node_level_train.py completed!"


# -------------------------------------------------------------------------------

echo "[INFO] Running node_level_test.py..."
sleep 2

python -m gnn.model.node_level_test
if [ $? -ne 0 ]; then
    echo "[! Error] node_level_test.py failed."
    exit 1
fi

echo "[INFO] homo_test.py completed!"