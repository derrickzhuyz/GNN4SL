echo "[INFO] Running homo_train.py..."
sleep 2

python -m gnn.model.homo_train
if [ $? -ne 0 ]; then
    echo "[! Error] homo_train.py failed."
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