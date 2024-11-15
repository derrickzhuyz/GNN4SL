echo "[INFO] Running homo_train.py..."
sleep 2

python -m gnn.model.homo_train
if [ $? -ne 0 ]; then
    echo "[! Error] preprocess.py failed."
    exit 1
fi

echo "[INFO] homo_train.py completed!"