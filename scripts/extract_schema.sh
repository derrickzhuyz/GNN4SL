echo "[INFO] You are running the script: extract_schema.sh"
sleep 2

wait
python extraction/preprocess.py

wait
python extraction/extract_db_schema.py

wait
python extraction/gold_schema_linking.py

wait
python extraction/postprocess.py

wait
echo "[INFO] Schema extraction completed!"

