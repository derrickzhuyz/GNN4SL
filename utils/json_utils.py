import json


def count_json_objects(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return len(data)