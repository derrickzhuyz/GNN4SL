import os
from pathlib import Path
import json
from eval_gnn_end2end_utils import *

# 假定以下函数已定义：compare_questions, extract_relevant_schema, calculate_precision, calculate_recall, calculate_f1

def process_file(file_path, results_list):
    # Skip non-JSON files
    if not file_path.endswith('.json'):
        return

    # 根据文件名判断benchmark
    file_name = os.path.basename(file_path)
    benchmark = 'bird' if 'bird_dev' in file_name else 'spider'

    # 根据benchmark选择ref_file
    if benchmark == 'spider':
        ref_file = 'evaluation/ref_file/linked_schema_spider_dev_ref.json'
    else:
        ref_file = 'evaluation/ref_file/linked_schema_bird_dev_ref.json'

    # gnn_schema_origin和gnn_schema路径
    gnn_schema_origin = file_path
    gnn_schema = str(Path(gnn_schema_origin).with_name("file3_" + os.path.basename(file_path)))

    try:
        # 执行步骤 1：比较schema linking
        flag1 = compare_questions(ref_file, gnn_schema_origin)

        # 执行步骤 2：提取标准schema格式
        if flag1:
            extract_relevant_schema(gnn_schema_origin, ref_file, gnn_schema)

        # 执行步骤 3：评估schema
        if benchmark == 'spider':
            golden_file = "data/gold_schema_linking/spider_dev_gold_schema_linking.json"
        else:
            golden_file = "data/gold_schema_linking/bird_dev_gold_schema_linking.json"

        precision = calculate_precision(golden_file, gnn_schema)
        recall = calculate_recall(golden_file, gnn_schema)
        f1_score = calculate_f1(precision, recall)

        # 将结果添加到结果列表中
        result = {
            "file_path": file_path,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        }

        results_list.append(result)
        print(f"Processed {file_path}: Precision={precision:.4f}, Recall={recall:.4f}, F1 Score={f1_score:.4f}")

    finally:
        # 删除中间生成的 gnn_schema 文件
        if os.path.exists(gnn_schema):
            os.remove(gnn_schema)
            print(f"Deleted intermediate file: {gnn_schema}")

def traverse_and_process(folder_path, output_file):
    # 用于存储所有结果的列表
    results_list = []

    # 遍历文件夹
    for root, _, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            process_file(file_path, results_list)

    # 将结果写入输出的 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(results_list, f, indent=4)

    print(f"Results saved to {output_file}")

if __name__ == '__main__':
    folder_path = 'gnn/results/link_level_rgat_12.15'  # 修改为需要遍历的文件夹路径
    output_file = 'gnn/results/link_level_rgat_12.15/rgat_schema_evaluation_results.json'  # 保存结果的文件路径

    traverse_and_process(folder_path, output_file)