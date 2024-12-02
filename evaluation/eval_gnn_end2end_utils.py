import json

def compare_questions(file1, file2):
    # 读取JSON文件
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
    
    # 判断两个文件是否包含相同数量的对象
    print(f"应该的大小{len(data1)}")
    print(f"gnn schema linking的大小{len(data2)}")
    if len(data1) != len(data2):
        print(f"{file2}文件中对象数量不一致，无法比对。")
        return False
    
    # 一一比对两个文件中对应对象的`question`字段
    for idx, (obj1, obj2) in enumerate(zip(data1, data2), start=1):
        question1 = obj1.get("question", "")
        question2 = obj2.get("question", "")
        
        if question1 == question2:
            # print(f"对象 {idx}: 问题相同 -> {question1}")
            continue
        else:
            print(f"对象 {idx}: 问题不同 -> 文件1: {question1}, 文件2: {question2}")
            return False
    return True

def extract_relevant_schema(file1_path, file2_path, file3_path):
    # 读取文件一和文件二
    with open(file1_path, 'r', encoding='utf-8') as f1, open(file2_path, 'r', encoding='utf-8') as f2:
        schema_linking_results = json.load(f1)
        reference_data = json.load(f2)

    # 存储文件三内容
    file3_content = []

    for result, ref in zip(schema_linking_results, reference_data):
        # 提取 database 信息
        database_name = ref.get("database", "")

        # 仅保留 relevant 为 true 的表和列
        relevant_tables = []
        for table in result["tables"]:
            if table["relevant"]:
                relevant_columns = [col["name"] for col in table["columns"] if col["relevant"]]
                relevant_tables.append({"table": table["name"], "columns": relevant_columns})

        # 构造文件三的对象
        extracted_schema = {
            "database": database_name,
            "tables": relevant_tables,
            "question": result["question"]
        }

        file3_content.append(extracted_schema)

    # 保存文件三
    with open(file3_path, 'w', encoding='utf-8') as f3:
        json.dump(file3_content, f3, ensure_ascii=False, indent=2)

def calculate_precision(golden_file, dev_file):
    # 加载文件内容
    with open(golden_file, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)
    with open(dev_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    correct_links = 0  # 预测正确的表/列数
    total_predictions = 0  # 预测的表/列总数

    # 遍历每个问题
    cnt = 0
    for golden, dev in zip(golden_data, dev_data):

        is_count_star = "count(*)" in golden["gold_sql"].lower()
        golden_links = set()
        for table in golden["tables"]:
            table_name = table["table"].lower()
            if not is_count_star or table["columns"]:  # 仅当没有 COUNT(*) 或列非空时才记录
                golden_links.update((table_name, col.lower()) for col in table["columns"])

        # 获取 Golden 的表和列
        # golden_links = set(
        #     (table["table"].lower(), col.lower()) 
        #     for table in golden["tables"] 
        #     for col in table["columns"]
        # )
        
        # 获取 Dev 的表和列
        dev_links = set(
            (table["table"].lower(), col.lower()) 
            for table in dev["tables"] 
            for col in table["columns"]
        )


        # 如果是 COUNT(*)，所有 dev_links 都视为正确
        if is_count_star:
            correct_links += len(dev_links)
        else:
        
        # 统计正确预测和总预测
            correct_links += len(golden_links & dev_links)  # 交集的数量
        total_predictions += len(dev_links)  # Dev 预测的数量

        cnt+=1
        # if(cnt %10 == 1) :
        #     print(f'处理了{cnt}个, correct links {correct_links}, total prediction links {total_predictions}')
    
    # 计算精确度
    precision = correct_links / total_predictions if total_predictions > 0 else 0
    return precision


def calculate_recall(golden_file, dev_file):
    # 加载文件内容
    with open(golden_file, 'r', encoding='utf-8') as f:
        golden_data = json.load(f)
    with open(dev_file, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)
    
    correct_links = 0  # 预测正确的表/列数
    total_golden_links = 0  # 实际相关的表/列总数

    # 遍历每个问题
    cnt = 0
    for golden, dev in zip(golden_data, dev_data):
        # 获取 Golden 的表和列
        # golden_links = set(
        #     (table["table"].lower(), col.lower()) 
        #     for table in golden["tables"] 
        #     for col in table["columns"]
        # )

                # 检查是否存在 COUNT(*)
        is_count_star = "count(*)" in golden["gold_sql"].lower()

        # 获取 Golden 的表和列，并标准化为小写
        golden_links = set()
        for table in golden["tables"]:
            table_name = table["table"].lower()
            #if not is_count_star or table["columns"]:  # 仅当没有 COUNT(*) 或列非空时才记录
            golden_links.update((table_name, col.lower()) for col in table["columns"])

        
        # 获取 Dev 的表和列
        dev_links = set(
            (table["table"].lower(), col.lower()) 
            for table in dev["tables"] 
            for col in table["columns"]
        )
        
                # 如果是 COUNT(*)，只考虑表级别
        if is_count_star:
            correct_links += len(dev_links)  # 所有 dev_links 视为正确
            total_golden_links += len(dev_links)  # 认为 golden 中也有同等数量的列
        else:
        # 统计正确预测和标准答案
            correct_links += len(golden_links & dev_links)  # 交集的数量
            total_golden_links += len(golden_links)  # Golden 的总数

        cnt+=1
        # if(cnt %10 == 1) :
        #     print(f'处理了{cnt}个, correct links {correct_links}, total golden links {total_golden_links}')
    
    # 计算召回率
    recall = correct_links / total_golden_links if total_golden_links > 0 else 0
    return recall

def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0  # 避免除以零
    return 2 * (precision * recall) / (precision + recall)





if __name__ == '__main__':

    benchmark = 'bird'
    gnn_schema = '../file3_20241125_132315_1.json'  # 提取出gnn_schema新建文件的路径

    # 1. 比较gnn的schema linking是否数目足够并且遵循顺序。
    if benchmark == 'spider':
        ref_file = '../linked_schema_spider_dev_new.json'  # 替换为第一个JSON文件的路径
    else:
        ref_file = '../linked_schema_bird_dev_gnn100.json'
    gnn_schema_origin = '../gnn/results/link_level/bird_dev_predictions_link_level_model_combined_20241125_132315_top_4_5.json'  # 替换为第二个JSON文件的路径

    flag1 = compare_questions(ref_file, gnn_schema_origin)

    # 2. 提取 gnn的schema为标准的schema格式，存放于
    if flag1:
        extract_relevant_schema(gnn_schema_origin, ref_file, gnn_schema)

    # 3. 评估gnn schema
    if benchmark == 'spider':
        golden_file = "../data/gold_schema_linking/spider_dev_gold_schema_linking.json"
    else:
        golden_file = "../data/gold_schema_linking/bird_dev_gold_schema_linking.json"
    
    precision = calculate_precision(golden_file, gnn_schema)
    recall = calculate_recall(golden_file, gnn_schema)
    print(f"Schema Linking Precision: {precision:.4f}")
    print(f"Schema Linking Recall: {recall:.4f}")

    f1_score = calculate_f1(precision, recall)
    print(f"F1 Score: {f1_score:.4f}")