import json

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
        if(cnt %10 == 1) :
            print(f'处理了{cnt}个, correct links {correct_links}, total prediction links {total_predictions}')
    
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
        if(cnt %10 == 1) :
            print(f'处理了{cnt}个, correct links {correct_links}, total golden links {total_golden_links}')
    
    # 计算召回率
    recall = correct_links / total_golden_links if total_golden_links > 0 else 0
    return recall

def calculate_f1(precision, recall):
    if precision + recall == 0:
        return 0  # 避免除以零
    return 2 * (precision * recall) / (precision + recall)

if __name__ == '__main__':
    golden_file = "../data/gold_schema_linking/spider_dev_gold_schema_linking.json"
    dev_file = "../linked_schema_spider_dev_4o.json"
    precision = calculate_precision(golden_file, dev_file)
    recall = calculate_recall(golden_file, dev_file)
    print(f"Schema Linking Precision: {precision:.4f}")
    print(f"Schema Linking Recall: {recall:.4f}")

    f1_score = calculate_f1(precision, recall)
    print(f"F1 Score: {f1_score:.4f}")