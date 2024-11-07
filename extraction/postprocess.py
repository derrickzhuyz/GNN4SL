import json
import re


"""This file should be used after gold schema linking has been done and stored"""


"""
Deal with the schema extraction when there is a '*' in the SQL query
"""
def process_schema_extraction_with_asterisk(file_path: str) -> None:
    print("No action.")
    return


"""
Get all column names for a specific table in a database.
:param db_schema_path (str): Path to the database schema file
:param database_name (str): Name of the database
:param table_name (str): Name of the table
:return: List of column names for the specified table
"""
def get_table_columns(db_schema_path: str, database_name: str, table_name: str) -> list:
    try:
        with open(db_schema_path, 'r') as f:
            schema = json.load(f)
        
        # Find the specified database
        db_schema = next((db for db in schema if db["database"] == database_name), None)
        if not db_schema:
            print(f"[! Error] Database '{database_name}' not found in schema.")
            return []
        
        # Find the specified table
        table = next((t for t in db_schema["tables"] if t["table"] == table_name), None)
        if not table:
            print(f"[! Error] Table '{table_name}' not found in database '{database_name}'.")
            return []
        
        # Return list of column names
        return table["columns"]
    except Exception as e:
        print(f"[! Error]An error occurred: {str(e)}")
        return []


"""
Align the information in the info file with the gold schema linking file
:param info_path (str): Path to the info file
:param gold_schema_path (str): Path to the gold schema linking file
:return: None
"""
def information_alignment(info_path: str, gold_schema_path: str) -> None:
    # Load info and gold schema linking data
    with open(info_path, 'r') as f:
        info_data = json.load(f)
    with open(gold_schema_path, 'r') as f:
        gold_data = json.load(f)
    
    if any(re.search(keyword, gold_schema_path, re.IGNORECASE) for keyword in ["bird_dev", "bird_train"]):
        # Bird
        for gold_item, info_item in zip(gold_data, info_data):
            gold_item.update({
                'question': info_item['question'],
                'evidence': info_item['evidence'],
                'difficulty': info_item.get('difficulty', None)
            })
    elif any(re.search(keyword, gold_schema_path, re.IGNORECASE) for keyword in ["spider_dev", "spider_train"]):
        # Spider
        for gold_item, info_item in zip(gold_data, info_data):
            gold_item['question'] = info_item['question']
            
    else:
        raise ValueError(f"[! Error] Invalid input.")
    
    with open(gold_schema_path, 'w') as f:
        json.dump(gold_data, f, indent=2)
    print(f"[i] Aligned data has been saved to {gold_schema_path}")
    return




if __name__ == "__main__":
    with open('config.json') as config_file:
        config = json.load(config_file)

    information_alignment(info_path=config['bird_paths']['dev_info'], gold_schema_path=config['gold_schema_linking_paths']['bird_dev'])
    information_alignment(info_path=config['bird_paths']['train_info'], gold_schema_path=config['gold_schema_linking_paths']['bird_train'])
    information_alignment(info_path=config['spider_paths']['dev_info'], gold_schema_path=config['gold_schema_linking_paths']['spider_dev'])
    information_alignment(info_path=config['spider_paths']['full_train_info'], gold_schema_path=config['gold_schema_linking_paths']['spider_train'])
    pass
