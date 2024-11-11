import json
import os
from typing import Dict, List, Set
from loguru import logger

logger.add("logs/extraction.log", rotation="1 MB", level="INFO", 
           format="{time} {level} {message}", compression="zip")


"""
Extract and label schema for each example based on gold schema linking
:param dataset_type: The type of dataset (spider or bird)
:param split: The split of the dataset (train or dev)
:return: None
"""
def extract_schema_linking(dataset_type: str, split: str, path_config: Dict) -> None:
    # Load schema file
    if dataset_type == 'bird':
        db_schema_file = path_config['db_schema_paths']['bird_train_schemas'] if split == 'train' else path_config['db_schema_paths']['bird_dev_schemas']
    else:
        db_schema_file = path_config['db_schema_paths'][f"{dataset_type}_schemas"]
    
    with open(db_schema_file, 'r') as f:
        db_schemas = json.load(f)
    db_schema_dict = {schema['database']: schema for schema in db_schemas}
    
    # Load gold schema linking file
    linking_file = path_config['gold_schema_linking_paths'][f"{dataset_type}_{split}"]
    with open(linking_file, 'r') as f:
        examples = json.load(f)
    
    # Process each example
    labeled_data = []
    for example in examples:
        db_name = example['database']
        if db_name not in db_schema_dict:
            logger.warning(f"Database {db_name} not found in schema file")
            continue
            
        # Get full schema
        schema = db_schema_dict[db_name]
        
        # Get relevant tables and columns from example
        relevant_tables = set()
        relevant_columns = set()
        for table in example['tables']:
            relevant_tables.add(table['table'].lower())
            for col in table['columns']:
                relevant_columns.add((table['table'].lower(), col.lower()))
        
        # Label schema elements
        labeled_tables = []
        for table in schema['tables']:
            table_name = table['table'].lower()
            labeled_columns = []
            
            for col in table['columns']:
                col_name = col.lower()
                labeled_columns.append({
                    'name': col,
                    'relevant': 1 if (table_name, col_name) in relevant_columns else 0
                })
            
            labeled_tables.append({
                'name': table['table'],
                'relevant': 1 if table_name in relevant_tables else 0,
                'columns': labeled_columns
            })
        
        # Create labeled example, preserving all original information
        labeled_example = {
            'database': db_name,
            'question': example['question'],
            'id': example['id'],
            'gold_sql': example.get('gold_sql', ''),
            'remarks': example.get('remarks', ''),
            'tables': labeled_tables,
            'foreign_keys': schema.get('foreign_keys', []),
            'table_count': schema.get('table_count', 0),
            'total_column_count': schema.get('total_column_count', 0),
            'foreign_key_count': schema.get('foreign_key_count', 0),
            'involved_table_count': len(relevant_tables),
            'involved_column_count': len(relevant_columns)
        }
        
        # Preserve any additional fields from the original question
        for key, value in example.items():
            if key not in labeled_example:
                labeled_example[key] = value
                
        labeled_data.append(labeled_example)
    
    # Save labeled data
    output_dir = path_config['labeled_dataset_paths']['labeled_dataset_base']
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f'{dataset_type}_{split}_labeled.json')
    
    with open(output_file, 'w') as f:
        json.dump(labeled_data, f, indent=2)
    
    logger.info(f"[INFO] Processed {len(labeled_data)} examples for {dataset_type} {split}")



if __name__ == "__main__":
    with open('config/path_config.json') as config_file:
        path_config = json.load(config_file)
    
    # Process Spider dataset
    extract_schema_linking('spider', 'train', path_config)
    extract_schema_linking('spider', 'dev', path_config)
    
    # Process BIRD dataset
    extract_schema_linking('bird', 'train', path_config)
    extract_schema_linking('bird', 'dev', path_config)