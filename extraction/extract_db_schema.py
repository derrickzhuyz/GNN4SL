import sqlite3
import json
import os


"""
Get the schema information of a database file (.sqlite)
:param db_path: The path of the database file
:return: A dictionary containing the schema information
"""
def get_database_schema(db_path):
    database_name = os.path.basename(db_path).replace('.sqlite', '')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract all non-system table names
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = cursor.fetchall()

    # Schema dictionary
    schema = {
        "database": database_name,
        "tables": [],
        "foreign_keys": []
    }
    for table_name in tables:
        table_name = table_name[0]
        # Get the column information of each table
        cursor.execute(f'PRAGMA table_info("{table_name}");')
        columns = cursor.fetchall()
        column_names = [column[1] for column in columns]
        # Add to the schema list
        schema["tables"].append({
            "table": table_name,
            "columns": column_names
        })
        # Get the foreign key information of the table
        cursor.execute(f'PRAGMA foreign_key_list("{table_name}");')
        foreign_keys = cursor.fetchall()
        # Add foreign key information to the schema
        for fk in foreign_keys:
            schema["foreign_keys"].append({
                "table": [table_name, fk[2]],  # Current table name and reference table name
                "column": [fk[3], fk[4]]       # Current column name and reference column name
            })

    conn.close()
    return schema


"""
Extract the schema information of all databases of a dataset and save to a JSON file
:param db_path: The path of the database folder
:param db_schema_json: The path of the output JSON file of entire database schema
:param nested_folder: Whether the databases are located in nested folders
:return: None
"""
def extract_entire_dataset_schemas(db_path, db_schema_json, nested_folder=False):
    all_schemas = []
    if nested_folder == False:
        # All database is located in one folder, no nested folder e.g. spider2-lite
        for file_name in os.listdir(db_path):
            if file_name.endswith('.sqlite'):
                db_file = os.path.join(db_path, file_name)
                schema = get_database_schema(db_file)
                all_schemas.append(schema)
    else:
        # Each database is located in a separate folder, e.g. spider and bird
        for folder_name in os.listdir(db_path):
            folder_path = os.path.join(db_path, folder_name)
            # Check if the folder exists
            if os.path.isdir(folder_path):
                db_file = os.path.join(folder_path, f"{folder_name}.sqlite")
                # Check if the database file exists
                if os.path.isfile(db_file):
                    schema = get_database_schema(db_file)
                    all_schemas.append(schema)
                else:
                    raise FileNotFoundError(f"Database files not found: {db_file}")
    
    # Save the schema information to a JSON file
    with open(db_schema_json, 'w', encoding='utf-8') as f:
        json.dump(all_schemas, f, ensure_ascii=False, indent=2)
        print(f"[i] Schema information of entire databases has been saved to {db_schema_json}")



if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)

    spider_db_path = config['spider_paths']['all_databases']
    spider_db_schema_json = config['db_schema_paths']['spider_schemas']
    extract_entire_dataset_schemas(spider_db_path, spider_db_schema_json, nested_folder=True)

    bird_train_db_path = config['bird_paths']['train_databases']
    bird_train_db_schema_json = config['db_schema_paths']['bird_train_schemas']
    extract_entire_dataset_schemas(bird_train_db_path, bird_train_db_schema_json, nested_folder=True)

    bird_dev_db_path = config['bird_paths']['dev_databases']
    bird_dev_db_schema_json = config['db_schema_paths']['bird_dev_schemas']
    extract_entire_dataset_schemas(bird_dev_db_path, bird_dev_db_schema_json, nested_folder=True)

    spider2_lite_localdb_path = config['spider2_lite_paths']['spider2_localdb']
    spider2_lite_db_schema_json = config['db_schema_paths']['spider2_lite_schemas']
    extract_entire_dataset_schemas(spider2_lite_localdb_path, spider2_lite_db_schema_json, nested_folder=False) # Note that file structure of spider2-lite is not nested

    