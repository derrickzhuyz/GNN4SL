import json
import re
from loguru import logger
from fix_db_schemas import *

logger.add("logs/extraction.log", rotation="1 MB", level="INFO", 
           format="{time} {level} {message}", compression="zip")


"""Class for postprocessing database schema files"""
class DBSchemaPostProcessor:

    """
    Initialize SchemaProcessor
    :param db_schema_path: Path to the database schema file
    """
    def __init__(self, db_schema_path: str):
        self.db_schema_path = db_schema_path
        self.schema = self._load_schema()


    """
    Load schema from file
    :return: A dictionary representing the schema
    """
    def _load_schema(self) -> dict:
        try:
            with open(self.db_schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[! Error] Failed to load schema: {str(e)}")
            return None


    """
    Save schema to file
    """
    def _save_schema(self) -> None:
        try:
            with open(self.db_schema_path, 'w', encoding='utf-8') as f:
                json.dump(self.schema, f, indent=2, ensure_ascii=False)
            logger.info(f"[i] Schema saved to {self.db_schema_path}")
        except Exception as e:
            logger.error(f"[! Error] Failed to save schema: {str(e)}")


    """
    Get all column names for a specific table in a database.
    :param database_name (str): Name of the database
    :param table_name (str): Name of the table
    :return: List of column names for the specified table
    """
    def get_table_columns(self, database_name: str, table_name: str) -> list:
        try:
            db_schema = next((db for db in self.schema if db["database"] == database_name), None)
            if not db_schema:
                logger.error(f"[! Error] Database '{database_name}' not found in schema.")
                return []
            
            table = next((t for t in db_schema["tables"] if t["table"] == table_name), None)
            if not table:
                logger.error(f"[! Error] Table '{table_name}' not found in database '{database_name}'.")
                return []
            
            return table["columns"]
        except Exception as e:
            logger.error(f"[! Error] An error occurred: {str(e)}")
            return []


    """
    NOTE: only work when there is only ONE primary key for a table, otherwise (none or multiple) keep the null for manual/additional processing

    Fill null in foreign keys with primary key of corresponding table
    :return: None
    """
    def fill_null_foreign_keys(self) -> None:
        try:
            databases = self.schema if isinstance(self.schema, list) else [self.schema]
            
            for db in databases:
                if not isinstance(db, dict) or 'foreign_keys' not in db:
                    continue
                
                # Create table to primary key mapping
                table_pk_map = {
                    table['table']: table['primary_keys'][0]
                    for table in db['tables']
                    if 'primary_keys' in table and len(table['primary_keys']) == 1
                }
                
                # Process foreign keys
                for fk in db['foreign_keys']:
                    if not fk.get('table') or not fk.get('column'):
                        continue
                    if None in fk['column']:
                        null_idx = fk['column'].index(None)
                        table_name = fk['table'][null_idx]
                        if table_name in table_pk_map:
                            fk['column'][null_idx] = table_pk_map[table_name]
                        else:
                            logger.warning(f"[!] Could not find single primary key for table '{table_name}', keeping null")

            self._save_schema()
        except Exception as e:
            logger.error(f"[! Error] An error occurred while processing null foreign keys: {str(e)}")


    """
    Deduplicate the foreign key information in entire database schema
    :return: None
    """
    def deduplicate_foreign_keys(self) -> None:
        try:
            databases = self.schema if isinstance(self.schema, list) else [self.schema]
            
            for db in databases:
                if not isinstance(db, dict) or 'foreign_keys' not in db:
                    continue
                
                unique_fks = set()
                cleaned_fks = []
                
                for fk in db['foreign_keys']:
                    # Incomplete foreign keys (where any table or column is None/null): keep the foreign key as is
                    if (not fk.get('table') or not fk.get('column') or 
                        None in fk['table'] or None in fk['column'] or
                        len(fk['table']) != 2 or len(fk['column']) != 2):
                        cleaned_fks.append(fk)
                        continue
                    
                    # Skip self-referential foreign keys (same table name)
                    if fk['table'][0] == fk['table'][1]:
                        continue
                    
                    # Create normalized representation
                    fk_signature = (tuple(sorted(fk['table'])), tuple(sorted(fk['column'])))
                    
                    if fk_signature not in unique_fks:
                        unique_fks.add(fk_signature)
                        cleaned_fks.append({
                            'table': fk['table'],
                            'column': fk['column']
                        })
                
                db['foreign_keys'] = cleaned_fks
                db['foreign_key_count'] = len(cleaned_fks)
            
            self._save_schema()
        except Exception as e:
            logger.error(f"[! Error] An error occurred while cleaning foreign keys: {str(e)}")


    """
    Analyze the schema and add some statistics to the schema
    :return: None
    """
    def add_stats(self) -> None:
        try:
            for database in self.schema:
                database.update({
                    "table_count": len(database.get("tables", [])),
                    "total_column_count": sum(len(table.get("columns", [])) 
                                           for table in database.get("tables", [])),
                    "foreign_key_count": len(database.get("foreign_keys", []))
                })
            
            self._save_schema()
        except Exception as e:
            logger.error(f"[! Error] An error occurred while adding stats: {str(e)}")



"""Class for aligning information in info file with gold schema linking file"""
class GoldSchemaLinkingInfoAligner:
    """
    Align the information in the info file with the gold schema linking file
    :param info_path (str): Path to the info file
    :param gold_schema_path (str): Path to the gold schema linking file
    :return: None
    """
    @staticmethod
    def information_alignment(info_path: str, gold_schema_path: str) -> None:
        try:
            with open(info_path, 'r') as f:
                info_data = json.load(f)
            with open(gold_schema_path, 'r') as f:
                gold_data = json.load(f)
            
            if any(re.search(keyword, gold_schema_path, re.IGNORECASE) for keyword in ["bird_dev", "bird_train"]):
                for gold_item, info_item in zip(gold_data, info_data):
                    gold_item.update({
                        'question': info_item['question'],
                        'evidence': info_item['evidence'],
                        'difficulty': info_item.get('difficulty', None)
                    })
            elif any(re.search(keyword, gold_schema_path, re.IGNORECASE) for keyword in ["spider_dev", "spider_train"]):
                for gold_item, info_item in zip(gold_data, info_data):
                    gold_item['question'] = info_item['question']
            else:
                raise ValueError("[! Error] Invalid input.")
            
            with open(gold_schema_path, 'w') as f:
                json.dump(gold_data, f, indent=2)
            logger.info(f"[i] Aligned data has been saved to {gold_schema_path}")
        except Exception as e:
            logger.error(f"[! Error] An error occurred during information alignment: {str(e)}")



if __name__ == "__main__":
    with open('config/path_config.json') as config_file:
        path_config = json.load(config_file)

    # Information alignment for gold schema linking files
    gold_schema_aligner = GoldSchemaLinkingInfoAligner()
    gold_schema_aligner.information_alignment(info_path=path_config['bird_paths']['dev_info'], 
                                      gold_schema_path=path_config['gold_schema_linking_paths']['bird_dev'])
    gold_schema_aligner.information_alignment(info_path=path_config['bird_paths']['train_info'], 
                                      gold_schema_path=path_config['gold_schema_linking_paths']['bird_train'])
    gold_schema_aligner.information_alignment(info_path=path_config['spider_paths']['dev_info'], 
                                      gold_schema_path=path_config['gold_schema_linking_paths']['spider_dev'])
    gold_schema_aligner.information_alignment(info_path=path_config['spider_paths']['full_train_info'], 
                                      gold_schema_path=path_config['gold_schema_linking_paths']['spider_train'])
    

    # Postprocess entire database schemas for each dataset
    db_schema_files = [
        path_config['db_schema_paths']['bird_dev_schemas'],
        path_config['db_schema_paths']['bird_train_schemas'],
        path_config['db_schema_paths']['spider_schemas'],
        path_config['db_schema_paths']['spider2_lite_schemas']
    ]

    for item in db_schema_files:
        processor = DBSchemaPostProcessor(item)
        # 1. Fill null foreign keys
        processor.fill_null_foreign_keys()
        # 2. deduplicate foreign keys
        processor.deduplicate_foreign_keys()
        # 3. Add stats for database schemas
        processor.add_stats()

    # Fix foreign keys for certain database that not automatically processed above
    fix_mondial_geo_foreign_keys(path_config['db_schema_paths']['bird_train_schemas'], database_name="mondial_geo") # mondial_geo (bird train)