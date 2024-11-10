import json
from loguru import logger

logger.add("logs/extraction.log", rotation="1 MB", level="INFO", 
           format="{time} {level} {message}", compression="zip")


"""
Fix the foreign keys in mondial_geo database (bird train)
Null is related to "province" table:
    if the other column is "Province", replace null with "Name";
    if the other column is "Country", replace null with "Country";
Null is related to "city" table:
    if the other column is "City", replace null with "Name";
    if the other column is "Province", replace null with "Province"
:param db_schema_path (str): Path to the database schema file
:return: None
"""
def fix_mondial_geo_foreign_keys(db_schema_path: str, database_name: str = "mondial_geo") -> None:
    try:
        with open(db_schema_path, 'r') as f:
            schema = json.load(f)
        
        # Handle both single database and array of databases
        databases = schema if isinstance(schema, list) else [schema]
        
        for db in databases:
            if not isinstance(db, dict) or db.get('database') != database_name:
                continue
            
            for fk in db['foreign_keys']:
                if not fk.get('table') or not fk.get('column'):
                    continue
                    
                # Check for null in columns
                if None in fk['column']:
                    null_idx = fk['column'].index(None)
                    target_table = fk['table'][null_idx]
                    other_column = fk['column'][1 - null_idx]
                    
                    # Rules for province table
                    if target_table == "province":
                        if other_column == "Province":
                            fk['column'][null_idx] = "Name"
                        elif other_column == "Country":
                            fk['column'][null_idx] = "Country"
                            
                    # Rules for city table
                    elif target_table == "city":
                        if other_column == "City":
                            fk['column'][null_idx] = "Name"
                        elif other_column == "Province":
                            fk['column'][null_idx] = "Province"
                            
                    logger.info(f"[i] Replaced null with '{fk['column'][null_idx]}' in foreign key between {fk['table'][0]} and {fk['table'][1]}")
        
        # Write the updated schema back to file
        with open(db_schema_path, 'w') as f:
            json.dump(schema, f, indent=2)
            
        logger.info(f"[i] Fixed {database_name} foreign keys in {db_schema_path}")
        
    except Exception as e:
        logger.error(f"[! Error] An error occurred while fixing {database_name} foreign keys: {str(e)}")



