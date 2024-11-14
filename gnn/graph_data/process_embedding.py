import json
import os
from pathlib import Path
from nl_embedder import NLEmbedder
from loguru import logger

logger.add("logs/embedding_process.log", rotation="1 MB", level="INFO",
           format="{time} {level} {message}", compression="zip")



"""
Process labeled dataset and store embeddings incrementally.
:param input_file: str, the path to the labeled schema linking dataset
:param output_dir: str, the output directory, labeled schema linking dataset with question embeddings
"""
def dataset_question_embedding(input_file: str, output_dir: str, vector_dim: int = 1024):
    api_key=os.getenv("OPENAI_API_KEY")
    base_url=os.getenv("OPENAI_BASE_URL")
    embedder = NLEmbedder(vector_dim=vector_dim, openai_api_key=api_key, base_url=base_url)

    # Create output directory if it doesn't exist and set output file path
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    try:
        with open(input_file, 'r') as f:
            labeled_data = json.load(f)

        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                processed_data = json.load(f)
            processed_ids = {item.get('id') for item in processed_data if item.get('id') is not None}
        else:
            processed_data = []
            processed_ids = set()
        
        # Process each example
        for example in labeled_data:
            if example.get('id') in processed_ids:
                logger.info(f"[i] Skipping already processed example {example.get('id')}")
                continue
                
            embedded_example = embedder.embed_question(example, embed_method='api_mock')
            processed_data.append(embedded_example)
            
            # Save after each example
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            status = embedded_example.get('status', 'unknown')
            logger.info(f"[i] Processed example {example.get('id')} - Status: {status}")
            
            if status == 'error':
                logger.error(f"[! Error] Error in example {example.get('id')}: {embedded_example.get('error_message')}")
        
        logger.info(f"[i] Completed processing all examples. Total processed: {len(processed_data)}")
    except Exception as e:
        logger.error(f"[! Error] Fatal error during processing: {str(e)}")
        raise


"""
Process database schema and store embeddings incrementally.
:param input_file: str, path to the database schema file (spider_schemas.json)
:param output_dir: str, output directory for embedded schema data
"""
def database_schema_embedding(input_file: str, output_dir: str, vector_dim: int = 1024):

    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize embedder
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    embedder = NLEmbedder(vector_dim=vector_dim, openai_api_key=api_key, base_url=base_url)
    
    # Create output file path
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    
    try:
        # Read input file
        with open(input_file, 'r') as f:
            schema_data = json.load(f)
        
        # Initialize or load existing processed data
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                processed_schemas = json.load(f)
            processed_dbs = {schema['database'] for schema in processed_schemas}
        else:
            processed_schemas = []
            processed_dbs = set()
        
        # Process each database schema
        for schema in schema_data:
            if schema['database'] in processed_dbs:
                logger.info(f"[i] Skipping already processed database {schema['database']}")
                continue
            
            embedded_schema = embedder.embed_schema_elements(schema, embed_method='api_mock')
            processed_schemas.append(embedded_schema)
            
            # Save after each schema
            with open(output_file, 'w') as f:
                json.dump(processed_schemas, f, indent=2)
            
            if 'remarks' in embedded_schema and embedded_schema['remarks']:
                logger.error(f"[! Error] Error in database {schema['database']}: {embedded_schema['remarks']}")
            else:
                logger.info(f"[i] Successfully processed database {schema['database']}")
        
        logger.info(f"[i] Completed processing all schemas. Total processed: {len(processed_schemas)}")
        
    except Exception as e:
        logger.error(f"[! Error] Fatal error during processing: {str(e)}")
        raise



if __name__ == "__main__":
    with open('config/path_config.json') as config_file:
        path_config = json.load(config_file)
    
    # Embed question in labeled schema linking dataset
    dataset_question_embedding(input_file=path_config['labeled_dataset_paths']['spider_dev'], 
                              output_dir=path_config['embed_question_paths']['embed_question_base'], 
                              vector_dim=3)
    # dataset_question_embedding(input_file=path_config['labeled_dataset_paths']['spider_train'], 
    #                           output_dir=path_config['embed_question_paths']['embed_question_base'], 
    #                           vector_dim=1024)
    # dataset_question_embedding(input_file=path_config['labeled_dataset_paths']['bird_dev'], 
    #                           output_dir=path_config['embed_question_paths']['embed_question_base'], 
    #                           vector_dim=1024)
    # dataset_question_embedding(input_file=path_config['labeled_schema_linking_paths']['bird_train'], 
    #                           output_dir=path_config['embed_question_paths']['embed_question_base'], 
    #                           vector_dim=1024)
    
    # Embed table/column name in database so that we can search coresponding table/column name when constructing graph dataset
    database_schema_embedding(input_file=path_config['db_schema_paths']['spider_schemas'], 
                              output_dir=path_config['embed_db_schema_paths']['embed_db_schema_base'], 
                              vector_dim=3)
    # database_schema_embedding(input_file=path_config['db_schema_paths']['bird_train_schemas'], 
    #                           output_dir=path_config['embed_db_schema_paths']['embed_db_schema_base'], 
    #                           vector_dim=1024)
    # database_schema_embedding(input_file=path_config['db_schema_paths']['bird_dev_schemas'], 
    #                           output_dir=path_config['embed_db_schema_paths']['embed_db_schema_base'], 
    #                           vector_dim=1024)