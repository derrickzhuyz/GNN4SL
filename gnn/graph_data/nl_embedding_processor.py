import json
import os
from pathlib import Path
from nl_embedder import NLEmbedder
from loguru import logger
from tqdm import tqdm
from typing import List, Dict

logger.add("logs/embedding.log", rotation="1 MB", level="INFO",
           format="{time} {level} {message}", compression="zip")


"""
Process labeled dataset and store embeddings incrementally with batching.
:param input_file: Path to the labeled schema linking dataset
:param output_dir: Output directory for embedded data
:param vector_dim: Dimension of the embedding vectors
:param batch_size: Number of examples to process before saving

Features:
1. Processes examples in batches to manage memory
2. Can resume from last successful save if interrupted
3. Shows progress with tqdm
4. Maintains original order of examples
5. Uses temporary file for safe saving
"""
def dataset_question_embedding(input_file: str, output_dir: str, vector_dim: int = 1024, batch_size: int = 100):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    temp_file = os.path.join(output_dir, f"temp_{os.path.basename(input_file)}")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    embedder = NLEmbedder(vector_dim=vector_dim, openai_api_key=api_key, base_url=base_url)
    
    try:
        # Load all input data first to maintain order
        with open(input_file, 'r') as f:
            labeled_data = json.load(f)
        
        # Create index mapping to preserve order
        id_to_index = {item['id']: idx for idx, item in enumerate(labeled_data)}
        
        # Load or initialize processed data
        processed_data = []
        processed_ids = set()
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                processed_data = json.load(f)
                processed_ids = {item.get('id') for item in processed_data}
            logger.info(f"[↻] Resuming from {len(processed_ids)} processed examples")
        
        # Process unprocessed examples in batches
        new_examples = []
        unprocessed_examples = [ex for ex in labeled_data if ex['id'] not in processed_ids]
        
        for example in tqdm(unprocessed_examples, desc="Processing examples"):
            try:
                embedded_example = embedder.embed_question(example, embed_method='api_mock')
                new_examples.append(embedded_example)
                
                # Save when batch size is reached
                if len(new_examples) >= batch_size:
                    # Sort batch by original order before saving
                    new_examples.sort(key=lambda x: id_to_index[x['id']])
                    _save_batch(new_examples, processed_data, output_file, temp_file)
                    new_examples = []
                    
            except Exception as e:
                logger.error(f"[✗] Error processing example {example['id']}: {str(e)}")
                continue
        
        # Save any remaining examples
        if new_examples:
            new_examples.sort(key=lambda x: id_to_index[x['id']])
            _save_batch(new_examples, processed_data, output_file, temp_file)
        
        logger.info(f"[✓] Successfully processed all examples while maintaining order")
        
    except Exception as e:
        logger.error(f"[✗] Fatal error during processing: {str(e)}")
        raise


"""
Helper function to safely save a batch of processed data while maintaining order.
:param new_batch: List of newly processed examples
:param processed_data: List of previously processed examples
:param output_file: Path to the final output file
:param temp_file: Path to temporary file for safe saving
"""
def _save_batch(new_batch: List[Dict], processed_data: List[Dict], output_file: str, temp_file: str):

    try:
        # Combine existing data with new batch
        updated_data = processed_data + new_batch
        
        # Save to temporary file first
        with open(temp_file, 'w') as f:
            json.dump(updated_data, f, indent=2)
        # If temporary save successful, replace the original file
        os.replace(temp_file, output_file)
        
        # Update processed_data reference
        processed_data.extend(new_batch)
        logger.info(f"[✓] Saved batch of {len(new_batch)} examples. Current processed: {len(processed_data)}.")
    except Exception as e:
        logger.error(f"[✗] Error saving batch: {str(e)}")
        if os.path.exists(temp_file):
            os.remove(temp_file)
        raise


"""
Process database schema and store embeddings incrementally with batching.
:param input_file: Path to the database schema file (spider_schemas.json)
:param output_dir: Output directory for embedded schema data
:param vector_dim: Dimension of the embedding vectors
:param batch_size: Number of databases to process before saving

Features:
1. Processes schemas in batches to manage memory
2. Can resume from last successful save if interrupted
3. Shows progress with tqdm
4. Maintains original order of schemas
5. Uses temporary file for safe saving
"""
def database_schema_embedding(input_file: str, output_dir: str, vector_dim: int = 1024, batch_size: int = 5):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    temp_file = os.path.join(output_dir, f"temp_{os.path.basename(input_file)}")

    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    embedder = NLEmbedder(vector_dim=vector_dim, openai_api_key=api_key, base_url=base_url)
    try:
        # Load all schema data first
        with open(input_file, 'r') as f:
            schema_data = json.load(f)
            
        # Create database name to index mapping for order preservation
        db_to_index = {schema['database']: idx for idx, schema in enumerate(schema_data)}
        
        # Load or initialize processed data
        processed_schemas = []
        processed_dbs = set()
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                processed_schemas = json.load(f)
                processed_dbs = {schema['database'] for schema in processed_schemas}
            logger.info(f"[↻] Resuming from {len(processed_dbs)} processed databases")
        
        # Process unprocessed schemas in batches
        new_schemas = []
        unprocessed_schemas = [s for s in schema_data if s['database'] not in processed_dbs]
        
        for schema in tqdm(unprocessed_schemas, desc="Processing database schemas"):
            try:
                embedded_schema = embedder.embed_schema_element_names(schema, embed_method='api_mock')
                new_schemas.append(embedded_schema)
                
                # Save when batch size is reached
                if len(new_schemas) >= batch_size:
                    # Sort batch by original order
                    new_schemas.sort(key=lambda x: db_to_index[x['database']])
                    _save_batch(new_schemas, processed_schemas, output_file, temp_file)
                    new_schemas = []
                    
            except Exception as e:
                logger.error(f"[✗] Error processing database {schema['database']}: {str(e)}")
                continue
        
        # Save any remaining schemas
        if new_schemas:
            new_schemas.sort(key=lambda x: db_to_index[x['database']])
            _save_batch(new_schemas, processed_schemas, output_file, temp_file)
        
        logger.info(f"[✓] Successfully processed all database schemas")
        
    except Exception as e:
        logger.error(f"[✗] Fatal error during processing: {str(e)}")
        raise



if __name__ == "__main__":
    with open('config/path_config.json') as config_file:
        path_config = json.load(config_file)

    # Uniform embedding dimension for all embedding files
    uniform_vector_dim = 3
    
    # Embed question in labeled schema linking dataset
    dataset_question_embedding(input_file=path_config['labeled_dataset_paths']['spider_dev'], 
                              output_dir=path_config['embed_question_paths']['embed_question_base'], 
                              vector_dim=uniform_vector_dim,
                              batch_size=100)
    dataset_question_embedding(input_file=path_config['labeled_dataset_paths']['spider_train'], 
                              output_dir=path_config['embed_question_paths']['embed_question_base'], 
                              vector_dim=uniform_vector_dim,
                              batch_size=100)
    dataset_question_embedding(input_file=path_config['labeled_dataset_paths']['bird_dev'], 
                              output_dir=path_config['embed_question_paths']['embed_question_base'], 
                              vector_dim=uniform_vector_dim,
                              batch_size=100)
    dataset_question_embedding(input_file=path_config['labeled_dataset_paths']['bird_train'], 
                              output_dir=path_config['embed_question_paths']['embed_question_base'], 
                              vector_dim=uniform_vector_dim,
                              batch_size=100)
    
    # Embed table/column name in database so that we can search coresponding table/column name when constructing graph dataset
    database_schema_embedding(input_file=path_config['db_schema_paths']['spider_schemas'], 
                              output_dir=path_config['embed_db_schema_paths']['embed_db_schema_base'], 
                              vector_dim=uniform_vector_dim,
                              batch_size=5)
    database_schema_embedding(input_file=path_config['db_schema_paths']['bird_train_schemas'], 
                              output_dir=path_config['embed_db_schema_paths']['embed_db_schema_base'], 
                              vector_dim=uniform_vector_dim,
                              batch_size=5)
    database_schema_embedding(input_file=path_config['db_schema_paths']['bird_dev_schemas'], 
                              output_dir=path_config['embed_db_schema_paths']['embed_db_schema_base'], 
                              vector_dim=uniform_vector_dim,
                              batch_size=5)
    database_schema_embedding(input_file=path_config['db_schema_paths']['spider2_lite_schemas'], 
                              output_dir=path_config['embed_db_schema_paths']['embed_db_schema_base'], 
                              vector_dim=uniform_vector_dim,
                              batch_size=5)

    pass