import json
import os
from pathlib import Path
from nl_embedding import NLEmbedder
from loguru import logger

logger.add("logs/embedding_process.log", rotation="1 MB", level="INFO",
           format="{time} {level} {message}", compression="zip")


def process_labeled_dataset(input_file: str, output_dir: str):
    """
    Process labeled dataset and store embeddings incrementally.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize embedder
    api_key=os.getenv("OPENAI_API_KEY")
    base_url=os.getenv("OPENAI_BASE_URL")
    embedder = NLEmbedder(openai_api_key=api_key, base_url=base_url)
    
    # Create output file path
    output_file = os.path.join(output_dir, os.path.basename(input_file))
    
    try:
        # Read input file
        with open(input_file, 'r') as f:
            labeled_data = json.load(f)
        
        # Initialize or load existing processed data
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
                logger.info(f"Skipping already processed example {example.get('id')}")
                continue
                
            embedded_example = embedder.embed_schema_with_relevance(example, embed_method='api_mock')
            processed_data.append(embedded_example)
            
            # Save after each example
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            status = embedded_example.get('status', 'unknown')
            logger.info(f"Processed example {example.get('id')} - Status: {status}")
            
            if status == 'error':
                logger.error(f"Error in example {example.get('id')}: {embedded_example.get('error_message')}")
        
        logger.info(f"Completed processing all examples. Total processed: {len(processed_data)}")
        
    except Exception as e:
        logger.error(f"Fatal error during processing: {str(e)}")
        raise

if __name__ == "__main__":
    input_file = "data/labeled_schema_linking_dataset/spider_dev_labeled.json"
    output_dir = "data/embed_schema_linking_dataset"
    
    process_labeled_dataset(input_file, output_dir)