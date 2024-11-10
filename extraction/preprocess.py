import json
from loguru import logger

logger.add("logs/extraction.log", rotation="1 MB", level="INFO", 
           format="{time} {level} {message}", compression="zip")

"""This file should be used before entire database schema extraction and before gold schema linking has been done and stored"""


"""
Extract gold SQL for the spider_train and spider_train_others datasets
:param input_file: path to the input file, i.e. info file, train_spider.json or train_others.json
:param output_file: path to the output file, i.e. gold sql file, train_gold.sql or train_others_gold.sql
"""
def extract_sql_from_train_others(input_file, output_file):
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)

        output_data = []
        for entry in data:
            db_id = entry.get('db_id')
            sql_query = entry.get('query')
            if db_id and sql_query:
                output_data.append(f"{sql_query}\t{db_id}")

        with open(output_file, 'w') as f:
            f.write("\n".join(output_data))
        
        logger.info(f"[i] Extracted {len(output_data)} SQL queries to {output_file}")
    except Exception as e:
        logger.error(f"[! Error] An error occurred: {e}")



if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)

    # For Spider (training): Combine info train_spider.json and train_others.json into full_train_info.json, then extract gold SQL to full_train_gold.sql
    try:
        with open(config['spider_paths']['train_info'], 'r') as f:
            train_spider_data = json.load(f)

        with open(config['spider_paths']['train_others_info'], 'r') as f:
            train_others_data = json.load(f)

        combined_data = train_spider_data + train_others_data

        with open(config['spider_paths']['full_train_info'], 'w') as f:
            json.dump(combined_data, f, indent=2)

        logger.info(f"[i] Combined JSON data has been saved to {config['spider_paths']['full_train_info']}")
    except Exception as e:
        logger.error(f"[! Error] An error occurred while combining JSON files: {e}")
    
    try:
        extract_sql_from_train_others(config['spider_paths']['full_train_info'], config['spider_paths']['full_train_gold_sql'])
    except Exception as e:
        logger.error(f"[! Error] An error occurred while extracting SQL from training set: {e}")
    