import json


"""
Count the number of objects in a JSON file
:param json_path: str, the path to the JSON file
:return: int, the number of objects in the JSON file
"""
def count_json_objects(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return len(data)


"""
Find and print the overlap and unique databases between two schema JSON files
:param file_path_1: str, the path to the first JSON file
:param file_path_2: str, the path to the second JSON file
"""
def find_database_overlap(file_path_1, file_path_2):
    # Load JSON data from the first file
    with open(file_path_1, 'r') as file:
        data_1 = json.load(file)
    
    # Load JSON data from the second file
    with open(file_path_2, 'r') as file:
        data_2 = json.load(file)
    
    # Extract database names from both datasets
    databases_1 = {entry['database'] for entry in data_1}
    databases_2 = {entry['database'] for entry in data_2}
    
    # Find overlap and unique databases
    overlap = databases_1.intersection(databases_2)
    only_in_1 = databases_1 - databases_2
    only_in_2 = databases_2 - databases_1
    
    # Print results in a structured format
    print("Database Overlap and Unique Databases:")
    print("=" * 40)
    
    print(f"Total Overlapping Databases: {len(overlap)}")
    print("Overlapping Databases:")
    if overlap:
        for db in overlap:
            print(f"- {db}")
    else:
        print("No overlapping databases found.")
    
    print(f"\nTotal Databases only in {file_path_1}: {len(only_in_1)}")
    print("Databases only in File 1:")
    if only_in_1:
        for db in only_in_1:
            print(f"- {db}")
    else:
        print("No databases found only in File 1.")
    
    print(f"\nTotal Databases only in {file_path_2}: {len(only_in_2)}")
    print("Databases only in File 2:")
    if only_in_2:
        for db in only_in_2:
            print(f"- {db}")
    else:
        print("No databases found only in File 2.")



if __name__ == "__main__":
    # # Check spider train and dev for overlap
    # find_database_overlap('data/gold_schema_linking/spider_dev_gold_schema_linking.json', 
    #                     'data/gold_schema_linking/spider_train_gold_schema_linking.json')
    # # Check bird train and dev for overlap
    # find_database_overlap('data/gold_schema_linking/bird_dev_gold_schema_linking.json', 
    #                     'data/gold_schema_linking/bird_train_gold_schema_linking.json')
    
    pass