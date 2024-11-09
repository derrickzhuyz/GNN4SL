import torch
from dataset import SchemaLinkingDataset
import json
import os

def load_original_data(dataset_type, split):
    schema_file = 'spider_schemas.json' if dataset_type == 'spider' else f'bird_{split}_schemas.json'
    linking_file = f'{dataset_type}_{split}_gold_schema_linking.json'
    
    with open(os.path.join('data/db_schema', schema_file), 'r') as f:
        schemas = json.load(f)
    with open(os.path.join('data/gold_schema_linking', linking_file), 'r') as f:
        questions = json.load(f)
    
    return schemas, questions

def analyze_graph(graph, question, schema):
    """Analyze a single graph's structure"""
    print(f"\n=== Graph Analysis for Question ===")
    print(f"Question: {question['question']}")
    print(f"Database: {question['database']}")
    
    # Node information
    print("\n=== Node Information ===")
    n_tables = graph['table'].x.size(0)
    n_columns = graph['column'].x.size(0)
    print(f"Number of table nodes: {n_tables}")
    print(f"Number of column nodes: {n_columns}")
    
    # Label distribution
    table_labels = graph['table'].y.tolist()
    column_labels = graph['column'].y.tolist()
    
    print("\n=== Table Nodes ===")
    for i, label in enumerate(table_labels):
        table_name = schema['tables'][i]['table']
        print(f"Table {i}: {table_name} {'✓' if label == 1 else '✗'}")
    
    print("\n=== Column Nodes ===")
    col_idx = 0
    for i, table in enumerate(schema['tables']):
        table_name = table['table']
        for j, col in enumerate(table['columns']):
            print(f"Column {col_idx}: {table_name}.{col} {'✓' if column_labels[col_idx] == 1 else '✗'}")
            col_idx += 1
    
    # Edge information
    print("\n=== Edge Information ===")
    if 'table__contains__column' in graph.edge_index_dict:
        contains_edges = graph.edge_index_dict['table__contains__column']
        print(f"\nContains Edges (table -> column): {contains_edges.size(1)} edges")
        print("Edge index tensor shape:", contains_edges.shape)
        print("Edge indices:")
        for i in range(contains_edges.size(1)):
            src = contains_edges[0][i].item()
            dst = contains_edges[1][i].item()
            src_table = schema['tables'][src]['table']
            dst_col = None
            col_count = 0
            for table in schema['tables']:
                if col_count + len(table['columns']) > dst:
                    dst_col = table['columns'][dst - col_count]
                    break
                col_count += len(table['columns'])
            print(f"  {i}: {src_table}({src}) -> {dst_col}({dst})")
    else:
        print("No contains edges found!")
    
    if 'column__foreign_key__column' in graph.edge_index_dict:
        fk_edges = graph.edge_index_dict['column__foreign_key__column']
        print(f"\nForeign Key Edges: {fk_edges.size(1)} edges")
        print("Edge index tensor shape:", fk_edges.shape)
        print("Edge indices:")
        for i in range(fk_edges.size(1)):
            src = fk_edges[0][i].item()
            dst = fk_edges[1][i].item()
            # Map indices back to column names
            src_col = None
            dst_col = None
            col_count = 0
            for table in schema['tables']:
                if col_count + len(table['columns']) > src:
                    src_col = f"{table['table']}.{table['columns'][src - col_count]}"
                if col_count + len(table['columns']) > dst:
                    dst_col = f"{table['table']}.{table['columns'][dst - col_count]}"
                if src_col and dst_col:
                    break
                col_count += len(table['columns'])
            print(f"  {i}: {src_col}({src}) <-> {dst_col}({dst})")
    else:
        print("No foreign key edges found!")

def main():
    # Set your parameters here
    dataset_type = 'spider'  # 'spider' or 'bird'
    split = 'train'           # 'train' or 'dev'
    question_id = 100           # Specify the question ID to analyze

    # Load dataset
    dataset = SchemaLinkingDataset(root='data/', dataset_type=dataset_type, split=split)
    schemas, questions = load_original_data(dataset_type, split)
    
    if question_id >= len(dataset):
        print(f"Error: question_id {question_id} is out of range. Dataset has {len(dataset)} questions.")
        return
    
    # Analyze specific question
    graph = dataset[question_id]
    question = questions[question_id]
    schema = next(s for s in schemas if s['database'] == question['database'])
    analyze_graph(graph, question, schema)

if __name__ == "__main__":
    main()