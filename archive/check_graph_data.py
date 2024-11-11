from graph_dataset import SchemaLinkingGraphDataset
import json
import os


def load_schema(dataset_type, split):
    schema_file = 'spider_schemas.json' if dataset_type == 'spider' else f'bird_{split}_schemas.json'
    schema_path = os.path.join('data/db_schema', schema_file)
    with open(schema_path, 'r') as f:
        schemas = json.load(f)
    return schemas

def get_node_names(graph_idx, dataset_type, split):
    schemas = load_schema(dataset_type, split)
    
    # Load question file to get database name
    linking_file = f'{dataset_type}_{split}_gold_schema_linking.json'
    with open(os.path.join('data/gold_schema_linking', linking_file), 'r') as f:
        questions = json.load(f)
    
    db_name = questions[graph_idx]['database']
    schema = next(s for s in schemas if s['database'] == db_name)
    
    # Get table names
    table_names = [t['table'] for t in schema['tables']]
    
    # Get column names with their table prefixes
    column_names = []
    for table in schema['tables']:
        for col in table['columns']:
            column_names.append(f"{table['table']}.{col}")
    
    return table_names, column_names

# Create separate datasets
spider_train = SchemaLinkingGraphDataset(root='data/', dataset_type='spider', split='train')
spider_dev = SchemaLinkingGraphDataset(root='data/', dataset_type='spider', split='dev')
bird_train = SchemaLinkingGraphDataset(root='data/', dataset_type='bird', split='train')
bird_dev = SchemaLinkingGraphDataset(root='data/', dataset_type='bird', split='dev')

# Example: get a graph from Spider training set
# graph_idx = 0  # Change this to check different questions
# dataset_type = 'spider'
# split = 'dev'
# graph = spider_dev[graph_idx]
# print(graph)

graph_idx = 0
dataset_type = 'bird'
split = 'train'
graph = bird_train[graph_idx]
print(graph)

# Get actual node names
table_names, column_names = get_node_names(graph_idx, dataset_type, split)

# Get the number of foreign key edges
fk_edges = graph['column', 'foreign_key', 'column'].edge_index
num_fk_edges = fk_edges.size(1)
print(f"\nNumber of foreign key edges: {num_fk_edges}")

# Display specific foreign key edges with their names
print("Foreign key edges (source <-> target):")
for i in range(num_fk_edges):
    src = fk_edges[0][i].item()
    dst = fk_edges[1][i].item()
    print(f"  Edge {i}: {column_names[src]} <-> {column_names[dst]}")

# Get the number of contains edges
contains_edges = graph['table', 'contains', 'column'].edge_index
num_contains_edges = contains_edges.size(1)
print(f"\nNumber of contains edges: {num_contains_edges}")

# Display specific contains edges with their names
print("Contains edges (table -> column):")
for i in range(num_contains_edges):
    src = contains_edges[0][i].item()
    dst = contains_edges[1][i].item()
    print(f"  Edge {i}: {table_names[src]} -> {column_names[dst]}")