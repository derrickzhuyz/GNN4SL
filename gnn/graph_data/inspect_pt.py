import torch
from torch_geometric.data import HeteroData, Data
import os
from torch.serialization import add_safe_globals
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr


# Add PyTorch Geometric classes to safe globals
add_safe_globals([BaseStorage, NodeStorage, EdgeStorage, HeteroData, Data, DataEdgeAttr, DataTensorAttr])


"""
Inspect the heterogeneous graph data in the .pt file
:param graph: HeteroData object
:param idx: index of the graph to inspect
:return: None
"""
def inspect_hetero_graph(graph: HeteroData, idx: int) -> None:
    # Print basic graph info
    print(f"\n=== Inspecting Heterogeneous Graph: {idx} ===")
    print("\nGraph Structure:")
    print(graph)

    # Inspect table nodes
    print("\nTable Nodes:")
    num_tables = graph['table'].x.size(0)
    print(f"Number of tables: {num_tables}")
    print("Table labels (1=relevant, 0=not relevant):")
    for i in range(num_tables):
        print(f"Table {i}: label={graph['table'].y[i].item()}")
    
    # Inspect column nodes
    print("\nColumn Nodes:")
    num_columns = graph['column'].x.size(0)
    print(f"Number of columns: {num_columns}")
    print("Column labels (1=relevant, 0=not relevant):")
    for i in range(num_columns):
        print(f"Column {i}: label={graph['column'].y[i].item()}")
    
    # Inspect contains edges
    print("\nContains Edges (table -> column):")
    contains_edges = graph['table', 'contains', 'column'].edge_index
    for i in range(contains_edges.size(1)):
        table_idx = contains_edges[0][i].item()
        col_idx = contains_edges[1][i].item()
        print(f"Table {table_idx} contains Column {col_idx}")
    
    # Inspect foreign key edges
    if ('column', 'foreign_key', 'column') in graph.edge_types:
        print("\nForeign Key Edges (column <-> column):")
        fk_edges = graph['column', 'foreign_key', 'column'].edge_index
        for i in range(fk_edges.size(1)):
            col1_idx = fk_edges[0][i].item()
            col2_idx = fk_edges[1][i].item()
            print(f"Column {col1_idx} <-> Column {col2_idx}")


"""
Inspect the homogeneous graph data in the .pt file
:param graph: Data object
:param idx: index of the graph to inspect
:return: None
"""
def inspect_homo_graph(graph: Data, idx: int) -> None:
    # Print basic graph info
    print(f"\n=== Inspecting Homogeneous Graph: {idx} ===")
    print("\nGraph Structure:")
    print(graph)
    
    # Inspect nodes
    num_nodes = graph.x.size(0)
    print(f"\nNumber of nodes: {num_nodes}")
    print(f"Node feature dimension: {graph.x.size(1)}")
    
    print("\nNode labels (1=relevant, 0=not relevant):")
    for i in range(num_nodes):
        print(f"Node {i}: label={graph.y[i].item()}")
    
    # Inspect edges
    print("\nEdges:")
    edge_index = graph.edge_index
    num_edges = edge_index.size(1)
    print(f"Total number of edges: {num_edges}")
    
    # Print first 10 edges as example
    print("\nFirst 10 edges:")
    for i in range(min(10, edge_index.size(1))):
        src_idx = edge_index[0][i].item()
        dst_idx = edge_index[1][i].item()
        print(f"Node {src_idx} -> Node {dst_idx}")
    
    # Print some statistics
    print("\nGraph Statistics:")
    print(f"Average node degree: {num_edges/num_nodes:.2f}")
    print(f"Number of relevant nodes: {graph.y.sum().item()}")
    print(f"Percentage of relevant nodes: {(graph.y.sum().item()/num_nodes)*100:.2f}%")



if __name__ == "__main__":

    # Load processed graph data
    dataset_type = 'spider'  # or 'bird'
    split = 'dev'  # or 'train'
    
    # Inspect heterogeneous graphs
    hetero_path = f'data/hetero_graph_schema_linking_dataset/{dataset_type}_{split}_schema_linking.pt'
    if os.path.exists(hetero_path):
        hetero_graphs = torch.load(hetero_path, weights_only=True)
        num_inspect = 1
        for i in range(min(num_inspect, len(hetero_graphs))):
            inspect_hetero_graph(hetero_graphs[i], i)

    print("\n*******************************************************")

    # Inspect homogeneous graphs
    homo_path = f'data/homo_graph_schema_linking_dataset/{dataset_type}_{split}_schema_linking_homo.pt'
    if os.path.exists(homo_path):
        homo_graphs = torch.load(homo_path, weights_only=False)
        num_inspect = 1
        for i in range(min(num_inspect, len(homo_graphs))):
            inspect_homo_graph(homo_graphs[i], i)