import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import HeteroData, Data
import os
from torch.serialization import add_safe_globals
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage
from torch_geometric.data.hetero_data import HeteroData
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr
from typing import Optional

# Add PyTorch Geometric classes to safe globals
add_safe_globals([BaseStorage, NodeStorage, EdgeStorage, HeteroData, Data, DataEdgeAttr, DataTensorAttr])


"""
Visualize link-level graph
:param G: NetworkX graph
:param title: Title of the graph
:param save_path: Path to save the graph visualization
"""
def visualize_link_level_graph(G: nx.Graph, title: str, save_path: Optional[str] = None):
    plt.figure(figsize=(25, 18))
    pos = nx.kamada_kawai_layout(G)
    
    # Separate nodes by type
    table_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'table']
    column_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'column']
    question_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'question']
    
    # Draw table nodes (light blue)
    nx.draw_networkx_nodes(G, pos,
                          nodelist=table_nodes,
                          node_color='lightblue',
                          node_size=1000,
                          alpha=0.75)
    
    # Draw column nodes (light green)
    nx.draw_networkx_nodes(G, pos,
                          nodelist=column_nodes,
                          node_color='lightgreen',
                          node_size=1000,
                          alpha=0.75)
    
    # Draw question nodes (light red)
    nx.draw_networkx_nodes(G, pos,
                          nodelist=question_nodes,
                          node_color='lightcoral',
                          node_size=1000,
                          alpha=0.6)
    
    # Separate edges by their endpoints
    schema_edges = [(u, v) for (u, v) in G.edges() 
                   if G.nodes[u]['type'] != 'question' and G.nodes[v]['type'] != 'question']
    relevance_edges = [(u, v) for (u, v) in G.edges() 
                      if G.nodes[u]['type'] == 'question' or G.nodes[v]['type'] == 'question']
    
    # Draw schema edges (gray)
    nx.draw_networkx_edges(G, pos,
                          edgelist=schema_edges,
                          edge_color='gray',
                          width=1,
                          alpha=0.8)
    
    # Draw relevance edges (red)
    nx.draw_networkx_edges(G, pos,
                          edgelist=relevance_edges,
                          edge_color='red',
                          width=0.8,
                          alpha=0.4)
    
    # Prepare node labels
    labels = {}
    for node in G.nodes():
        first_embedding_value = G.nodes[node]['embedding'][0]
        node_name = G.nodes[node]['name']
        label = f"{node_name[:20]}{'..' if len(node_name) > 20 else ''}\n"
        label += f"Emb: \n{first_embedding_value:.3f}"
        labels[node] = label
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, labels,
                          font_size=8,
                          bbox=dict(facecolor='white',
                                  edgecolor='none',
                                  alpha=0.2,
                                  pad=0.5))
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                   markersize=12, label='Table Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
                   markersize=12, label='Column Node'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral', 
                   markersize=12, label='Question Node'),
        plt.Line2D([0], [0], color='gray', label='Schema Edge'),
        plt.Line2D([0], [0], color='red', label='Relevance Edge')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.title(title)
    plt.axis('off')
    
    # Add graph information
    info_text = (f"Total Nodes: {G.number_of_nodes()}\n"
                f"  - Tables: {len(table_nodes)}\n"
                f"  - Columns: {len(column_nodes)}\n"
                f"  - Questions: {len(question_nodes)}\n"
                f"Total Edges: {G.number_of_edges()}\n"
                f"  - Schema Edges: {len(schema_edges)}\n"
                f"  - Relevance Edges: {len(relevance_edges)}\n"
                f"Note: Emb is the 1st value of embedding vector.")
    plt.text(0.0, 0.02, info_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
             verticalalignment='top')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)


"""
Inspect link-level graph data created by NetworkX
:param graph: PyG Data object
:param idx: index of the graph
:param visualize: whether to visualize the graph
"""
def inspect_link_level_graph(graph: Data, idx: int, visualize: bool = True, dataset: str = '', split: str = '') -> None:
    print(f"\n=== Inspecting Link-level Graph: {idx} ===")
    print("\nGraph Structure:")
    print(graph)
    
    # Convert PyG graph to NetworkX
    edge_index = graph.edge_index.numpy()
    G = nx.Graph()
    
    # Add nodes with their features and names
    num_nodes = graph.x.size(0)
    for i in range(num_nodes):
        G.add_node(i, 
                  embedding=graph.x[i].numpy(),
                  name=graph.node_names[i],
                  type=graph.node_types[i])
    
    # Add edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(src, dst)
    
    # Print statistics
    print("\nGraph Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    
    # Count nodes by type
    node_types = {}
    for node in G.nodes():
        node_type = G.nodes[node]['type']
        node_types[node_type] = node_types.get(node_type, 0) + 1
    print("\nNode Types Distribution:")
    for node_type, count in node_types.items():
        print(f"  - {node_type}: {count}")
    
    # Print detailed node information
    print("\nNode Information:")
    for node in G.nodes():
        node_data = G.nodes[node]
        embedding_values = node_data['embedding'][:3]
        print(f"Node {node}: {node_data['name']} ({node_data['type']})")
        print(f"  - Embedding (first three values): {embedding_values}")
    
    # Print edge information
    print("\nEdge Information:")
    for edge in G.edges():
        src_name = G.nodes[edge[0]]['name']
        src_type = G.nodes[edge[0]]['type']
        dst_name = G.nodes[edge[1]]['name']
        dst_type = G.nodes[edge[1]]['type']
        print(f"Edge: {edge[0]}({src_type}:{src_name}) <-> {edge[1]}({dst_type}:{dst_name})")
    
    if visualize:
        # Use the database name in the title and filename
        db_name = graph.database_name
        visualize_link_level_graph(G, title=f"Graph structure for {dataset} {split}: '{db_name}' database", 
                                 save_path=f"data/schema_linking_graph_dataset/visualizations/link_graph_{dataset}_{split}_idx_{idx}_{db_name}.png")


"""
Count the number of graphs in a .pt file.
:param file_path: Path to the .pt file.
:return: Number of graphs in the file.
"""
def count_graphs_in_file(file_path: str) -> int:

    try:
        data_list = torch.load(file_path, weights_only=False)
        return len(data_list)
    except Exception as e:
        print(f"Error loading file {file_path}: {str(e)}")
        return 0



if __name__ == "__main__":
    # Create visualization directory
    os.makedirs("data/schema_linking_graph_dataset/visualizations", exist_ok=True)
    
    dataset_type = 'spider'  # 'spider' or 'bird'
    split_type = 'train'  # 'dev' or 'train'
    idx_to_inspect = 0
    
    # Path to the link-level graph dataset
    nx_link_path = f'data/schema_linking_graph_dataset/link_level_graph_dataset/{dataset_type}_{split_type}_link_level_graph.pt'
    
    # Inspect link-level graphs if the file exists
    if os.path.exists(nx_link_path):
        nx_graphs = torch.load(nx_link_path, weights_only=False)
        inspect_link_level_graph(graph=nx_graphs[idx_to_inspect], idx=idx_to_inspect,
                               visualize=True, dataset=dataset_type, split=split_type)
    
    # Count the number of graphs in the .pt file
    num_graphs = count_graphs_in_file(nx_link_path)
    print('======================================================================')
    print(f"Number of graphs in {nx_link_path}: {num_graphs}")