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

def visualize_graph(G: nx.Graph, title: str, save_path: Optional[str] = None):
    """
    Visualize a NetworkX graph with different colors for tables and columns
    """
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Draw nodes
    table_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'table']
    column_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'column']
    
    # Draw relevant/non-relevant nodes with different colors
    relevant_nodes = [n for n, attr in G.nodes(data=True) if attr.get('relevant') == 1]
    non_relevant_nodes = [n for n, attr in G.nodes(data=True) if attr.get('relevant') == 0]
    
    # Draw edges
    contains_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('edge_type') == 'contains']
    fk_edges = [(u, v) for (u, v, d) in G.edges(data=True) if d.get('edge_type') == 'foreign_key']
    
    # Draw nodes with different colors and sizes
    nx.draw_networkx_nodes(G, pos, nodelist=table_nodes, node_color='lightblue', 
                          node_size=1000, label='Tables')
    nx.draw_networkx_nodes(G, pos, nodelist=column_nodes, node_color='lightgreen', 
                          node_size=500, label='Columns')
    
    # Highlight relevant nodes with a red border
    nx.draw_networkx_nodes(G, pos, nodelist=relevant_nodes, node_color='none',
                          node_size=1200, node_shape='o', linewidths=2,
                          edgecolors='red', label='Relevant')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edgelist=contains_edges, edge_color='gray',
                          width=1, label='Contains')
    nx.draw_networkx_edges(G, pos, edgelist=fk_edges, edge_color='red',
                          width=1, style='dashed', label='Foreign Key')
    
    # Add labels
    labels = nx.get_node_attributes(G, 'name')
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title(title)
    plt.legend()
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def visualize_homo_graph(G: nx.Graph, title: str, save_path: Optional[str] = None):
    """
    Visualize a homogeneous NetworkX graph with node labels and relevance information
    """
    plt.figure(figsize=(15, 10))
    
    # Use a better layout for visualization
    pos = nx.kamada_kawai_layout(G)  # or spring_layout with adjusted parameters
    
    # Prepare node colors based on relevance
    node_colors = ['lightcoral' if G.nodes[node]['relevant'] == 1 else 'lightblue' 
                  for node in G.nodes()]
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos,
                          node_color=node_colors,
                          node_size=1000,
                          alpha=0.7)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          width=1,
                          alpha=0.5)
    
    # Prepare node labels with more information
    labels = {}
    for node in G.nodes():
        # Get node embedding norm as a simple representation
        emb_norm = round(float(torch.norm(torch.tensor(G.nodes[node]['embedding']))), 2)
        label = f"Node {node}\n"
        if 'name' in G.nodes[node]:
            label += f"{G.nodes[node]['name']}\n"
        label += f"Rel: {G.nodes[node]['relevant']}\n"
        label += f"Emb: {emb_norm}"
        labels[node] = label
    
    # Draw labels with white background for better visibility
    nx.draw_networkx_labels(G, pos, labels,
                          font_size=8,
                          bbox=dict(facecolor='white',
                                  edgecolor='none',
                                  alpha=0.7,
                                  pad=0.5))
    
    # Add legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 markerfacecolor='lightcoral', markersize=10,
                                 label='Relevant Node'),
                      plt.Line2D([0], [0], marker='o', color='w',
                                 markerfacecolor='lightblue', markersize=10,
                                 label='Non-relevant Node')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(title)
    plt.axis('off')
    
    # Add graph information as text
    info_text = (f"Nodes: {G.number_of_nodes()}\n"
                f"Edges: {G.number_of_edges()}\n"
                f"Relevant Nodes: {sum(1 for _, attr in G.nodes(data=True) if attr['relevant'] == 1)}")
    plt.text(0.02, 0.98, info_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
             verticalalignment='top')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.show()

def inspect_homo_graph_nx(graph: Data, idx: int, visualize: bool = True) -> None:
    """
    Inspect the homogeneous graph data created using NetworkX
    """
    print(f"\n=== Inspecting Homogeneous Graph (NetworkX-based): {idx} ===")
    print("\nGraph Structure:")
    print(graph)
    
    # Convert PyG graph to NetworkX for inspection
    edge_index = graph.edge_index.numpy()
    G = nx.Graph()
    
    # Add nodes with their features and labels
    num_nodes = graph.x.size(0)
    for i in range(num_nodes):
        G.add_node(i, 
                  embedding=graph.x[i].numpy(),
                  relevant=graph.y[i].item())
    
    # Add edges
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i], edge_index[1, i]
        G.add_edge(src, dst)
    
    # Print statistics
    print("\nGraph Statistics:")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"Number of relevant nodes: {graph.y.sum().item()}")
    print(f"Percentage of relevant nodes: {(graph.y.sum().item()/num_nodes)*100:.2f}%")
    
    # Print node information
    print("\nNode Information:")
    for node in G.nodes():
        emb_norm = torch.norm(torch.tensor(G.nodes[node]['embedding'])).item()
        print(f"Node {node}: relevant={G.nodes[node]['relevant']}, embedding_norm={emb_norm:.2f}")
    
    # Print edge information
    print("\nEdge Information:")
    for edge in G.edges():
        print(f"Edge: {edge[0]} <-> {edge[1]}")
    
    # Analyze graph properties
    print("\nGraph Properties:")
    print(f"Is connected: {nx.is_connected(G)}")
    print(f"Number of connected components: {nx.number_connected_components(G)}")
    
    if visualize:
        visualize_homo_graph(G, f"Homogeneous Graph {idx} Structure", 
                           save_path=f"data/visualizations/homo_graph_{idx}.png")


def compare_graphs(original_path: str, nx_path: str, num_samples: int = 1):
    """
    Compare graphs from original implementation and NetworkX implementation
    """
    print(f"\n=== Comparing Original and NetworkX-based Graphs ===")
    
    original_graphs = torch.load(original_path, weights_only=False)
    nx_graphs = torch.load(nx_path, weights_only=False)
    
    for i in range(min(num_samples, len(original_graphs))):
        print(f"\nComparing Graph {i}:")
        
        orig_graph = original_graphs[i]
        nx_graph = nx_graphs[i]
        
        # Compare basic properties
        print("\nStructure Comparison:")
        print(f"Original num nodes: {orig_graph.x.size(0)}")
        print(f"NetworkX num nodes: {nx_graph.x.size(0)}")
        print(f"Original num edges: {orig_graph.edge_index.size(1)}")
        print(f"NetworkX num edges: {nx_graph.edge_index.size(1)}")
        
        # Compare node features and labels
        print("\nFeature Comparison:")
        print(f"Features match: {torch.allclose(orig_graph.x, nx_graph.x)}")
        print(f"Labels match: {torch.allclose(orig_graph.y, nx_graph.y)}")
        
        # Compare edge structure
        print("\nEdge Structure Comparison:")
        orig_edges = set(map(tuple, orig_graph.edge_index.t().tolist()))
        nx_edges = set(map(tuple, nx_graph.edge_index.t().tolist()))
        print(f"Edge sets match: {orig_edges == nx_edges}")
        
        if orig_edges != nx_edges:
            print("Edge differences:")
            print(f"Edges in original but not in NetworkX: {orig_edges - nx_edges}")
            print(f"Edges in NetworkX but not in original: {nx_edges - orig_edges}")

if __name__ == "__main__":
    # Create visualization directory
    os.makedirs("visualizations", exist_ok=True)
    
    dataset_type = 'spider'  # or 'bird'
    split = 'dev'  # or 'train'
    
    # Inspect and compare homogeneous graphs
    original_homo_path = f'data/homo_graph_schema_linking_dataset/{dataset_type}_{split}_schema_linking_homo.pt'
    nx_homo_path = f'data/homo_graph_schema_linking_dataset_nx/{dataset_type}_{split}_schema_linking_homo_nx.pt'
    
    if os.path.exists(original_homo_path) and os.path.exists(nx_homo_path):
        # Compare the two implementations
        # compare_graphs(original_homo_path, nx_homo_path, num_samples=1)
        
        # Inspect NetworkX-based graphs
        nx_graphs = torch.load(nx_homo_path, weights_only=False)
        num_inspect = 1
        for i in range(min(num_inspect, len(nx_graphs))):
            inspect_homo_graph_nx(graph=nx_graphs[i], idx=i, visualize=True) 
