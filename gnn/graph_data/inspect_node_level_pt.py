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
Visualize a node-level NetworkX graph with node labels and relevance information
:param G: NetworkX graph
:param title: title of the graph
:param save_path: path to save the graph
"""
def visualize_node_level_graph(G: nx.Graph, title: str, save_path: Optional[str] = None):

    plt.figure(figsize=(15, 10))
    
    # Use a better layout for visualization
    pos = nx.kamada_kawai_layout(G)  # or spring_layout with adjusted parameters
    
    # Separate nodes by type and relevance
    table_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'table']
    column_nodes = [n for n, attr in G.nodes(data=True) if attr['type'] == 'column']
    relevant_nodes = [n for n in G.nodes() if G.nodes[n]['relevant'] == 1]
    
    # Draw table nodes (light blue)
    nx.draw_networkx_nodes(G, pos,
                          nodelist=[n for n in table_nodes if n not in relevant_nodes],
                          node_color='lightblue',
                          node_size=1000,
                          alpha=0.75)
    
    # Draw column nodes (light green)
    nx.draw_networkx_nodes(G, pos,
                          nodelist=[n for n in column_nodes if n not in relevant_nodes],
                          node_color='lightgreen',
                          node_size=1000,
                          alpha=0.75)
    
    # Draw relevant table nodes (light blue with red border)
    relevant_table_nodes = [n for n in relevant_nodes if G.nodes[n]['type'] == 'table']
    nx.draw_networkx_nodes(G, pos,
                          nodelist=relevant_table_nodes,
                          node_color='lightblue',
                          node_size=1000,
                          edgecolors='red',
                          linewidths=3,
                          alpha=0.75)
    
    # Draw relevant column nodes (light green with red border)
    relevant_column_nodes = [n for n in relevant_nodes if G.nodes[n]['type'] == 'column']
    nx.draw_networkx_nodes(G, pos,
                          nodelist=relevant_column_nodes,
                          node_color='lightgreen',
                          node_size=1000,
                          edgecolors='red',
                          linewidths=3,
                          alpha=0.75)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos,
                          edge_color='gray',
                          width=1,
                          alpha=0.8)
    
    # Prepare node labels with more information
    labels = {}
    for node in G.nodes():
        # Get the first embedding value and format it to 3 decimal places
        first_embedding_value = G.nodes[node]['embedding'][0]  # Get the first embedding value
        label = f"{G.nodes[node]['name']}\n"
        # label += f"({G.nodes[node]['type']})\n"
        label += f"Rel: {G.nodes[node]['relevant']}\n"
        label += f"Emb: \n{first_embedding_value:.3f}"  # Format to 3 decimal places
        labels[node] = label
    
    # Draw labels with white background for better visibility
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
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                   markeredgecolor='red', markeredgewidth=3,
                   markersize=9, label='Relevant Node')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.title(title)
    plt.axis('off')
    
    # Add graph information as text
    info_text = (f"Total Nodes: {G.number_of_nodes()}\n"
                f"  - Tables: {len(table_nodes)}\n"
                f"  - Columns: {len(column_nodes)}\n"
                f"Total Relevant Nodes: {len(relevant_nodes)}\n"
                f"  - Relevant Tables: {len(relevant_table_nodes)}\n"
                f"  - Relevant Columns: {len(relevant_column_nodes)}\n"
                f"Total Edges: {G.number_of_edges()}\n"
                f"Note: Emb is the 1st value of embedding vector.")
    plt.text(0.0, 0.02, info_text,
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
             verticalalignment='top')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)


"""
Inspect the node-level graph data created by NetworkX
:param graph: PyG Data object
:param idx: index of the graph
:param visualize: whether to visualize the graph
"""
def inspect_node_level_graph(graph: Data, idx: int, visualize: bool = True, dataset: str = '', split: str = '') -> None:
    print(f"\n=== Inspecting Node-level Graph: {idx} ===")
    print("\nGraph Structure:")
    print(graph)
    
    # Convert PyG graph to NetworkX for inspection
    edge_index = graph.edge_index.numpy()
    G = nx.Graph()
    
    # Add nodes with their features, labels, and names
    num_nodes = graph.x.size(0)
    for i in range(num_nodes):
        G.add_node(i, 
                  embedding=graph.x[i].numpy(),
                  relevant=graph.y[i].item(),
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
    print(f"Number of relevant nodes: {graph.y.sum().item()}")
    print(f"Percentage of relevant nodes: {(graph.y.sum().item()/num_nodes)*100:.2f}%")
    
    # Print detailed node information
    print("\nNode Information:")
    for node in G.nodes():
        node_data = G.nodes[node]
        embedding_values = node_data['embedding'][:3]  # Get the first three embedding values
        print(f"Node {node}: {node_data['name']} ({node_data['type']})")
        print(f"  - Relevant: {node_data['relevant']}")
        print(f"  - Embedding (first three values): {embedding_values}")  # Print the first three embedding values directly
    
    # Print edge information with node names
    print("\nEdge Information:")
    for edge in G.edges():
        src_name = G.nodes[edge[0]]['name']
        dst_name = G.nodes[edge[1]]['name']
        print(f"Edge: {edge[0]}({src_name}) <-> {edge[1]}({dst_name})")
    
    # Analyze graph properties
    print("\nGraph Properties:")
    print(f"Is connected: {nx.is_connected(G)}")
    print(f"Number of connected components: {nx.number_connected_components(G)}")
    
    if visualize:
        visualize_node_level_graph(G, title=f"Graph structure for {dataset} {split}: Example No. {idx}", 
                           save_path=f"data/schema_linking_graph_dataset/visualizations/node_graph_{dataset}_{split}_idx_{idx}.png")



if __name__ == "__main__":
    # Create visualization directory
    os.makedirs("visualizations", exist_ok=True)
    
    dataset_type = 'spider'  # 'spider' or 'bird'
    split_type = 'train'  # 'dev' or 'train'
    idx_to_inspect = 0
    
    # Inspect and compare node-level graphs
    nx_node_path = f'data/schema_linking_graph_dataset/node_level_graph_dataset/{dataset_type}_{split_type}_node_level_graph.pt'
    
    if os.path.exists(nx_node_path):
        # Inspect NetworkX-based graphs
        nx_graphs = torch.load(nx_node_path, weights_only=False)
        inspect_node_level_graph(graph=nx_graphs[idx_to_inspect], idx=idx_to_inspect, visualize=True, dataset=dataset_type, split=split_type)
