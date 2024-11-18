import json
import torch
import os
import networkx as nx
import numpy as np
from tqdm import tqdm
from loguru import logger
from typing import Dict, List, Tuple
from torch_geometric.data import Data, Dataset

logger.add("logs/schema2graph_dataset_nx.log", rotation="1 MB", level="INFO",
           format="{time} {level} {message}", compression="zip")


# Add PyTorch Geometric classes to safe globals
add_safe_globals([BaseStorage, NodeStorage, EdgeStorage, Data, DataEdgeAttr, DataTensorAttr])



class SchemaLinkingHomoGraphDatasetNX(Dataset):
    """
    A homogeneous graph dataset where all nodes (tables and columns) and edges 
    (contains and foreign keys) are treated as single types.
    
    Args:
        root: Root directory where the dataset should be saved
        dataset_type: One of ['spider', 'bird'] to specify which dataset to use
        split: One of ['train', 'dev'] to specify train or dev split
        transform: Optional transform to be applied on a sample
    """
    def __init__(self, root, dataset_type='spider', split='train', transform=None):
        self.dataset_type = dataset_type
        self.split = split
        self.graph_data_dir = os.path.join(root, 'homo_graph_schema_linking_dataset_nx')
        
        # Load raw data during initialization
        path_config = self._load_json('config/path_config.json')
        labeled_path = os.path.join(path_config['labeled_dataset_paths']['labeled_dataset_base'],
                                  f'{dataset_type}_{split}_labeled.json')
        self.raw_data = self._load_json(labeled_path)
        
        super().__init__(root, transform)

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.dataset_type}_{self.split}_schema_linking_homo_nx.pt']

    @property
    def processed_dir(self) -> str:
        return self.graph_data_dir


    """
    Create a graph from a single example using NetworkX library
    :param example: Dictionary containing the example data
    :return: Graph object
    """
    def _create_graph(self, example: Dict) -> Tuple[nx.Graph, torch.Tensor, torch.Tensor]:
        # Initialize NetworkX graph
        G = nx.Graph()
        
        # Track node information
        nodes = []  # List of node names
        node_labels = []  # Relevance labels for nodes
        node_embeddings = []  # List to store embeddings
        database_name = example['database']
        
        # Add nodes to the graph
        node_idx = 0
        for table in example['tables']:
            # Add table node
            table_name = table['name'].lower()
            nodes.append(table_name)
            node_labels.append(table['relevant'])
            
            # Get table embedding
            table_embedding = self._get_embedding(database_name, table_name, is_table=True)
            node_embeddings.append(table_embedding)
            
            # Add node to NetworkX graph with attributes
            G.add_node(node_idx, 
                      name=table_name, 
                      type='table',
                      embedding=table_embedding.numpy())
            table_idx = node_idx
            node_idx += 1
            
            # Add column nodes
            for col in table['columns']:
                col_name = col['name'].lower()
                nodes.append(col_name)
                node_labels.append(col['relevant'])
                
                # Get column embedding
                col_embedding = self._get_embedding(database_name, col_name, is_table=False)
                node_embeddings.append(col_embedding)
                
                # Add column node to NetworkX graph
                G.add_node(node_idx,
                          name=col_name,
                          type='column',
                          embedding=col_embedding.numpy())
                
                # Add edge between table and column
                G.add_edge(table_idx, node_idx, edge_type='contains')
                node_idx += 1

        # Add foreign key edges
        if 'foreign_keys' in example:
            for fk in example['foreign_keys']:
                if fk['column'] is None or any(c is None for c in fk['column']):
                    logger.warning(f"[* Warning] Skipping invalid foreign key: {fk}")
                    continue
                
                try:
                    # Find indices of the two columns
                    col1_name = fk['column'][0].lower()
                    col2_name = fk['column'][1].lower()
                    col1_idx = [i for i, n in enumerate(nodes) if n == col1_name][0]
                    col2_idx = [i for i, n in enumerate(nodes) if n == col2_name][0]
                    
                    # Add foreign key edge
                    G.add_edge(col1_idx, col2_idx, edge_type='foreign_key')
                except (IndexError, ValueError):
                    continue

        # Convert node embeddings and labels to tensors
        x = torch.stack(node_embeddings)
        y = torch.tensor(node_labels)
        
        return G, x, y


    """
    Process the raw data to create and save graph data
    """
    def process(self):
        path_config = self._load_json('config/path_config.json')
        labeled_path = os.path.join(path_config['labeled_dataset_paths']['labeled_dataset_base'],
                                  f'{self.dataset_type}_{self.split}_labeled.json')
        labeled_data = self._load_json(labeled_path)

        logger.info(f"[i] Processing {len(labeled_data)} examples for {self.dataset_type}_{self.split}")
        data_list = []
        
        for example in tqdm(labeled_data, desc=f"Creating {self.dataset_type}_{self.split} graphs"):
            try:
                G, x, y = self._create_graph(example)
                
                # Convert NetworkX graph to edge_index format
                edge_index = torch.tensor(list(G.edges)).t().contiguous()
                
                # Create PyG Data object
                data = Data(x=x, edge_index=edge_index, y=y)
                data_list.append(data)
                
            except Exception as e:
                logger.error(f"[! Error] Error processing example {example.get('question_id', 'unknown')}: {str(e)}")
                continue

        logger.info(f"Successfully created {len(data_list)} graphs")
        
        # Save processed data
        processed_path = os.path.join(self.graph_data_dir, self.processed_file_names[0])
        os.makedirs(self.graph_data_dir, exist_ok=True)
        torch.save(data_list, processed_path)
        logger.info(f"[i] Saved processed data to {processed_path}")


    """
    Load JSON file
    :param path: Path to the JSON file
    :return: Dictionary containing the JSON data
    """
    def _load_json(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)


    """
    Get the number of graphs in the dataset
    :return: Number of graphs in the dataset
    """
    def len(self) -> int:
        processed_path = os.path.join(self.graph_data_dir, self.processed_file_names[0])
        data_list = torch.load(processed_path, weights_only=False)
        return len(data_list)


    """
    Get the graph at a given index
    :param idx: Index of the graph to retrieve
    :return: Graph at the given index
    """
    def get(self, idx: int) -> Data:
        processed_path = os.path.join(self.graph_data_dir, self.processed_file_names[0])
        data_list = torch.load(processed_path, weights_only=False)
        return data_list[idx]


    """
    Retrieve embedding for a table or column name from the stored schema embeddings.
    :param database_name: Name of the database (e.g., 'concert_singer')
    :param name: Name of the table or column to retrieve embedding for
    :param is_table: Boolean indicating whether the name refers to a table or column
    :return: Embedding vector for the table's or column's name
    """
    def _get_embedding(self, database_name: str, name: str, is_table: bool) -> torch.Tensor:
        path_config = self._load_json('config/path_config.json')
        
        # For Spider dataset, always use spider_schemas.json regardless of split; for Bird dataset, use the split version
        schema_filename = ('spider_schemas.json' if self.dataset_type == 'spider' 
                          else f'{self.dataset_type}_{self.split}_schemas.json')
        
        schema_path = os.path.join(
            path_config['embed_db_schema_paths']['embed_db_schema_base'], 
            schema_filename
        )
        schemas = self._load_json(schema_path)
        
        # Find the database schema
        db_schema = next((schema for schema in schemas if schema['database'] == database_name), None)
        if db_schema is None:
            raise ValueError(f"Database {database_name} not found in schema embeddings")
        
        if is_table:
            # Find table embedding
            table = next((t for t in db_schema['tables'] if t['table'].lower() == name.lower()), None)
            if table is None:
                raise ValueError(f"Table {name} not found in database {database_name}")
            return torch.tensor(table['table_name_embedding'])
        else:
            # Find column embedding
            for table in db_schema['tables']:
                # Find the original case-sensitive column name
                original_name = next((col for col in table['columns'] 
                                    if col.lower() == name.lower()), None)
                if original_name:
                    return torch.tensor(table['column_name_embeddings'][original_name])
            
            raise ValueError(f"Column {name} not found in database {database_name}")



if __name__ == "__main__":
    # Create datasets using NetworkX version
    # spider_train = SchemaLinkingHomoGraphDatasetNX(root='data/', dataset_type='spider', split='train')
    spider_dev = SchemaLinkingHomoGraphDatasetNX(root='data/', dataset_type='spider', split='dev')
    # bird_train = SchemaLinkingHomoGraphDatasetNX(root='data/', dataset_type='bird', split='train')
    # bird_dev = SchemaLinkingHomoGraphDatasetNX(root='data/', dataset_type='bird', split='dev') 