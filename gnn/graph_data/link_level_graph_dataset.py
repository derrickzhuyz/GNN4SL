import json
import torch
import os
import networkx as nx
import numpy as np
from tqdm import tqdm
from loguru import logger
from typing import Dict, List, Tuple
from torch_geometric.data import Data, Dataset

logger.add("logs/link_level_graph_dataset.log", rotation="1 MB", level="INFO",
           format="{time} {level} {message}", compression="zip")



class LinkLevelGraphDataset(Dataset):
    """
    A link-level graph dataset where edges between question nodes and schema nodes 
    (tables/columns) represent relevance relationships.
    
    There are three types of nodes: question, table, and column; one type of edge.
    
    Args:
        root: Root directory where the dataset should be saved
        dataset_type: One of ['spider', 'bird'] to specify which dataset to use
        split: One of ['train', 'dev'] to specify train or dev split
        embed_method: To specify which embedding method used, to store the graph data in the corresponding subdirectory
        transform: Optional transform to be applied on a sample
    """
    def __init__(self, root, dataset_type='spider', split='train', embed_method: str = None, transform=None):
        self.dataset_type = dataset_type
        self.split = split
        self.embed_method = embed_method
        self.graph_data_dir = os.path.join(root, 'link_level_graph_dataset', embed_method)
        
        # Create the graph data directory if it doesn't exist
        os.makedirs(self.graph_data_dir, exist_ok=True)
        
        # Load raw data with question embeddings during initialization
        path_config = self._load_json('config/path_config.json')
        labeled_path = os.path.join(path_config['embed_question_paths']['embed_question_base'],
                              embed_method,
                              f'{dataset_type}_{split}_labeled.json')
        self.raw_data = self._load_json(labeled_path)
        
        super().__init__(root, transform)

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.dataset_type}_{self.split}_link_level_graph.pt']

    @property
    def processed_dir(self) -> str:
        return self.graph_data_dir


    """
    Create initial schema graph for a database based on data/embed_db_schema/...
    :param database_name: Name of the database
    :return: NetworkX graph and node mapping
    """
    def _create_schema_graph(self, database_name: str) -> Tuple[nx.Graph, Dict]:
        G = nx.Graph()
        node_mapping = {}  # Maps (node_type, node_name) to node index
        node_idx = 0
        
        # Load schema information
        path_config = self._load_json('config/path_config.json')
        schema_filename = ('spider_schemas.json' if self.dataset_type == 'spider' 
                          else f'{self.dataset_type}_{self.split}_schemas.json')
        schema_path = os.path.join(
            path_config['embed_db_schema_paths']['embed_db_schema_base'],
            self.embed_method,
            schema_filename
        )
        schemas = self._load_json(schema_path)
        db_schema = next((schema for schema in schemas if schema['database'] == database_name), None)
        
        if db_schema is None:
            raise ValueError(f"Database {database_name} not found in schema embeddings")

        # Add table and column nodes
        for table in db_schema['tables']:
            table_name = table['table'].lower()
            table_embedding = torch.tensor(table['table_name_embedding'])
            
            # Add table node
            G.add_node(node_idx,
                      name=table_name,
                      type='table',
                      embedding=table_embedding.numpy())
            node_mapping[('table', table_name)] = node_idx
            table_idx = node_idx
            node_idx += 1
            
            # Add column nodes and edges
            for col_name in table['columns']:
                col_name_lower = col_name.lower()
                col_embedding = torch.tensor(table['column_name_embeddings'][col_name])
                
                G.add_node(node_idx,
                          name=col_name_lower,
                          type='column',
                          table=table_name,
                          embedding=col_embedding.numpy())
                node_mapping[('column', col_name_lower)] = node_idx
                
                # Add edge between table and column (no edge type)
                G.add_edge(table_idx, node_idx)
                node_idx += 1

        # Add foreign key edges
        if 'foreign_keys' in db_schema:
            for fk in db_schema['foreign_keys']:
                if fk['column'] is None or any(c is None for c in fk['column']) or any(t is None for t in fk['table']):
                    logger.warning(f"[* Warning] ({self.dataset_type}_{self.split}) Skipping invalid foreign key: {fk}")
                    continue
                try:
                    # Get both table and column names
                    table1_name = fk['table'][0].lower()
                    table2_name = fk['table'][1].lower()
                    col1_name = fk['column'][0].lower()
                    col2_name = fk['column'][1].lower()
                    # Find the correct node indices by matching both table and column
                    col1_idx = None
                    col2_idx = None
                    
                    for i, attr in G.nodes(data=True):
                        if (attr['type'] == 'column' and 
                            attr['name'] == col1_name and 
                            attr['table'] == table1_name):
                            col1_idx = i
                        elif (attr['type'] == 'column' and 
                              attr['name'] == col2_name and 
                              attr['table'] == table2_name):
                            col2_idx = i
                    
                    if col1_idx is not None and col2_idx is not None:
                        # Add foreign key edge
                        G.add_edge(col1_idx, col2_idx, edge_type='foreign_key')
                    else:
                        logger.warning(f"[* Warning] ({self.dataset_type}_{self.split}) Could not find matching columns for foreign key: {fk}")
                except Exception as e:
                    logger.warning(f"[* Warning] ({self.dataset_type}_{self.split}) Error adding foreign key edge in {database_name}: {str(e)}")
                    continue

        return G, node_mapping


    """
    Add question nodes to the graph: question embedding has been pre-computed in data/embed_question_labeled_dataset/...
    :param G: NetworkX graph
    :param example: Dictionary containing the example data
    :param node_mapping: Dictionary mapping (node_type, node_name) to node index
    :return: Updated NetworkX graph
    """
    def _add_question_nodes(self, G: nx.Graph, example: Dict, node_mapping: Dict) -> nx.Graph:
        question = example['question']
        question_idx = len(G.nodes)
        
        # Use pre-computed question embedding
        question_embedding = torch.tensor(example['question_embedding'])
        
        # Add question node
        G.add_node(question_idx,
                  name=question,
                  type='question',
                  embedding=question_embedding.numpy())
        
        # Add edges between question and relevant schema elements
        for table in example['tables']:
            table_name = table['name'].lower()
            if table['relevant']:
                table_idx = node_mapping[('table', table_name)]
                G.add_edge(question_idx, table_idx)
            
            for col in table['columns']:
                col_name = col['name'].lower()
                if col['relevant']:
                    col_idx = node_mapping[('column', col_name)]
                    G.add_edge(question_idx, col_idx)
        
        return G


    """
    Process the raw data to create and save graph data - one graph per database
    containing all relevant question nodes
    """
    def process(self):
        logger.info(f"[i] Processing {len(self.raw_data)} examples for {self.dataset_type}_{self.split} ...")
        data_list = []
        
        # Group examples by database
        db_examples = {}
        for example in self.raw_data:
            db_name = example['database']
            if db_name not in db_examples:
                db_examples[db_name] = []
            db_examples[db_name].append(example)
        
        # Process each database
        for db_name, examples in tqdm(db_examples.items(), desc=f"Creating {self.dataset_type}_{self.split} graphs"):
            try:
                # Create initial schema graph for this database
                G, node_mapping = self._create_schema_graph(db_name)
                
                # Add all question nodes for this database
                for example in examples:
                    G = self._add_question_nodes(G, example, node_mapping)
                
                # Convert final graph to PyG Data object
                edge_index = torch.tensor(list(G.edges)).t().contiguous()
                node_embeddings = [G.nodes[i]['embedding'] for i in range(len(G.nodes))]
                x = torch.tensor(np.array(node_embeddings))
                
                # Store node information
                node_names = [G.nodes[i]['name'] for i in range(len(G.nodes))]
                node_types = [G.nodes[i]['type'] for i in range(len(G.nodes))]
                
                data = Data(x=x,
                           edge_index=edge_index,
                           node_names=node_names,
                           node_types=node_types,
                           database_name=db_name)  # Add database name to the graph
                data_list.append(data)
                
            except Exception as e:
                logger.error(f"[✗] Error processing database {db_name}: {str(e)}")
                continue
        
        # Save processed data
        processed_path = os.path.join(self.graph_data_dir, self.processed_file_names[0])
        os.makedirs(self.graph_data_dir, exist_ok=True)
        torch.save(data_list, processed_path)
        logger.info(f"[✓] Saved {len(data_list)} database graphs to {processed_path}")


    """
    Load a JSON file
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



if __name__ == "__main__":
    # Create datasets
    used_embed_method = 'sentence_transformer' # To store graph data in the subdirectory named by embedding method
    spider_train = LinkLevelGraphDataset(root='data/schema_linking_graph_dataset/', 
                                       dataset_type='spider', 
                                       split='train',
                                       embed_method=used_embed_method)
    spider_dev = LinkLevelGraphDataset(root='data/schema_linking_graph_dataset/', 
                                     dataset_type='spider', 
                                     split='dev',
                                     embed_method=used_embed_method)
    bird_train = LinkLevelGraphDataset(root='data/schema_linking_graph_dataset/', 
                                     dataset_type='bird', 
                                     split='train',
                                     embed_method=used_embed_method)
    bird_dev = LinkLevelGraphDataset(root='data/schema_linking_graph_dataset/', 
                                   dataset_type='bird', 
                                   split='dev',
                                   embed_method=used_embed_method)
