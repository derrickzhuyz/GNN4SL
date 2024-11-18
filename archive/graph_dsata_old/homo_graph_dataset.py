import json
import torch
import os
from tqdm import tqdm
from loguru import logger
from typing import Dict, List
from torch_geometric.data import Data, Dataset
from torch.serialization import add_safe_globals
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage
from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr

logger.add("logs/schema2graph_dataset.log", rotation="1 MB", level="INFO", 
           format="{time} {level} {message}", compression="zip")


# Add PyTorch Geometric classes to safe globals
add_safe_globals([BaseStorage, NodeStorage, EdgeStorage, Data, DataEdgeAttr, DataTensorAttr])


class SchemaLinkingHomoGraphDataset(Dataset):
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
        self.graph_data_dir = os.path.join(root, 'homo_graph_schema_linking_dataset')
        
        # Load raw data during initialization
        path_config = self._load_json('config/path_config.json')
        labeled_path = os.path.join(path_config['labeled_dataset_paths']['labeled_dataset_base'], 
                                  f'{dataset_type}_{split}_labeled.json')
        self.raw_data = self._load_json(labeled_path)
        
        super().__init__(root, transform)

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{self.dataset_type}_{self.split}_labeled.json']

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.dataset_type}_{self.split}_schema_linking_homo.pt']

    @property
    def processed_dir(self) -> str:
        return self.graph_data_dir


    """
    Process the raw data to create a list of graphs
    """
    def process(self):
        path_config = self._load_json('config/path_config.json')
        labeled_path = os.path.join(path_config['labeled_dataset_paths']['labeled_dataset_base'], self.raw_file_names[0])
        labeled_data = self._load_json(labeled_path)

        logger.info(f"[i] Processing {len(labeled_data)} examples for {self.dataset_type}_{self.split}")
        data_list = []
        
        # Add progress bar
        for example in tqdm(labeled_data, desc=f"Creating {self.dataset_type}_{self.split} graphs"):
            try:
                graph = self._create_graph(example)
                if graph is not None:
                    data_list.append(graph)
            except Exception as e:
                logger.error(f"[! Error] Error processing example {example.get('question_id', 'unknown')}: {str(e)}")
                continue

        logger.info(f"Successfully created {len(data_list)} graphs")
        
        processed_path = os.path.join(self.graph_data_dir, self.processed_file_names[0])
        os.makedirs(self.graph_data_dir, exist_ok=True)
        torch.save(data_list, processed_path)
        logger.info(f"[i] Saved processed data to {processed_path}")


    """
    Create a graph from a single example
    :param example: Dictionary containing the example data
    :return: Graph object
    """
    def _create_graph(self, example: Dict) -> Data:
        # Create nodes (combining tables and columns into a single list)
        nodes = []  # List of node names
        node_labels = []  # Relevance labels for nodes
        node_embeddings = []  # List to store embeddings
        database_name = example['database']
        
        # Add all nodes (tables and columns) to a single list
        for table in example['tables']:
            # Add table node
            table_name = table['name'].lower()
            nodes.append(table_name)
            node_labels.append(table['relevant'])
            # Get table embedding
            table_embedding = self._get_embedding(database_name, table_name, is_table=True)
            node_embeddings.append(table_embedding)
            
            # Add column nodes
            for col in table['columns']:
                col_name = col['name'].lower()
                nodes.append(col_name)
                node_labels.append(col['relevant'])
                # Get column embedding
                col_embedding = self._get_embedding(database_name, col_name, is_table=False)
                node_embeddings.append(col_embedding)

        # Stack embeddings to create node features
        x = torch.stack(node_embeddings)
        y = torch.tensor(node_labels)

        # Create edges
        edge_index = []
        
        # Add schema structure edges
        for table_idx, table in enumerate(example['tables']):
            table_name = table['name'].lower()
            # Connect table with its columns
            for col in table['columns']:
                try:
                    col_idx = nodes.index(col['name'].lower())
                    # Add edges for table containing column
                    edge_index.append([table_idx, col_idx])
                    # edge_index.append([col_idx, table_idx])
                except ValueError:
                    continue

        # Add foreign key edges
        if 'foreign_keys' in example:
            for fk in example['foreign_keys']:
                if fk['column'] is None or any(c is None for c in fk['column']):
                    logger.warning(f"[* Warning] Skipping invalid foreign key: {fk}")
                    continue
                
                try:
                    # Find indices of the two columns
                    col1_idx = nodes.index(fk['column'][0].lower())
                    col2_idx = nodes.index(fk['column'][1].lower())
                    # Add edges for foreign key relationship between two columns
                    edge_index.append([col1_idx, col2_idx])
                    # edge_index.append([col2_idx, col1_idx])
                except ValueError:
                    continue

        if edge_index:
            edge_index = torch.tensor(edge_index).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)


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
    # Create datasets (labeled homogeneous graph data)
    spider_train = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='spider', split='train')
    spider_dev = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='spider', split='dev')
    bird_train = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='bird', split='train')
    bird_dev = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='bird', split='dev')


# if __name__ == "__main__":
#     # Test the _get_embedding function
#     dataset = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='spider', split='dev')
    
#     try:
#         # Test table embedding
#         table_emb = dataset._get_embedding('concert_singer', 'stadium', is_table=True)
#         print("Successfully retrieved table embedding for 'stadium':")
#         print(table_emb)
#         print(f"Embedding shape: {table_emb.shape}")
        
#         # Test column embedding
#         col_emb = dataset._get_embedding('concert_singer', 'Stadium_ID', is_table=False)
#         print("\nSuccessfully retrieved column embedding for 'Stadium_ID':")
#         print(col_emb)
#         print(f"Embedding shape: {col_emb.shape}")
        
#         # Test error cases
#         try:
#             dataset._get_embedding('invalid_db', 'stadium', is_table=True)
#         except ValueError as e:
#             print(f"\nExpected error for invalid database: {e}")
            
#         try:
#             dataset._get_embedding('concert_singer', 'invalid_table', is_table=True)
#         except ValueError as e:
#             print(f"Expected error for invalid table: {e}")
            
#         try:
#             dataset._get_embedding('concert_singer', 'invalid_column', is_table=False)
#         except ValueError as e:
#             print(f"Expected error for invalid column: {e}")
            
#     except Exception as e:
#         print(f"Unexpected error occurred: {e}")