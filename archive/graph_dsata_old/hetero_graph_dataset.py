import json
import torch
import os
from loguru import logger
from typing import Dict, List, Set, Tuple
from torch_geometric.data import HeteroData, Dataset
from torch.serialization import add_safe_globals
from torch_geometric.data.storage import BaseStorage, NodeStorage, EdgeStorage


logger.add("logs/schema2graph_dataset.log", rotation="1 MB", level="INFO", 
           format="{time} {level} {message}", compression="zip")


# Add PyTorch Geometric classes to safe globals
add_safe_globals([BaseStorage, NodeStorage, EdgeStorage, HeteroData])


"""Graph Dataset Class for Schema Linking"""
class SchemaLinkingHeteroGraphDataset(Dataset):

    """
    Args:
        root: Root directory where the dataset should be saved
        dataset_type: One of ['spider', 'bird'] to specify which dataset to use
        split: One of ['train', 'dev'] to specify train or dev split
        transform: Optional transform to be applied on a sample
    """ 
    def __init__(self, root, dataset_type='spider', split='train', transform=None):
        self.dataset_type = dataset_type
        self.split = split
        self.graph_data_dir = os.path.join(root, 'hetero_graph_schema_linking_dataset')
        super().__init__(root, transform)
    
    """
    Property methods:
    - raw_file_names: The names of the raw files
    - processed_file_names: The names of the processed files
    - processed_dir: The directory of the processed files
    """

    @property
    def raw_file_names(self) -> List[str]:
        return [f'{self.dataset_type}_{self.split}_labeled.json']

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.dataset_type}_{self.split}_schema_linking.pt']

    @property
    def processed_dir(self) -> str:
        return self.graph_data_dir


    """
    Process the raw data into a list of HeteroData objects
    """
    def process(self):
        # Load labeled dataset
        path_config = self._load_json('config/path_config.json')
        labeled_path = os.path.join(path_config['labeled_dataset_paths']['labeled_dataset_base'], self.raw_file_names[0])
        labeled_data = self._load_json(labeled_path)
    
        # Process each example and create graphs
        data_list = []
        for example in labeled_data:
            graph = self._create_graph(example)
            if graph is not None:
                data_list.append(graph)
        
        # Save processed data
        processed_path = os.path.join(self.graph_data_dir, self.processed_file_names[0])
        os.makedirs(self.graph_data_dir, exist_ok=True)
        torch.save(data_list, processed_path)


    """
    Create a HeteroData object for a given question and schema
    :param example: A dictionary representing a labeled example
    :return: A HeteroData object
    """
    def _create_graph(self, example: Dict) -> HeteroData:
        data = HeteroData()
        
        # Create table nodes
        table_names = [t['name'].lower() for t in example['tables']]
        data['table'].x = torch.eye(len(table_names))
        data['table'].y = torch.tensor([t['relevant'] for t in example['tables']])
        
        # Create column nodes
        columns = []
        column_labels = []
        for table in example['tables']:
            table_name = table['name'].lower()
            for col in table['columns']:
                columns.append((table_name, col['name'].lower()))
                column_labels.append(col['relevant'])
        
        data['column'].x = torch.eye(len(columns))
        data['column'].y = torch.tensor(column_labels)
        
        # Create contains edges (table -> column)
        edge_index_contains = []
        for i, (table_name, _) in enumerate(columns):
            table_idx = table_names.index(table_name)
            edge_index_contains.append([table_idx, i])
        
        if edge_index_contains:
            data['table', 'contains', 'column'].edge_index = torch.tensor(
                edge_index_contains).t().contiguous()
        
        # Create foreign key edges
        if 'foreign_keys' in example:
            edge_index_fk = []
            seen_pairs = set()
            
            for fk in example['foreign_keys']:
                if fk['column'] is None or any(c is None for c in fk['column']):
                    logger.warning(f"[* Warning] Skipping invalid foreign key: {fk}")
                    continue
                
                table1, table2 = [t.lower() for t in fk['table']]
                col1, col2 = [c.lower() for c in fk['column']]
                
                try:
                    # Find indices of the two columns involved in the foreign key
                    col1_idx = columns.index((table1, col1))
                    col2_idx = columns.index((table2, col2))
                    
                    # Sort indices to ensure consistent ordering
                    sorted_pair = tuple(sorted([col1_idx, col2_idx]))
                    if sorted_pair not in seen_pairs:
                        seen_pairs.add(sorted_pair)
                        # Add bidirectional edges between the columns
                        # edge_index_fk.extend([[col1_idx, col2_idx], [col2_idx, col1_idx]])
                        edge_index_fk.extend([[col1_idx, col2_idx]])
                except ValueError:
                    continue
            
            if edge_index_fk:
                # Create the bidirectional edges between columns
                data['column', 'foreign_key', 'column'].edge_index = torch.tensor(
                    edge_index_fk).t().contiguous()
        
        return data


    """
    Load a JSON file
    :param path: The path to the JSON file
    :return: A dictionary representing the JSON data
    """
    def _load_json(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)


    """
    Get the number of graphs in the dataset
    :return: The number of graphs in the dataset
    """
    def len(self) -> int:
        processed_path = os.path.join(self.graph_data_dir, self.processed_file_names[0])
        data_list = torch.load(processed_path, weights_only=False)
        return len(data_list)


    """
    Get a graph from the dataset by index
    :param idx: The index of the graph to get
    :return: A HeteroData object
    """
    def get(self, idx: int) -> HeteroData:
        processed_path = os.path.join(self.graph_data_dir, self.processed_file_names[0])
        data_list = torch.load(processed_path, weights_only=False)
        return data_list[idx]



if __name__ == "__main__":
    # Create datasets (labeled heterogenous graph data)
    spider_train = SchemaLinkingHeteroGraphDataset(root='data/', dataset_type='spider', split='train')
    spider_dev = SchemaLinkingHeteroGraphDataset(root='data/', dataset_type='spider', split='dev')
    bird_train = SchemaLinkingHeteroGraphDataset(root='data/', dataset_type='bird', split='train')
    bird_dev = SchemaLinkingHeteroGraphDataset(root='data/', dataset_type='bird', split='dev')