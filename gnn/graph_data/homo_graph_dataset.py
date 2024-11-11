import json
import torch
import os
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

    def process(self):
        path_config = self._load_json('config/path_config.json')
        labeled_path = os.path.join(path_config['labeled_dataset_paths']['labeled_dataset_base'], self.raw_file_names[0])
        labeled_data = self._load_json(labeled_path)
    
        data_list = []
        for example in labeled_data:
            graph = self._create_graph(example)
            if graph is not None:
                data_list.append(graph)
        
        processed_path = os.path.join(self.graph_data_dir, self.processed_file_names[0])
        os.makedirs(self.graph_data_dir, exist_ok=True)
        torch.save(data_list, processed_path)

    def _create_graph(self, example: Dict) -> Data:
        # Create nodes for tables and columns
        nodes = []  # List of (name, is_table) tuples
        node_labels = []  # Relevance labels for nodes
        
        # Add table nodes
        for table in example['tables']:
            nodes.append((table['name'].lower(), True))  # True indicates it's a table
            node_labels.append(table['relevant'])
            
            # Add column nodes for this table
            for col in table['columns']:
                nodes.append((col['name'].lower(), False))  # False indicates it's a column
                node_labels.append(col['relevant'])

        # Create node features (one-hot encoding)
        x = torch.eye(len(nodes))
        y = torch.tensor(node_labels)

        # Create edges
        edge_index = []
        
        # Add contains edges (table -> column)
        for table_idx, (table_name, is_table) in enumerate(nodes):
            if is_table:  # if it's a table node
                table_columns = next(t['columns'] for t in example['tables'] 
                                  if t['name'].lower() == table_name)
                
                # Find corresponding column indices
                for col in table_columns:
                    try:
                        col_idx = nodes.index((col['name'].lower(), False))
                        edge_index.append([table_idx, col_idx])
                        edge_index.append([col_idx, table_idx])  # Make it bidirectional
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
                    col1_name, col2_name = [c.lower() for c in fk['column']]
                    col1_idx = nodes.index((col1_name, False))
                    col2_idx = nodes.index((col2_name, False))
                    
                    # Add bidirectional edges
                    edge_index.append([col1_idx, col2_idx])
                    edge_index.append([col2_idx, col1_idx])
                except ValueError:
                    continue

        if edge_index:
            edge_index = torch.tensor(edge_index).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, y=y)

    def _load_json(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

    def len(self) -> int:
        processed_path = os.path.join(self.graph_data_dir, self.processed_file_names[0])
        data_list = torch.load(processed_path, weights_only=False)
        return len(data_list)

    def get(self, idx: int) -> Data:
        processed_path = os.path.join(self.graph_data_dir, self.processed_file_names[0])
        data_list = torch.load(processed_path, weights_only=False)
        return data_list[idx]


if __name__ == "__main__":
    # Create datasets (labeled homogeneous graph data)
    spider_train = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='spider', split='train')
    spider_dev = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='spider', split='dev')
    bird_train = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='bird', split='train')
    bird_dev = SchemaLinkingHomoGraphDataset(root='data/', dataset_type='bird', split='dev')