import json
import torch
from torch_geometric.data import HeteroData, Dataset
from typing import Dict, List, Set, Tuple
import os

class SchemaLinkingDataset(Dataset):
    def __init__(self, root, dataset_type='spider', split='train', transform=None):
        """
        Args:
            root: Root directory where the dataset should be saved
            dataset_type: One of ['spider', 'bird'] to specify which dataset to use
            split: One of ['train', 'dev'] to specify train or dev split
            transform: Optional transform to be applied on a sample
        """
        self.dataset_type = dataset_type
        self.split = split
        super().__init__(root, transform)
    
    @property
    def raw_file_names(self) -> List[str]:
        schema_file = 'spider_schemas.json' if self.dataset_type == 'spider' else f'bird_{self.split}_schemas.json'
        linking_file = f'{self.dataset_type}_{self.split}_gold_schema_linking.json'
        return [schema_file, linking_file]

    @property
    def processed_file_names(self) -> List[str]:
        return [f'{self.dataset_type}_{self.split}_schema_linking.pt']

    def process(self):
        # Load schema file
        schema_path = os.path.join(self.root, 'db_schema', self.raw_file_names[0])
        schemas = self._load_json(schema_path)
        schema_dict = {schema['database']: schema for schema in schemas}
        
        # Load question file
        linking_path = os.path.join(self.root, 'gold_schema_linking', self.raw_file_names[1])
        questions = self._load_json(linking_path)
        
        # Process each question and create graphs
        data_list = []
        for q in questions:
            graph = self._create_graph(q, schema_dict[q['database']])
            if graph is not None:
                data_list.append(graph)
        
        # Save processed data
        processed_path = os.path.join(self.processed_dir, self.processed_file_names[0])
        os.makedirs(self.processed_dir, exist_ok=True)
        torch.save(data_list, processed_path)

    def _create_graph(self, question: Dict, schema: Dict) -> HeteroData:
        data = HeteroData()
        
        # Get relevant tables and columns from question
        relevant_tables = set()
        relevant_columns = set()
        for table in question['tables']:
            relevant_tables.add(table['table'].lower())
            for col in table['columns']:
                relevant_columns.add((table['table'].lower(), col.lower()))
        
        # Create table nodes (all tables in schema)
        table_names = [t['table'].lower() for t in schema['tables']]
        data['table'].x = torch.eye(len(table_names))  # One-hot encoding
        data['table'].y = torch.tensor([1 if t in relevant_tables else 0 
                                      for t in table_names])
        
        # Create column nodes (all columns in schema)
        columns = []
        column_labels = []
        for table in schema['tables']:
            table_name = table['table'].lower()
            for col in table['columns']:
                col_name = col.lower()
                columns.append((table_name, col_name))
                column_labels.append(1 if (table_name, col_name) in relevant_columns else 0)
        
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
        if 'foreign_keys' in schema:
            edge_index_fk = []
            seen_pairs = set()  # To track unique pairs
            for fk in schema['foreign_keys']:
                if fk['column'] is None or any(c is None for c in fk['column']):
                    print(f"Warning: Skipping invalid foreign key: {fk}")
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

    def _load_json(self, path: str) -> Dict:
        with open(path, 'r') as f:
            return json.load(f)

    def len(self) -> int:
        processed_path = os.path.join(self.processed_dir, self.processed_file_names[0])
        return len(torch.load(processed_path))

    def get(self, idx: int) -> HeteroData:
        processed_path = os.path.join(self.processed_dir, self.processed_file_names[0])
        data_list = torch.load(processed_path)
        return data_list[idx] 