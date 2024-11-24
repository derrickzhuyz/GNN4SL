import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import json
import os
from typing import Dict, List, Tuple
from loguru import logger

logger.add("logs/link_level_model.log", rotation="1 MB", level="INFO",
           format="{time} {level} {message}", compression="zip")

class LinkLevelModel(nn.Module):
    def __init__(self, in_channels: int = 384, hidden_channels: int = 256, num_heads: int = 4, num_layers: int = 3, dropout: float = 0.1):
        """
        Link prediction model using Graph Attention Networks
        
        Args:
            in_channels: Input feature dimension (384 for your embeddings)
            hidden_channels: Hidden layer dimension
            num_heads: Number of attention heads
            num_layers: Number of GAT layers
            dropout: Dropout rate
        """
        super().__init__()
        self.num_layers = num_layers
        
        # First GAT layer (from in_channels to hidden_channels)
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(in_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
        )
        
        # Middle GAT layers
        for _ in range(num_layers - 2):
            self.convs.append(
                GATConv(hidden_channels, hidden_channels // num_heads, heads=num_heads, dropout=dropout)
            )
        
        # Last GAT layer (combine heads)
        self.convs.append(
            GATConv(hidden_channels, hidden_channels, heads=1, dropout=dropout)
        )
        
        # Link prediction layers
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model"""
        # Graph convolution layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        
        return x

    def predict_links(self, data: Data) -> Tuple[List[Dict], torch.Tensor]:
        """
        Predict relevance between question nodes and schema nodes (tables and columns)
        
        Args:
            data: PyG Data object containing:
                - node_types: List of node types ('question', 'table', 'column')
                - node_names: Optional list of node names
                - edge_index: Tensor of shape [2, num_edges] containing edge connections
                - x: Node feature matrix
        
        Returns:
            predictions: List of dictionaries with structure:
                {
                    'question_idx': int,  # Index of the question node
                    'question': str,      # Question text or ID
                    'tables': [           # List of relevant tables
                        {
                            'name': str,      # Table name
                            'relevant': bool,  # Whether table is relevant
                            'score': float,    # Prediction score
                            'columns': [       # List of columns in this table
                                {
                                    'name': str,      # Column name
                                    'relevant': bool,  # Whether column is relevant
                                    'score': float,    # Prediction score
                                },
                                ...
                            ]
                        },
                        ...
                    ]
                }
            scores: Tensor containing all prediction scores
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for inference
            # Get node embeddings through GAT layers
            node_embeddings = self.forward(data.x, data.edge_index)
            scores_list = []  # Store all prediction scores
            predictions = []  # Store structured predictions
            
            # Find all question nodes in the graph
            question_indices = [i for i, type_ in enumerate(data.node_types[0]) if type_ == 'question']
            logger.info(f"Found {len(question_indices)} question nodes")
            
            # Create a dictionary to group columns by their parent tables
            tables = {}  # Format: {table_idx: {'name': str, 'columns': List[Dict]}}
            
            # First pass: identify all tables
            for i, type_ in enumerate(data.node_types[0]):
                if type_ == 'table':
                    # Store table info with a default name if node_names not provided
                    tables[i] = {
                        'name': data.node_names[0][i] if hasattr(data, 'node_names') else f'table_{i}',
                        'columns': []
                    }
            
            # Second pass: associate columns with their tables using edge information
            for i, type_ in enumerate(data.node_types[0]):
                if type_ == 'column':
                    # Find all edges connected to this column
                    edge_mask = (data.edge_index[0] == i) | (data.edge_index[1] == i)
                    connected_edges = data.edge_index[:, edge_mask]
                    
                    # Look for a table connection in the edges
                    for edge_idx in range(connected_edges.shape[1]):
                        src, dst = connected_edges[:, edge_idx]
                        other_node = dst if src == i else src
                        # If the connected node is a table, associate column with it
                        if data.node_types[0][other_node] == 'table':
                            if other_node in tables:
                                tables[other_node]['columns'].append({
                                    'idx': i,  # Store column index for later embedding lookup
                                    'name': data.node_names[0][i] if hasattr(data, 'node_names') else f'column_{i}'
                                })
                            break
            
            # Process each question to generate predictions
            for q_idx in question_indices:
                q_embedding = node_embeddings[q_idx]  # Get question node embedding
                
                # Create prediction structure for this question
                question_pred = {
                    'question_idx': q_idx,
                    'question': data.node_names[0][q_idx] if hasattr(data, 'node_names') else f'question_{q_idx}',
                    'tables': []
                }
                
                # Process each table and its columns
                for table_idx, table_info in tables.items():
                    # Predict table relevance
                    table_embedding = node_embeddings[table_idx]
                    pair_embedding = torch.cat([q_embedding, table_embedding])
                    table_score = torch.sigmoid(self.link_predictor(pair_embedding))
                    scores_list.append(table_score)
                    
                    # Create prediction structure for this table
                    table_pred = {
                        'name': table_info['name'],
                        'relevant': bool(table_score > 0.5),  # Binary relevance threshold
                        'score': float(table_score),
                        'columns': []
                    }
                    
                    # Process each column in the table
                    for col in table_info['columns']:
                        # Predict column relevance
                        col_embedding = node_embeddings[col['idx']]
                        pair_embedding = torch.cat([q_embedding, col_embedding])
                        col_score = torch.sigmoid(self.link_predictor(pair_embedding))
                        scores_list.append(col_score)
                        
                        # Add column prediction
                        table_pred['columns'].append({
                            'name': col['name'],
                            'relevant': bool(col_score > 0.5),  # Binary relevance threshold
                            'score': float(col_score)
                        })
                    
                    question_pred['tables'].append(table_pred)
                
                predictions.append(question_pred)
            
            # Handle case where no predictions were generated
            if not scores_list:
                logger.warning("No scores generated for any question, returning empty predictions.")
                return predictions, torch.tensor([], device=data.x.device)
            
            # Convert list of scores to tensor
            try:
                scores = torch.stack(scores_list)
            except RuntimeError as e:
                logger.error(f"Error stacking scores: {e}")
                logger.error(f"Number of scores: {len(scores_list)}")
                return predictions, torch.tensor([], device=data.x.device)
            
            return predictions, scores

    def save_predictions(self, predictions: List[Dict], output_dir: str, split: str, dataset_type: str):
        """Save predictions to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{dataset_type}_{split}_predictions.json')
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        logger.info(f"[✓] Saved predictions to {output_path}")
