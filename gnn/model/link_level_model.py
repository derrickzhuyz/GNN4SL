import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import json
import os
from typing import Dict, List, Tuple
from loguru import logger
import sys

# Remove default logger
logger.remove()
# Add file handler with INFO level
logger.add("logs/link_level_model.log", 
           rotation="50 MB", 
           level="INFO",
           format="{time} {level} {message}",
           compression="zip")
# Add console handler with WARNING level
logger.add(sys.stderr, level="WARNING")


class LinkLevelGNN(nn.Module):
    def __init__(self, in_channels: int = 384, hidden_channels: int = 128, num_layers: int = 2, dropout: float = 0.1):
        """
        Link prediction model using Graph Convolutional Networks
        
        Args:
            in_channels: Input feature dimension (such as 384 for sentence transformer embeddings)
            hidden_channels: Hidden layer dimension
            num_layers: Number of GCN layers
            dropout: Dropout rate
        """
        super().__init__()
        self.num_layers = num_layers
        
        # First GCN layer
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        # Middle GCN layers
        for _ in range(max(num_layers - 2, 0)):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Last GCN layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        # Link prediction layers
        self.link_predictor = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1)
        )


    """
    Forward pass through the model
    :param x: node features
    :param edge_index: edge indices
    :return: node embeddings
    """
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        # Graph convolution layers
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        
        return x


    """
    Predict relevance between question nodes and schema nodes (tables and columns)
    :param data: graph data
    :return: predictions and scores
    """
    def predict_links(self, data: Data, threshold: float = 0.5) -> Tuple[List[Dict], torch.Tensor]:
        self.eval()
        with torch.no_grad():
            # Get node embeddings through GCN layers
            node_embeddings = self.forward(data.x, data.edge_index)
            scores_list = []
            predictions = []

            # Process the single graph
            question_indices = [i for i, type_ in enumerate(data.node_types[0]) if type_ == 'question']
            logger.info(f"Found {len(question_indices)} question nodes")

            # Create a dictionary to group columns by their parent tables
            tables = {}
            column_to_table = {}

            # Identify all tables
            for i, type_ in enumerate(data.node_types[0]):
                if type_ == 'table':
                    table_name = data.node_names[0][i] if data.node_names else f'table_{i}'
                    logger.info(f"Found table: {table_name} at index {i}")
                    tables[i] = {
                        'name': table_name,
                        'columns': []
                    }

            # Find column-table relationships from edge_index
            edge_index_np = data.edge_index.cpu().numpy()

            for i in range(data.edge_index.shape[1]):
                src, dst = edge_index_np[:, i]
                if src < len(data.node_types[0]) and dst < len(data.node_types[0]):
                    src_type = data.node_types[0][src]
                    dst_type = data.node_types[0][dst]

                    # If this edge connects a column and a table
                    if (src_type == 'column' and dst_type == 'table'):
                        column_to_table[src] = dst
                    elif (dst_type == 'column' and src_type == 'table'):
                        column_to_table[dst] = src

            logger.info(f"Found {len(column_to_table)} column-table relationships")

            # Add columns to their respective tables
            for col_idx, table_idx in column_to_table.items():
                if table_idx in tables:
                    col_name = data.node_names[0][col_idx] if data.node_names else f'column_{col_idx}'
                    tables[table_idx]['columns'].append({
                        'idx': col_idx,
                        'name': col_name
                    })

            # Process each question
            for q_idx in question_indices:
                q_embedding = node_embeddings[q_idx]
                question_name = data.node_names[0][q_idx] if data.node_names else f'question_{q_idx}'

                question_pred = {
                    'question_idx': q_idx,
                    'question': question_name,
                    'tables': []
                }

                # Process each table and its columns
                for table_idx, table_info in tables.items():
                    table_embedding = node_embeddings[table_idx]
                    pair_embedding = torch.cat([q_embedding, table_embedding])
                    table_score = torch.sigmoid(self.link_predictor(pair_embedding))
                    scores_list.append(table_score)

                    table_pred = {
                        'name': table_info['name'],
                        'relevant': bool(table_score > threshold),
                        'score': float(table_score),
                        'columns': []
                    }

                    # Process all columns for this table
                    for col in table_info['columns']:
                        col_embedding = node_embeddings[col['idx']]
                        pair_embedding = torch.cat([q_embedding, col_embedding])
                        col_score = torch.sigmoid(self.link_predictor(pair_embedding))
                        scores_list.append(col_score)

                        col_pred = {
                            'name': col['name'],
                            'relevant': bool(col_score > threshold),
                            'score': float(col_score)
                        }
                        table_pred['columns'].append(col_pred)

                    question_pred['tables'].append(table_pred)

                predictions.append(question_pred)

            # Handle empty predictions case
            if not scores_list:
                logger.warning("No scores generated for any question, returning empty predictions.")
                return predictions, torch.tensor([], device=data.x.device)

            # Convert scores to tensor
            scores = torch.stack(scores_list)
            return predictions, scores
    

    """
    Save predictions to JSON file
    :param predictions: predictions
    :param output_dir: output directory
    :param split: split name
    :param dataset_type: dataset type
    :param model_name: model name
    """
    def save_predictions(self, predictions: List[Dict], output_dir: str, split: str, dataset_type: str, model_name: str):
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{dataset_type}_{split}_predictions_{model_name}.json')
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        logger.info(f"[✓] Saved predictions to {output_path}")
