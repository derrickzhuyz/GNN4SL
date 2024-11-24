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
        Predict relevance between question nodes and schema nodes
        
        Args:
            data: PyG Data object containing the graph
            
        Returns:
            predictions: List of dictionaries containing predictions for each question
            scores: Tensor of prediction scores
        """
        self.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient calculation for inference
            # Get node embeddings
            node_embeddings = self.forward(data.x, data.edge_index)
            
            predictions = []  # Initialize list to store predictions for each question
            scores_list = []  # Initialize list to store prediction scores
            
            # Find question nodes
            question_indices = [i for i, type_ in enumerate(data.node_types) if type_ == 'question']
            
            # For each question node
            for q_idx in question_indices:
                question_name = data.node_names[q_idx]  # Get the name of the question node
                q_embedding = node_embeddings[q_idx]  # Get the embedding for the question node
                
                question_pred = {
                    'question': question_name,
                    'database': data.database_name,
                    'tables': [],  # Initialize list to store table predictions
                }
                
                # Group schema nodes by table
                table_columns = {}  # Dictionary to hold table names and their corresponding columns
                for i, (name, type_) in enumerate(zip(data.node_names, data.node_types)):
                    if type_ == 'table':
                        table_columns[name] = {'idx': i, 'columns': []}  # Initialize entry for table
                    elif type_ == 'column':
                        table_name = None
                        # Find the connected table node
                        for edge_idx in range(data.edge_index.size(1)):
                            src, dst = data.edge_index[:, edge_idx]
                            # Check if the current column is connected to a table
                            if (src == i and data.node_types[dst] == 'table') or \
                               (dst == i and data.node_types[src] == 'table'):
                                connected_idx = dst if src == i else src
                                table_name = data.node_names[connected_idx]  # Get the name of the connected table
                                break
                        # If a table is found, add the column to the corresponding table entry
                        if table_name and table_name in table_columns:
                            table_columns[table_name]['columns'].append({'name': name, 'idx': i})
                
                # Predict relevance for each table and its columns
                for table_name, table_info in table_columns.items():
                    # Predict table relevance
                    table_embedding = node_embeddings[table_info['idx']]  # Get the embedding for the table
                    pair_embedding = torch.cat([q_embedding, table_embedding])  # Concatenate question and table embeddings
                    table_score = torch.sigmoid(self.link_predictor(pair_embedding))  # Get the relevance score for the table
                    scores_list.append(table_score)  # Store the score
                    
                    table_pred = {
                        'name': table_name,
                        'relevant': bool(table_score > 0.5),  # Determine if the table is relevant based on the score
                        'score': float(table_score),  # Convert score to float for easier handling
                        'columns': []  # Initialize list to store column predictions
                    }
                    
                    # Predict column relevance
                    for col in table_info['columns']:
                        col_embedding = node_embeddings[col['idx']]  # Get the embedding for the column
                        pair_embedding = torch.cat([q_embedding, col_embedding])  # Concatenate question and column embeddings
                        col_score = torch.sigmoid(self.link_predictor(pair_embedding))  # Get the relevance score for the column
                        scores_list.append(col_score)  # Store the score
                        
                        # Add column prediction to the table prediction
                        table_pred['columns'].append({
                            'name': col['name'],
                            'relevant': bool(col_score > 0.5),  # Determine if the column is relevant based on the score
                            'score': float(col_score)  # Convert score to float for easier handling
                        })
                    
                    question_pred['tables'].append(table_pred)  # Add the table prediction to the question prediction
                
                predictions.append(question_pred)  # Add the question prediction to the overall predictions
            
            scores = torch.stack(scores_list)  # Stack all scores into a tensor
            
            return predictions, scores  # Return the predictions and scores

    def save_predictions(self, predictions: List[Dict], output_dir: str, split: str, dataset_type: str):
        """Save predictions to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{dataset_type}_{split}_predictions.json')
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        logger.info(f"[✓] Saved predictions to {output_path}")
