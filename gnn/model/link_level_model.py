import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
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


class BaseLinkLevelGNN(nn.Module, ABC):
    def __init__(self, 
                 in_channels: int = 384, 
                 hidden_channels: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 threshold: float = 0.5,
                 prediction_method: str = 'concat_mlp'):
        """
        The base class for link prediction GNN models
        
        Args:
            in_channels: Input feature dimension (e.g. 384 for Sentence Transformer, 1536 for text-embedding-3-small etc.)
            hidden_channels: Hidden layer dimension
            num_layers: Number of GNN layers
            dropout: Dropout rate
            threshold: Threshold for binary prediction (score > threshold -> relevant, score <= threshold -> irrelevant)
            prediction_method: Either 'dot_product' or 'concat_mlp'
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.threshold = threshold
        self.prediction_method = prediction_method

        if prediction_method == 'concat_mlp':
            self.link_predictor = nn.Sequential(
                nn.Linear(hidden_channels * 2, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, 1)
            )
        elif prediction_method == 'dot_product':
            pass
        else:
            raise ValueError("prediction_method must be either 'dot_product' or 'concat_mlp'")
        


    """
    Forward pass through the model (abstract method)
    :param x: Node feature matrix (embeddings)
    :param edge_index: Graph connectivity in COO format
    :return: Node embeddings after passing through GNN layers
    """
    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        pass

    
    """ 
    NOTE: Deprecated Function!
    Predict link between two node embeddings. Two options:
        - 'concat_mlp': Concatenate embeddings and pass through MLP
        - 'dot_product': Normalize embeddings and compute dot product
    :param src_embedding: Source node embedding
    :param dst_embedding: Destination node embedding
    :return: Prediction score
    """
    def _deprecated_predict_pair(self, src_embedding: torch.Tensor, dst_embedding: torch.Tensor) -> torch.Tensor:
        if self.prediction_method == 'concat_mlp':
            pair_embedding = torch.cat([src_embedding, dst_embedding])
            return torch.sigmoid(self.link_predictor(pair_embedding))
        elif self.prediction_method == 'dot_product':
            # Normalize embeddings (optional but recommended for dot product)
            src_norm = F.normalize(src_embedding, p=2, dim=-1)
            dst_norm = F.normalize(dst_embedding, p=2, dim=-1)
            # Scale dot product and apply sigmoid
            return torch.sigmoid(self.scale * torch.sum(src_norm * dst_norm))
        else:
            raise ValueError("Prediction method not supported! Must be either 'dot_product' or 'concat_mlp'")


    """
    Predict relevance between question nodes and schema nodes (tables and columns) and return predictions and scores
    :param data: PyG Data object containing:
            - x: Node feature matrix (embeddings)
            - edge_index: Graph connectivity in COO format
            - node_types: List of node types ('question', 'table', 'column')
            - node_names: List of node names (question text, table names, column names)
    :return:
        predictions: List of dictionaries containing predictions for each question
        scores: Tensor of all prediction scores
    """
    def predict_links(self, data: Data) -> Tuple[List[Dict], torch.Tensor]:
        """Predict relevance between question nodes and schema nodes"""
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            # Get node embeddings through GCN layers
            node_embeddings = self.forward(data.x, data.edge_index)
            
            # Normalize embeddings if using dot product
            if self.prediction_method == 'dot_product':
                node_embeddings = F.normalize(node_embeddings, p=2, dim=-1)
            
            scores_list = []
            predictions = []

            # Find all question nodes in the graph
            question_indices = [i for i, type_ in enumerate(data.node_types[0]) if type_ == 'question']
            logger.info(f"Found {len(question_indices)} question nodes")

            # Create dictionaries to store table and column information
            tables = {}  # Maps table index to table info
            column_to_table = {}  # Maps column index to its parent table index

            # Step 1: Identify all tables in the graph
            for i, type_ in enumerate(data.node_types[0]):
                if type_ == 'table':
                    table_name = data.node_names[0][i] if data.node_names else f'table_{i}'
                    logger.info(f"Found table: {table_name} at index {i}")
                    tables[i] = {
                        'name': table_name,
                        'columns': []  # Will store column information
                    }

            # Step 2: Find column-table relationships from edge_index
            edge_index_np = data.edge_index.cpu().numpy()
            for i in range(data.edge_index.shape[1]):
                src, dst = edge_index_np[:, i]
                if src < len(data.node_types[0]) and dst < len(data.node_types[0]):
                    src_type = data.node_types[0][src]
                    dst_type = data.node_types[0][dst]

                    # Store column-table relationships based on edges
                    if (src_type == 'column' and dst_type == 'table'):
                        column_to_table[src] = dst
                    elif (dst_type == 'column' and src_type == 'table'):
                        column_to_table[dst] = src

            logger.info(f"Found {len(column_to_table)} column-table relationships")

            # Step 3: Add columns to their respective tables
            for col_idx, table_idx in column_to_table.items():
                if table_idx in tables:
                    col_name = data.node_names[0][col_idx] if data.node_names else f'column_{col_idx}'
                    tables[table_idx]['columns'].append({
                        'idx': col_idx,
                        'name': col_name
                    })

            # Step 4: Process each question node
            for q_idx in question_indices:
                q_embedding = node_embeddings[q_idx]  # Get question embedding
                question_name = data.node_names[0][q_idx] if data.node_names else f'question_{q_idx}'

                # Initialize prediction structure for this question
                question_pred = {
                    'question_idx': q_idx,
                    'question': question_name,
                    'tables': []
                }

                # Step 5: Process each table and its columns
                for table_idx, table_info in tables.items():
                    # Predict table relevance
                    table_embedding = node_embeddings[table_idx]
                    # table_score = self.predict_pair(q_embedding, table_embedding)
                    if self.prediction_method == 'concat_mlp':
                        pair_embedding = torch.cat([q_embedding, table_embedding])
                        table_score = torch.sigmoid(self.link_predictor(pair_embedding))
                    else:  # dot_product
                        table_score = torch.sigmoid((q_embedding * table_embedding).sum(dim=-1))
                    
                    # Append table score and create table prediction entry
                    scores_list.append(table_score)
                    table_pred = {
                        'name': table_info['name'],
                        'relevant': bool(table_score > self.threshold),
                        'score': float(table_score),
                        'columns': []
                    }
                    # Process all columns belonging to this table
                    for col in table_info['columns']:
                        # Predict column relevance
                        col_embedding = node_embeddings[col['idx']]
                        # col_score = self.predict_pair(q_embedding, col_embedding)
                        if self.prediction_method == 'concat_mlp':
                            pair_embedding = torch.cat([q_embedding, col_embedding])
                            col_score = torch.sigmoid(self.link_predictor(pair_embedding))
                        else:  # dot_product
                            col_score = torch.sigmoid((q_embedding * col_embedding).sum(dim=-1))
                        
                        # Append column score and create column prediction entry
                        scores_list.append(col_score)
                        col_pred = {
                            'name': col['name'],
                            'relevant': bool(col_score > self.threshold),
                            'score': float(col_score)
                        }
                        table_pred['columns'].append(col_pred)

                    question_pred['tables'].append(table_pred)

                predictions.append(question_pred)

            # Handle empty predictions case
            if not scores_list:
                logger.warning("No scores generated for any question, returning empty predictions.")
                return predictions, torch.tensor([], device=data.x.device)

            # Convert all scores to a single tensor
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
        output_path = os.path.join(output_dir, f'test_{dataset_type}_{split}_By_{model_name}.json')
        
        with open(output_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        
        logger.info(f"[✓] Saved predictions to {output_path}")


    """
    Print the model structure including configuration parameters and layer information
    """
    def print_model_structure(self):
        print("=" * 50)
        print(f"Model Structure of Link-level GNN: {self.__class__.__name__}")
        print("-" * 50)
        print("Configuration:")
        print(f"- Input Channels: {self.in_channels}")
        print(f"- Hidden Channels: {self.hidden_channels}")
        print(f"- Number of Layers: {self.num_layers}")
        print(f"- Dropout Rate: {self.dropout}")
        print(f"- Prediction Threshold: {self.threshold}")
        print(f"- Prediction Method: {self.prediction_method}")
        
        print("Model Layers:")
        for name, module in self.named_children():
            if isinstance(module, nn.ModuleList):
                print(f"- {name}:")
                for idx, layer in enumerate(module):
                    print(f"  * Layer {idx}: {layer}")
            else:
                print(f"{name}: {module}")
        print("=" * 50)



"""
Link-level GNN model based on Graph Convolutional Networks
Inherits from the base class of BaseLinkLevelGNN
"""
class LinkLevelGCN(BaseLinkLevelGNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # GCN layers
        from torch_geometric.nn import GCNConv
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.in_channels, self.hidden_channels))
        
        # Middle GCN layers
        for _ in range(max(self.num_layers - 2, 0)):
            self.convs.append(GCNConv(self.hidden_channels, self.hidden_channels))
        
        # Last GCN layer
        self.convs.append(GCNConv(self.hidden_channels, self.hidden_channels))


    """
    Forward pass through GCN layers
    :param x: Node feature matrix (embeddings)
    :param edge_index: Graph connectivity in COO format
    :return: Node embeddings after passing through GCN layers
    """
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    


"""
Link-level GNN model based on Graph Attention Networks
Inherits from the base class of BaseLinkLevelGNN
"""
class LinkLevelGAT(BaseLinkLevelGNN):
    def __init__(self, 
                 in_channels: int = 384, 
                 hidden_channels: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 threshold: float = 0.5,
                 prediction_method: str = 'dot_product',
                 num_heads: int = 4):
        """
        Link prediction model using Graph Attention Networks
        
        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            num_layers: Number of GAT layers
            dropout: Dropout rate
            threshold: Threshold for binary prediction
            prediction_method: Either 'dot_product' or 'concat_mlp'
            num_heads: Number of attention heads
        """
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            threshold=threshold,
            prediction_method=prediction_method
        )
        
        # Import GATConv here to avoid potential circular imports
        from torch_geometric.nn import GATConv
        self.convs = nn.ModuleList()
        
        # First GAT layer
        self.convs.append(
            GATConv(self.in_channels, 
                    self.hidden_channels // num_heads, 
                    heads=num_heads, 
                    dropout=dropout)
        )
        
        # Middle GAT layers
        for _ in range(max(self.num_layers - 2, 0)):
            self.convs.append(
                GATConv(self.hidden_channels, 
                        self.hidden_channels // num_heads, 
                        heads=num_heads, 
                        dropout=dropout)
            )
        
        # Last GAT layer (combine heads)
        self.convs.append(
            GATConv(self.hidden_channels, 
                    self.hidden_channels, 
                    heads=1, 
                    dropout=dropout)
        )


    """
    Forward pass through GAT layers
    :param x: Node feature matrix (embeddings)
    :param edge_index: Graph connectivity in COO format
    :return: Node embeddings after passing through GAT layers
    """
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x



# TODO: New GNN model to inherit from BaseLinkLevelGNN ......