import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from typing import Dict, List

class SchemaLinkingGNN(torch.nn.Module):
    def __init__(self, hidden_channels: int, num_layers: int):
        super().__init__()
        
        # Initial node embeddings (use fixed input dimensions)
        self.table_encoder = Linear(1, hidden_channels)  # Changed from -1 to 1
        self.column_encoder = Linear(1, hidden_channels)  # Changed from -1 to 1
        
        # Stack of heterogeneous graph convolution layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('table', 'contains', 'column'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
                ('column', 'foreign_key', 'column'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
                ('column', 'rev_contains', 'table'): SAGEConv((hidden_channels, hidden_channels), hidden_channels),
            }, aggr='mean')
            self.convs.append(conv)
        
        # Output layers for node classification
        self.table_classifier = Linear(hidden_channels, 1)
        self.column_classifier = Linear(hidden_channels, 1)

    def forward(self, x_dict, edge_index_dict):
        # Initial node embeddings (reshape input to have feature dim 1)
        x_dict = {
            'table': self.table_encoder(x_dict['table'].unsqueeze(-1).float()),
            'column': self.column_encoder(x_dict['column'].unsqueeze(-1).float())
        }
        
        # Message passing layers
        for conv in self.convs:
            x_dict_new = conv(x_dict, edge_index_dict)
            # Apply ReLU activation and residual connection
            x_dict = {
                key: F.relu(x + x_new)
                for (key, x), x_new in zip(x_dict.items(), x_dict_new.values())
            }
        
        # Node classification
        table_out = torch.sigmoid(self.table_classifier(x_dict['table']))
        column_out = torch.sigmoid(self.column_classifier(x_dict['column']))
        
        return table_out.squeeze(), column_out.squeeze()

