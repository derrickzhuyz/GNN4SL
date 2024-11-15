import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, Linear
from typing import Dict, List

class SchemaLinkingHomoGNN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_channels: int, num_layers: int):
        super().__init__()
        
        # Initial node encoder
        self.node_encoder = Linear(input_dim, hidden_channels)
        
        # Stack of graph convolution layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = SAGEConv(hidden_channels, hidden_channels)
            self.convs.append(conv)
        
        # Output layer for node classification
        self.classifier = Linear(hidden_channels, 1)

    def forward(self, data):
        # Get node features and edge indices
        x, edge_index = data.x, data.edge_index
        
        # Initial node embeddings
        x = self.node_encoder(x.float())
        
        # Message passing layers
        for conv in self.convs:
            x_new = conv(x, edge_index)
            # Apply ReLU activation and residual connection
            x = F.relu(x + x_new)
        
        # Node classification
        out = torch.sigmoid(self.classifier(x))
        
        return out.squeeze()

    def loss_function(self, pred, target):
        # Binary cross entropy loss for node classification
        return F.binary_cross_entropy(pred, target.float())


