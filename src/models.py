"""
This module contains the graph models for our congestion model.
"""

import torch
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_layers):
        super(GCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.layers.append(GCNConv(num_features, hidden_channels))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_channels, hidden_channels))
        self.layers.append(GCNConv(hidden_channels, 1))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, training=self.training)
        return x