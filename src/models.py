"""
This module contains the graph models for our congestion model.
"""

import torch
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, args, num_features):
        super().__init__()
        self.args = args
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 1)  # Change the output layer to have one output neuron

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=self.args.drop_rate, train=self.training)
        x = self.conv2(x, edge_index)

        return x