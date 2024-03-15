"""
This module contains the graph models for our congestion model.
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d, Dropout
from torch_geometric.nn import GCNConv, GATConv

class AdvancedGCNRegression(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, conv1_out_features=32, conv2_out_features=64, conv3_out_features=64, gat_out_features=64, gat_heads=4, dropout_rate=0.2):
        super(AdvancedGCNRegression, self).__init__()
        self.conv1 = GCNConv(num_node_features, conv1_out_features)
        self.bn1 = BatchNorm1d(conv1_out_features)
        self.conv2 = GCNConv(conv1_out_features, conv2_out_features)
        self.bn2 = BatchNorm1d(conv2_out_features)
        self.conv3 = GCNConv(conv2_out_features, conv3_out_features)
        self.bn3 = BatchNorm1d(conv3_out_features)
        self.attention = GATConv(conv3_out_features, gat_out_features, heads=gat_heads, concat=True)
        self.bn_attention = BatchNorm1d(gat_out_features * gat_heads)
        self.dropout = Dropout(dropout_rate)
        self.lin = Linear(gat_out_features * gat_heads, 1)
        
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.attention(x, edge_index))
        x_res = F.relu(self.conv3(x, edge_index))
        x = x + x_res
        x = self.dropout(x)
        x = F.relu(self.attention(x, edge_index))
        x = F.relu(self.lin(x))
        
        return x
