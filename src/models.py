"""
This module contains the database models for the application.
"""

import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import DataLoader, Batch

# data.x = data.x.float()  # Convert node features to float
# data.y = data.y.float()
# mean = data.x.mean(dim=0, keepdim=True)
# std = data.x.std(dim=0, keepdim=True)
# data.x = (data.x - mean) / std

class GCN(torch.nn.Module):
    def __init__(self, args, num_features):
        super(GCN, self).__init__()
        self.args = args
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, 1)  # Change the output layer to have one output neuron

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = torch.dropout(x, p=0.5, train=self.training)
        x = self.conv2(x, edge_index)

        return x

def train_gcn(model, data, epochs=100):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()  # Change the loss function to MSE
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        print(f"Epoch {epoch} Model Loss: {loss.item()}")
        loss.backward()
        optimizer.step()
