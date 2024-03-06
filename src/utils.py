import torch
from torch_geometric.data import Data, DataLoader


def inductive_split(data_list, train_prop, val_prop, test_prop, features):
    # Calculate the number of samples for each split
    total_samples = len(data_list)
    train_samples = int(total_samples * train_prop)
    val_samples = int(total_samples * val_prop)
    test_samples = int(total_samples * test_prop)

    # Split the data list into train, val, and test sets
    train_set = data_list[:train_samples]
    val_set = data_list[train_samples:train_samples+val_samples]
    test_set = data_list[train_samples+val_samples:]

    # Select the desired features from the data sets
    train_set = [Data(x=d.x[features], edge_index=d.edge_index, edge_attr=d.edge_weights, y=d.y) for d in train_set]
    val_set = [Data(x=d.x[features], edge_index=d.edge_index, edge_attr=d.edge_weights, y=d.y) for d in val_set]
    test_set = [Data(x=d.x[features], edge_index=d.edge_index, edge_attr=d.edge_weights, y=d.y) for d in test_set]

    # Convert the data sets to DataLoader objects with batch size 1
    train_loader = DataLoader(train_set, batch_size=1)
    val_loader = DataLoader(val_set, batch_size=1)
    test_loader = DataLoader(test_set, batch_size=1)

    return train_loader, val_loader, test_loader


