from arguments import params
# from torch_geometric.data import ClusterData, ClusterLoader

from torch.optim import Adam
from torch.nn import MSELoss
from torch import load as torch_load
import torch.nn.functional as F

import os

from models import GCN
from batching import Batcher
from load import load_cache

device = 'cpu'

def train_gcn(args, model, data):
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = MSELoss()  # Change the loss function to MSE

    data.x = data.x.float()  # Convert node features to float
    data.y = data.y.float()

    model.train()
    for epoch in range(2000):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out.reshape(-1), data.y)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch} Model Loss: {loss.item()}")


def train_cluster_gcn(args, model, loader):
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = MSELoss()

    model.train()
    for epoch in range(args.n_epochs):
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index)
            loss =  F.mse_loss(out.reshape(-1), batch.y)  # Compute loss on the batch
            loss.backward()
            optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch {epoch} Model Loss: {loss.item()}")


def run_eval(args, model, data, split="train"):
    pass


    

if __name__ == '__main__':
    args = params()
    print('Params initialized')

    # write a function that checks if cache exists
    print('Loading Cache')
    dataset = load_cache(args)
    print(f'Loaded {len(dataset)} from cache')

    if args.task == 'baseline':
        model = GCN(22, 128, 4)
        train_gcn(args, model, dataset[0])
        run_eval(args, model, dataset, split='test')
    elif args.task == 'cluster':

        model = GCN(22, 16, 4)
        print('Model initialized')
        batcher = Batcher(args)
        batcher = batcher(dataset)
        print('Data loader initialized')
        train_cluster_gcn(args, model, batcher)
        # run_eval(args, model, dataset, split='test')


