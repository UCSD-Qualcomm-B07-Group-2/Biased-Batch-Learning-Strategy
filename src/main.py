from arguments import params
# from torch_geometric.data import ClusterData, ClusterLoader

from torch.optim import Adam
from torch.nn import MSELoss
from torch import load as torch_load
import torch.nn.functional as F
from torch import no_grad
import torch
import os

from models import AdvancedGCNRegression
from batching import Batcher
from load import load_cache
import numpy as np

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


def train_cluster_gcn(args, model, batchers_train, batchers_val):

    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = MSELoss()


    for epoch in range(args.n_epochs):
        train_loss = []
        val_loss = []

        model.train()
        for batcher in batchers_train:
            loader = batcher.create_random_batches()
            for i, batch in enumerate(loader):
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index)
                loss =  F.mse_loss(out.reshape(-1), batch.y)  # Compute loss on the batch
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
        
        model.eval()
        with no_grad():
            for batcher in batchers_val:
                loader = batcher.create_random_batches()
                for batch in loader:
                    # print(batch.edge_index.max())
                    out = model(batch.x, batch.edge_index)
                    loss =  F.mse_loss(out.reshape(-1), batch.y.float())  # Compute loss on the batch
                    # print(out.shape[0], batch.y.shape[0], loss.item())
                    val_loss.append(loss.item())


        if epoch % 100 == 0:
            print(f"Epoch {epoch} Train Loss: {np.mean(train_loss)} Validation Loss: {np.mean(val_loss)}")



def run_eval(args, model, data, split="train"):
    pass


    

if __name__ == '__main__':
    args = params()
    print('Params initialized')

    # write a function that checks if cache exists
    print('Loading Cache')
    dataset = load_cache(args)
    print(f'Loaded {len(dataset)} from cache')

    train = [dataset[0], dataset[2], dataset[3]]
    val = [dataset[4]]
    test = [dataset[5]]

    # if args.task == 'baseline':
    #     model = AdvancedGCNRegression(22, 128, 4)
    #     train_gcn(args, model, dataset[0])
    #     run_eval(args, model, dataset, split='test')
    # elif args.task == 'cluster':

    model = AdvancedGCNRegression(26, 0)
    print('Model initialized')
    batchers_train = [Batcher(args)(d) for d in train]
    batchers_val = [Batcher(args)(d) for d in val]
    # batchers_test = [Batcher(args)(d) for d in test]

    # HOW TO CALL DIFFERENT TYPE OF BATCHING
    # print(batcher.create_random_batches())
    # print(batcher.create_random_walk_batches())
    # print(batcher.create_weighted_random_walk_batches())

    print('Data loader initialized')

    # SRUJAN OR WHOEVER REPLACE THIS WITH YOUR MODEL TRAINING. MAKE SURE TO TAKE IN TRAINING ARGS AND BATCHER
    train_cluster_gcn(args, model, batchers_train, batchers_val)
    # run_eval(args, model, dataset, split='test')


