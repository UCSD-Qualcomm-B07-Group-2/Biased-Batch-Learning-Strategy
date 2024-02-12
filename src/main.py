from arguments import params
# from torch_geometric.data import ClusterData, ClusterLoader

from torch.optim import Adam
from torch.nn import MSELoss
from torch import load as torch_load

import os

from models import GCN
from batching import Batcher

device = 'cpu'

def load_cache(args):
    # Construct the file path
    datasets = []
    for m in range(1, 2):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, '..', 'cache', f'data_{m}.pt')

        # Check if the file exists
        if os.path.exists(file_path):
            # Load data from cache
            data = torch_load(file_path)
            datasets.append(data)

    # If the file does not exist, return None
    return datasets

def train_gcn(args, model, dataset):
    optimizer = Adam(model.parameters(), lr=0.001)
    criterion = MSELoss()  # Change the loss function to MSE


    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = criterion(out, data.y)
        print(f"Epoch {epoch} Model Loss: {loss.item()}")
        loss.backward()
        optimizer.step()


def train_cluster_gcn(args, model, loader):
    from torch_geometric.utils import to_networkx
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = MSELoss()

    model.train()
    for epoch in range(args.n_epochs):
        for batch in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.float())  # Compute loss on the batch
            loss.backward()
            optimizer.step()
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
        model = GCN(args, num_features=4).to(device)
        train_gcn(args, model, dataset)
        run_eval(args, model, dataset, split='test')
    elif args.task == 'cluster':

        model = GCN(args, num_features=4).to(device)
        print('Model initialized')
        batcher = Batcher(args)
        batcher = batcher(dataset)
        print('Data loader initialized')
        train_cluster_gcn(args, model, batcher)
        # run_eval(args, model, dataset, split='test')


