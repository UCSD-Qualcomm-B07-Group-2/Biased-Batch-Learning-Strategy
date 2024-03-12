from arguments import params
# from torch_geometric.data import ClusterData, ClusterLoader

from torch.optim import Adam
from torch.nn import MSELoss
from torch import load as torch_load
import torch.nn.functional as F
from torch import no_grad
import torch
import os
from utils import *
from models import AdvancedGCNRegression
from batching import Batcher
from load import *
import numpy as np
from tqdm import tqdm
import tracemalloc
import time
from sklearn.metrics import mean_squared_error, r2_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def train_gcn(args, model, data, optimizer_choice='adam', lr=0.01, epochs=200):
    criterion = torch.nn.MSELoss()
    if optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    elif optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_choice == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer")

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    model.train()
    for epoch in tqdm(range(args.n_epochs)):
        total_loss = 0
        for train_data in data:
            optimizer.zero_grad()
            # Unpack the batch data
            x, edge_index, batch = train_data.x, train_data.edge_index, train_data.batch
            # If the model expects edge attributes, include them as well
            edge_attr = train_data.edge_attr if 'edge_attr' in train_data else None

            # Adjust the model's forward call according to its expected input
            if edge_attr is not None:
                pred = model(x, edge_index, edge_attr, batch).to(device)
            else:
                pred = model(x, edge_index, batch).to(device)

            loss = criterion(pred.squeeze(), train_data.y.to(device).float())
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
        average_loss = total_loss / len(data)
    return model


def train_cluster_gcn(args, model, batchers_train, batchers_val):

    optimizer = Adam(model.parameters(), lr=0.1)
    criterion = MSELoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    for epoch in tqdm(range(args.n_epochs)):
        train_loss = []
        val_loss = []

        model.train()
        for batcher in batchers_train:
            loader = batcher.create_random_batches()
            for i, batch in enumerate(loader):
                optimizer.zero_grad()
                out = model(batch.x, batch.edge_index).to(device)
                loss = F.mse_loss(out.reshape(-1), batch.y)  # Compute loss on the batch
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()
                scheduler.step()
        
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


        if epoch % 10 == 0:
            print(f"Epoch {epoch} Train Loss: {np.mean(train_loss)} Validation Loss: {np.mean(val_loss)}")
    return model


def run_training_process(all_datasets, args):
    test_losses = []
    models = []
    predictions = []
    iteration_times = []
    memory_usages = []
    actual_test_dataset = all_datasets[-1]
    training_datasets = all_datasets[:-1]

    tracemalloc.start()
    master_batchers = [Batcher(args)(d.to(device)) for d in all_datasets[:-1]]
    for i in range(len(training_datasets)):
        print(f"LOOCV iteration {i + 1}/{len(training_datasets)}: Testing on dataset {i + 1}")
        training_data = [data for j, data in enumerate(training_datasets) if j != i]

        model = AdvancedGCNRegression(num_node_features=all_datasets[0].x.size(1),
                            num_edge_features=(0 if all_datasets[0].edge_attr is None else all_datasets[0].edge_attr.size(1))
                            ).to(device)
        if args.task == 'cluster':
            batchers_train = [master_batchers[j] for j in range(len(all_datasets)) if j != i and j != len(all_datasets) - 1]
            batchers_val = [master_batchers[i]]
            start_time = time.time()
            train_cluster_gcn(args, model, batchers_train, batchers_val)
        else:
            merged_training_dataset = merge_datasets(training_data)
            start_time = time.time()
            train_gcn(args, model, [merged_training_dataset], 'adam', 0.01, 200)
        model_path = f"model_{i}.pt"
        torch.save(model.state_dict(), model_path)
        models.append(model_path)

        test_loss = test(model, training_datasets[i])
        test_losses.append(test_loss)
        print(f'Test Loss for dataset {i + 1} as test set: {test_loss:.2f}')

        model.eval()
        with torch.no_grad():
            preds = model(actual_test_dataset.x.to(device), actual_test_dataset.edge_index.to(device))
            predictions.append(preds.cpu().numpy())

        end_time = time.time()
        iteration_duration = end_time - start_time
        iteration_times.append(iteration_duration)
        print(f"Time taken for iteration {i + 1}: {iteration_duration:.2f} seconds")
        current, peak = tracemalloc.get_traced_memory()
        memory_usages.append(peak)
        print(f"Current memory usage: {current / 10 ** 6}MB; Peak: {peak / 10 ** 6}MB")

    tracemalloc.stop()

    average_test_loss = sum(test_losses) / len(test_losses)
    print(f'Average Valid Loss after LOOCV: {average_test_loss:.2f}')
    actual_labels = actual_test_dataset.y.cpu().numpy()
    individual_losses = [mean_squared_error(actual_labels, preds) for preds in predictions]
    average_individual_loss = np.mean(individual_losses)
    print(f'Average Test Loss after LOOCV: {average_individual_loss:.2f}')
    average_predictions = np.mean(predictions, axis=0)
    ensemble_loss = mean_squared_error(actual_labels, average_predictions)
    print(f'Ensemble Loss: {ensemble_loss:.2f}')
    average_memory_usage = sum(memory_usages) / len(memory_usages) / 10 ** 6
    print(f"Average memory usage across all iterations: {average_memory_usage:.2f} MB")
    print(f"Total time taken for all iterations: {sum(iteration_times):.2f} seconds")



def run_eval(args, model, dataset, split="val"):
    '''
    Should return the loss
    '''
    pass


    

if __name__ == '__main__':
    args = params()
    print('Params initialized')
    all_datasets = load_cache_v2()
    print("loaded")
    run_training_process(all_datasets, args)
    # write a function that checks if cache exists
    # print('Loading Cache')
    # dataset = load_cache(args)
    # print(f'Loaded {len(dataset)} from cache')
    #
    # train = [dataset[0], dataset[2], dataset[3]]
    # val = [dataset[4]]
    # test = [dataset[5]]

    # if args.task == 'baseline':
    #     model = AdvancedGCNRegression(22, 128, 4)
    #     train_gcn(args, model, dataset[0])
    #     run_eval(args, model, dataset, split='test')
    # elif args.task == 'cluster':

    # model = AdvancedGCNRegression(26, 0)
    # print('Model initialized')
    # batchers_train = [Batcher(args)(d) for d in train]
    # batchers_val = [Batcher(args)(d) for d in val]
    # batchers_test = [Batcher(args)(d) for d in test]

    # HOW TO CALL DIFFERENT TYPE OF BATCHING
    # print(batcher.create_random_batches())
    # print(batcher.create_random_walk_batches())
    # print(batcher.create_weighted_random_walk_batches())

    # print('Data loader initialized')

    # SRUJAN OR WHOEVER REPLACE THIS WITH YOUR MODEL TRAINING. MAKE SURE TO TAKE IN TRAINING ARGS AND BATCHER
    # train_cluster_gcn(args, model, batchers_train, batchers_val)
    # run_eval(args, model, dataset, split='test')


