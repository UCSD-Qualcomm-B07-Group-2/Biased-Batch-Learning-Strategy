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
import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_gcn(args, model, train_data, val_data, optimizer_choice='sgd', lr=0.01, epochs=200, patience=40):
    criterion = torch.nn.MSELoss()
    optimizer = None
    if optimizer_choice == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    elif optimizer_choice == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    elif optimizer_choice == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError("Unsupported optimizer")

    best_val_loss = float('inf')
    patience_counter = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for batch in train_data:
            optimizer.zero_grad()
            x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
            pred = model(x.to(device), edge_index.to(device), batch_idx.to(device))
            loss = criterion(pred.squeeze(), batch.y.to(device).float())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_data)

        model.eval()
        with torch.no_grad():
            x, edge_index = val_data.x, val_data.edge_index
            pred = model(x.to(device), edge_index.to(device))
            val_loss = criterion(pred.squeeze(), val_data.y.to(device).float()).item()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {average_loss}, Validation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Stopping early at epoch {epoch} due to no improvement in validation loss.")
                break

        scheduler.step()

    return model


def train_cluster_gcn(args, model, batchers_train, batchers_val, patience=40, seed=33):
    set_seed(seed)
    optimizer = Adam(model.parameters(), lr=0.1)
    criterion = MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    for epoch in tqdm(range(args.n_epochs)):
        train_loss = []
        val_loss = []

        model.train()
        for batcher in batchers_train:
            if(args.batching_types == 'random'):
                print("poop3")
                loader = batcher.create_random_batches()
            if (args.batching_types == 'random-walk'):
                print("poop2")
                loader = batcher.create_random_walk_batches()
            if (args.batching_types == 'weighted-random-walk'):
                print("poop")
                loader = batcher.create_weighted_random_walk_batches()
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

        # if np.mean(val_loss) < best_val_loss:
        #     best_val_loss = np.mean(val_loss)
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"Stopping early at epoch {epoch} due to no improvement in validation loss.")
        #         break
    return model

def run_training_process(all_datasets, args):
    val_losses = []
    models = []
    predictions = []
    val_preds = []
    iteration_times = []
    memory_usages = []
    actual_test_dataset = all_datasets[-1]
    training_datasets = all_datasets[:-1]
    model_output_dir = os.path.join('saved_models', args.model_output_dir)
    os.makedirs(model_output_dir, exist_ok=True)
    tracemalloc.start()
    if args.task == 'cluster':
        master_batchers = [Batcher(args)(d.to(device)) for d in all_datasets[:-1]]
    for i in range(len(training_datasets)):
        print(f"LOOCV iteration {i + 1}/{len(training_datasets)}: Testing on dataset {i + 1}")
        training_data = [data for j, data in enumerate(training_datasets) if j != i]

        model = AdvancedGCNRegression(num_node_features=all_datasets[0].x.size(1)).to(device)
        if args.task == 'cluster':
            batchers_train = [master_batchers[j] for j in range(len(all_datasets)) if j != i and j != len(all_datasets) - 1]
            batchers_val = [master_batchers[i]]
            start_time = time.time()
            train_cluster_gcn(args, model, batchers_train, batchers_val)
        else:
            merged_training_dataset = merge_datasets(training_data)
            start_time = time.time()
            train_gcn(args, model, [merged_training_dataset], training_datasets[i], 'adam', 0.01, 200)
        model_path = os.path.join(model_output_dir, f"model_{i}.pt")
        torch.save(model.state_dict(), model_path)
        models.append(model_path)

        val_loss, val_pred = test(model, training_datasets[i])
        val_losses.append(val_loss)
        val_preds.append(val_pred)
        print(f'Validation Loss for dataset {i + 1} as test set: {val_loss:.2f}')

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


    actual_labels = actual_test_dataset.y.cpu().numpy()
    individual_losses = [mean_squared_error(actual_labels, preds) for preds in predictions]
    average_individual_loss = np.mean(individual_losses)
    average_test_loss = sum(val_losses) / len(val_losses)
    print(f'Average Valid Loss after LOOCV: {average_test_loss:.2f}')
    print(f'Average Test Loss after LOOCV: {average_individual_loss:.2f}')
    average_predictions = np.mean(predictions, axis=0)
    ensemble_loss = mean_squared_error(actual_labels, average_predictions)
    print(f'Ensemble Loss: {ensemble_loss:.2f}')
    average_memory_usage = sum(memory_usages) / len(memory_usages) / 10 ** 6
    print(f"Average memory usage across all iterations: {average_memory_usage:.2f} MB")
    print(f"Total time taken for all iterations: {sum(iteration_times):.2f} seconds")

    

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


