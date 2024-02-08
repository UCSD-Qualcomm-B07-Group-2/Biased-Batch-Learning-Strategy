from arguments import params
from torch_geometric.data import ClusterData, ClusterLoader

from torch import load as torch_load
import os

from models import GCN

device = 'cpu'

def load_cache():
    # Construct the file path
    datasets = []
    for m in range(1, 4):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, '..', 'cache', f'data_{m}.pt')

        # Check if the file exists
        if os.path.exists(file_path):
            # Load data from cache
            data = torch_load(file_path)
            datasets.append(data)

    # If the file does not exist, return None
    return datasets

if __name__ == '__main__':
    args = params()

    # write a function that checks if cache exists
    datasets = load_cache()
    batcher = ClusterData(datasets[0], 10)
    print(batcher)


    # if args.task == 'baseline':
    #     model = GCN(args, batcher, target_size=60).to(device)
    #     baseline_train(args, model, datasets, batcher)
    #     run_eval(args, model, datasets, batcher, split='test')
    # elif args.task == 'custom': # you can have multiple custom task for different techniques
    #     model = CustomModel(args, batcher, target_size=60).to(device)
    #     custom_train(args, model, datasets, batcher)
    #     run_eval(args, model, datasets, batcher, split='test')


