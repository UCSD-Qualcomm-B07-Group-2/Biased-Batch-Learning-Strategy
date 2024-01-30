from batching import Batcher
from load import load_data
# import psutil
import memory_profiler

# process = psutil.Process()
# mem_before = process.memory_info().rss
# mem_after = process.memory_info().rss
# print(f'Memory used by data memory usage: {mem_after - mem_before}')

@profile
def batching():
    data = load_data()
    batcher = Batcher(batch_size=4, method="random")
    batch_generator = batcher(data)

    for batch in batch_generator:
        print(batch)

@profile
def non_batching():
    data = load_data()
    batcher = Batcher(batch_size=4, method="random")
    batch_generator = batcher(data)
    batches = list(batch_generator)
    for batch in batches:
        print(batch)

if __name__ == '__main__':
    # non_batching()
    batching()


