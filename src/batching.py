from torch_geometric.data import Data
from torch_geometric.transforms import RandomNodeSplit
import numpy as np
import torch
from partitioning import multiple_fm, KL, mincut_maxflow
import os
import networkx as nx

class Batcher:

    def __init__(self, batch_size=1, method="random"):
        self.batch_size = batch_size
        self.method = method
    
    def __call__(self, data):
        if self.method == "random":
            return self.random_batch(data, self.batch_size)
        elif self.method == "fm":
            return self.fm_batch(data, self.batch_size, 11)
        elif self.method=="kl":
            return self.kl_batch(data, self.batch_size)
        elif self.method=="mincut":
            return self.min_cut_batch(data, self.batch_size)
        else:
            raise ValueError(f"Unknown method {self.method}")
    

    
    def random_batch(self, data, num_batches):
        res_partitions = []
        for _ in range(num_batches):
            node_mask = RandomNodeSplit(num_test=1/num_batches)(data).test_mask
            edge_mask = np.isin(data.edge_index, np.argwhere(node_mask))
            partition_x = data.x[node_mask]
            partition_y = data.y[node_mask]
            partition_edge_index = data.edge_index[:, np.all(edge_mask, axis=0)]
            # res_partitions.append(Data(x=partition_x, y=partition_y, edge_index=partition_edge_index))
            yield Data(x=partition_x, y=partition_y, edge_index=partition_edge_index)
        # return res_partitions
    
    def fm_batch(self, data, k, max_imbalance):
        edges = [(u.item(), v.item(), w) for u, v ,w in zip(data.edge_index[0],data.edge_index[1], data.edge_attr)]

        G = nx.DiGraph()
        G.add_weighted_edges_from(edges)

        partitions = multiple_fm(G, [], k, max_imbalance)

        res_partitions = []

        for partition in partitions:
            edge_mask = np.isin(data.edge_index, partition.nodes)
            partition_x = data.x[list(partition.nodes)]
            partition_y = data.y[list(partition.nodes)]
            partition_edge_index = data.edge_index[:, np.all(edge_mask, axis=0)]
            yield Data(x=partition_x, y=partition_y, edge_index=partition_edge_index)
            # res_partitions.append(Data(x=partition_x, y=partition_y, edge_index=partition_edge_index))


        return res_partitions
    
    def min_cut_batch(self, data, k):
        edges = [(u.item(), v.item(), w) for u, v ,w in zip(data.edge_index[0],data.edge_index[1], data.edge_attr)]

        G = nx.DiGraph()
        G.add_weighted_edges_from(edges)

        partitions = mincut_maxflow(G, k)

        return partitions

        # res_partitions = []

        # for partition in partitions:
        #     edge_mask = np.isin(data.edge_index, partition.nodes)
        #     partition_x = data.x[list(partition.nodes)]
        #     partition_y = data.y[list(partition.nodes)]
        #     partition_edge_index = data.edge_index[:, np.all(edge_mask, axis=0)]
        #     res_partitions.append(Data(x=partition_x, y=partition_y, edge_index=partition_edge_index))

        # return res_partitions

    def kl_batch(self, data, k):
        edges = [(u.item(), v.item(), w) for u, v ,w in zip(data.edge_index[0],data.edge_index[1], data.edge_attr)]

        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        
        partitions = KL(G, k)

        res_partitions = []

        for partition in partitions:
            edge_mask = np.isin(data.edge_index, partition.nodes)
            partition_x = data.x[list(partition.nodes)]
            partition_y = data.y[list(partition.nodes)]
            partition_edge_index = data.edge_index[:, np.all(edge_mask, axis=0)]
            res_partitions.append(Data(x=partition_x, y=partition_y, edge_index=partition_edge_index))

        return res_partitions