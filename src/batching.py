from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import RandomNodeSplit
import numpy as np
import torch
from partitioning import multiple_fm, KL, mincut_maxflow
import os
import networkx as nx
from torch_geometric.utils import to_networkx, from_networkx

class Batcher:
    """
    A class used to create batches of data.

    ...

    Attributes
    ----------
    batch_size : int
        the size of each batch
    method : str
        the method used to create batches ("random", "fm", "kl", or "mincut")

    Methods
    -------
    __call__(data)
        Creates batches from the given data using the specified method.
    """

    def __init__(self, args, as_graph=False):
        """
        Constructs all the necessary attributes for the Batcher object.

        Parameters
        ----------
            batch_size : int
                the size of each batch (default is 1)
            method : str
                the method used to create batches (default is "random")
            as_graph: bool
                whether to return the batches as a graph or not
        """
        self.args = args
        self.batch_size = args.batch_size
        self.method = args.batcher
        self.as_graph = as_graph
    
    def __call__(self, dataset, G=None):
        """
        Creates batches from the given data using the specified method.

        Parameters
        ----------
            data : Data
                the data to be batched

        Returns
        -------
            batches : generator
                a generator that yields batches of data
        """
        if self.as_graph:
            if self.method == "random":
                return self.random_batch(dataset, self.batch_size)
            elif self.method == "k_clique":
                return self.k_cliques_batch(dataset)
            # elif self.method == "fm":
            #     return self.fm_partitions(dataset, self.batch_size, 10)
            # elif self.method=="kl":
            #     return self.kl_batch(dataset, self.batch_size)
            # elif self.method=="mincut":
            #     return self.min_cut_batch(dataset, self.batch_size)
            else:
                raise ValueError(f"Unknown method {self.method}")
        else:
            if self.method == "random":
                return self.random_batch(dataset, self.batch_size)
            elif self.method == "k_clique":
                return self.k_cliques_batch(dataset)
            # elif self.method == "fm":
            #     return self.fm_batch(dataset, self.batch_size, 10)
            # elif self.method=="kl":
            #     return self.kl_batch(dataset, self.batch_size)
            # elif self.method=="mincut":
            #     return self.min_cut_batch(dataset, self.batch_size)
            else:
                raise ValueError(f"Unknown method {self.method}")
     
    def random_batch(self, data, num_batches):
        """
        Creates batches from the given dataset using random node splitting.

        This method splits the nodes of the graph randomly into batches. Each batch
        contains approximately the same number of nodes.

        Parameters
        ----------
            data : Data
                the data to be batched
            num_batches : int
                the number of batches to create

        Yields
        ------
            Data
                a Data object representing a batch
        """
        for _ in range(num_batches):
            node_mask = RandomNodeSplit(num_test=1/num_batches)(data).test_mask
            yield data.subgraph(node_mask)

    def fm_partitions(self, data, k, max_imbalance):
        edges = [(u.item(), v.item(), w) for u, v ,w in zip(data.edge_index[0],data.edge_index[1], data.edge_attr)]
        G = nx.DiGraph()
        G.add_weighted_edges_from(edges)
        G.remove_edges_from(nx.selfloop_edges(G))
        return multiple_fm(G, [], k, max_imbalance)
    
    def fm_batch(self, data, k, max_imbalance):

        partitions = self.fm_partitions(data, k, max_imbalance)

        for partition in partitions:
            edge_mask = np.isin(data.edge_index, partition.nodes)
            partition_x = data.x[list(partition.nodes)]
            partition_y = data.y[list(partition.nodes)]
            partition_edge_index = data.edge_index[:, np.all(edge_mask, axis=0)]
            yield Data(x=partition_x, y=partition_y, edge_index=partition_edge_index)
    
    def min_cut_batch(self, data, k):
        """
        Creates batches from the given data using the mincut_maxflow algorithm.

        This method partitions the graph into batches using the mincut_maxflow algorithm. Each batch
        contains approximately the same number of nodes.

        Parameters
        ----------
            data : Data
                the data to be batched
            k : int
                the number of partitions to create

        Yields
        ------
            Data
                a Data object representing a batch
        """
        edges = [(u.item(), v.item(), w) for u, v ,w in zip(data.edge_index[0],data.edge_index[1], data.edge_attr)]
        G = nx.DiGraph()
        G.add_weighted_edges_from(edges)
        partitions = mincut_maxflow(G, k)

        for partition in partitions:
            edge_mask = np.isin(data.edge_index, partition.nodes)
            partition_x = data.x[list(partition.nodes)]
            partition_y = data.y[list(partition.nodes)]
            partition_edge_index = data.edge_index[:, np.all(edge_mask, axis=0)]
            yield Data(x=partition_x, y=partition_y, edge_index=partition_edge_index)

    def kl_batch(self, data, k):
        """
        Creates batches from the given data using the Kernighan–Lin (KL) algorithm.

        This method partitions the graph into batches using the KL algorithm. Each batch
        contains approximately the same number of nodes.

        Parameters
        ----------
            data : Data
                the data to be batched
            k : int
                the number of partitions to create

        Yields
        ------
            Data
                a Data object representing a batch
        """
        edges = [(u.item(), v.item(), w) for u, v ,w in zip(data.edge_index[0],data.edge_index[1], data.edge_attr)]
        G = nx.Graph()
        G.add_weighted_edges_from(edges)
        partitions = KL(G, k)

        for partition in partitions:
            edge_mask = np.isin(data.edge_index, partition.nodes)
            partition_x = data.x[list(partition.nodes)]
            partition_y = data.y[list(partition.nodes)]
            partition_edge_index = data.edge_index[:, np.all(edge_mask, axis=0)]
            yield Data(x=partition_x, y=partition_y, edge_index=partition_edge_index)

    def k_cliques_batch(self, datasets):
            
            data_list = []
            for data in datasets:
                G = to_networkx(data, to_undirected=True)
                cliques = list(nx.algorithms.community.k_clique_communities(G, self.args.batch_size))

                # Convert cliques to Data objects
                for clique in cliques:
                    nodes = list(clique)
                    subgraph = G.subgraph(nodes)
                    clique_data = from_networkx(subgraph)
                    clique_data.x = data.x[nodes]
                    clique_data.y = data.y[nodes]
                    data_list.append(clique_data)

            # Create a DataLoader for batching
            loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=True)    
            return loader