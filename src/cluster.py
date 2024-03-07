import copy
import sys
from dataclasses import dataclass
from typing import List, Literal, Optional
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.typing import pyg_lib
from torch_geometric.utils import index_sort, narrow, select, sort_edge_index
from torch_geometric.utils.map import map_index
from torch_geometric.utils.sparse import index2ptr, ptr2index
import numpy as np
from scipy.sparse import coo_matrix
import networkx as nx
from load import load_cache
from torch_geometric.loader import ClusterData, ClusterLoader
from torch_geometric.data import Batch

import os.path as osp

import torch.utils.data

import torch_geometric.typing

import matplotlib.pyplot as plt


def metis(edge_index: Tensor, num_nodes: int, num_parts: int) -> Tensor:
        # Computes a node-level partition assignment vector via METIS.
        row, index = sort_edge_index(edge_index, num_nodes=num_nodes)
        indptr = index2ptr(row, size=num_nodes)


        # Compute METIS partitioning:
        cluster: Optional[Tensor] = None

        if torch_geometric.typing.WITH_TORCH_SPARSE:
            try:
                cluster = torch.ops.torch_sparse.partition(
                    indptr.cpu(),
                    index.cpu(),
                    None,
                    num_parts,
                    False,
                ).to(edge_index.device)
            except (AttributeError, RuntimeError):
                pass

        if cluster is None and torch_geometric.typing.WITH_METIS:
            cluster = pyg_lib.partition.metis(
                indptr.cpu(),
                index.cpu(),
                num_parts,
                recursive=False,
            ).to(edge_index.device)

        if cluster is None:
            raise ImportError(f"requires either "
                              f"'pyg-lib' or 'torch-sparse'")

        return cluster

def create_graph_and_clusters(data, num_parts):

    G = nx.Graph()

    for i in range(num_parts):
        G.add_node(i)

    cluster = metis(data.edge_index, data.x.shape[0], 50)
    cluster_new, node_perm = index_sort(cluster, max_value=num_parts)
    partptr = index2ptr(cluster_new, size=num_parts)

    nodes_to_clusters = {}
    for node_id, cluster_id in enumerate(cluster.tolist()):
        nodes_to_clusters[node_id] = cluster_id

    cluster_data = list(ClusterData(data, num_parts))

    for i, cd in enumerate(cluster_data):
        mapped = cd.edge_index + partptr[i]
        mapped = node_perm[mapped]
        cd.original_edge_index = mapped
        cd.between_edges = {}
        cluster_data[i] = cd

    original_indices = torch.argsort(node_perm)
    for source_node, target_node in data.edge_index.t().tolist():
        source_cluster = nodes_to_clusters[source_node]
        target_cluster = nodes_to_clusters[target_node]
        if source_cluster != target_cluster:

            min_cluster = min(source_cluster, target_cluster)
            max_cluster = max(source_cluster, target_cluster)

            if min_cluster == source_cluster:
                min_node = source_node
                max_node = target_node
            else:
                min_node = target_node
                max_node = source_node

            between_edges = cluster_data[min_cluster].between_edges
            tensor1 = between_edges.get(max_cluster, torch.tensor([]).int())
            tensor2 = torch.tensor([[original_indices[min_node] - partptr[min_cluster], original_indices[max_node] - partptr[max_cluster]]]).T
            between_edges[max_cluster] = torch.cat([tensor1, tensor2], dim=1)

            if G.has_edge(source_cluster, target_cluster):
                # If the edge exists, increment its weight
                G[source_cluster][target_cluster]['weight'] += 1
            else:
                # If the edge does not exist, add it with an initial weight
                G.add_edge(source_cluster, target_cluster, weight=1)

    return G, cluster_data, node_perm

def sample_groups(G, q, method="random"):
    nodes = list(G.nodes())
    groups = []
    while len(nodes) > 0:
        if len(nodes) < q:
            group = nodes
        else:
            if method == "rw":
                group = random_walk(nodes, q)
            elif method == "wrw":
                group = weighted_random_walk(G, nodes, q)
            else:
                group = np.random.choice(nodes, q, replace=False)
        nodes = [n for n in nodes if n not in group]
        groups.append(group)
    return groups

def random_walk(nodes, q):
    # Start at a random node
    node = np.random.choice(list(nodes))
    subgraph = G.subgraph(nodes)
    walk = [node]

    # Perform the random walk
    for _ in range(q - 1):
        neighbors = [n for n in subgraph.neighbors(node) if n not in walk]
        if neighbors:
            node = np.random.choice(neighbors)
            walk.append(node)
        else:
            break

    return walk

def weighted_random_walk(G, nodes, q):
    # Start at a random node
    node = np.random.choice(list(nodes))
    subgraph = G.subgraph(nodes)
    walk = [node]

    # Perform the weighted random walk
    for _ in range(q - 1):
        neighbors = [n for n in subgraph.neighbors(node) if n not in walk]
        if neighbors:
            # Get the weights of the edges to the neighbors
            weights = [subgraph[node][n]['weight'] for n in neighbors]
            # Normalize the weights
            weights = [w / sum(weights) for w in weights]
            # Choose a neighbor based on the weights
            node = np.random.choice(neighbors, p=weights)
            walk.append(node)
        else:
            break

    return walk

def create_batch(clusters, cluster_ids):
    batch = None
    edge_index = []
    x = []
    y = []
    between_edges = []
    new_partptr = [cluster[1].x.shape[0] for cluster in clusters]
    new_partptr = np.insert(np.cumsum(new_partptr), 0, 0)
    prev_cluster_id = None
    cluster_id_to_id = dict(zip(cluster_ids, range(len(cluster_ids))))

    for i, cluster in enumerate(clusters):
        cluster_id, data = cluster

        # building combined edge_index
        edge_index.append(data.edge_index + new_partptr[i])
        # building combined x data
        x.append(data.x)
        # building combined y data
        y.append(data.y)
        
        # building combined between_edges
        for target_cluster_id, edges in data.between_edges.items():
            if target_cluster_id in cluster_ids:
                edges[0] += new_partptr[cluster_id_to_id[cluster_id]]    
                edges[1] += new_partptr[cluster_id_to_id[target_cluster_id]] 
                edge_index.append(edges)
    
        

    return Data(x=torch.cat(x, dim=0), y=torch.cat(y, dim=0), edge_index=torch.cat(edge_index, dim=1))

def create_batches(groups, cluster_data):
    batches = []
    for group in groups:
        clusters = []
        for cluster_id in sorted(group):
            clusters.append  ((cluster_id, cluster_data[cluster_id]))
        batch = create_batch(clusters, sorted(group))
        batches.append(batch)
        # batches.append(batch)
    
    return batches


def example():
    np.random.seed(42)

    # load in data
    data = load_cache()
    d = data[0]

    # this creates clusters
    G, clusters, node_perm = create_graph_and_clusters(d, 50)

    # this will perform weighted random walk and create the cluster groups
    groups = sample_groups(G, 3, method="wrw")

    # given the groups this will batch them
    batches = create_batches(groups, clusters)

    return batches
