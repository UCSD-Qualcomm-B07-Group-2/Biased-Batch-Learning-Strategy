import networkx as nx
import random
import numpy as np
import pandas as pd
import heapq
from scipy.sparse import coo_matrix
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from networkx.algorithms.community import kernighan_lin_bisection as kl_bisection

def calculate_initial_gain(G, node, partition):
    internal, external = 0, 0
    for neighbor in G[node]:
        if neighbor in partition:
            internal += G[node][neighbor].get('weight', 1)
        else:
            external += G[node][neighbor].get('weight', 1)
    return external - internal

def update_gains_and_queue(G, node, partition_a, partition_b, heap, node_to_heap):
    for neighbor in G[node]:
        if neighbor not in node_to_heap:
            continue
        if neighbor in partition_a:
            new_gain = calculate_initial_gain(G, neighbor, partition_a)
        else:
            new_gain = calculate_initial_gain(G, neighbor, partition_b)
        
        heap_entry = node_to_heap[neighbor]
        heap_entry[0] = -new_gain  # Negate for max-heap
        heapq.heapify(heap)  # Re-heapify

def is_balanced(partition_a, partition_b, max_imbalance):
    size_a = len(partition_a)
    size_b = len(partition_b)
    return abs(size_a - size_b) <= max_imbalance

def create_subgraphs(G, partition_a, partition_b):
    # Create subgraphs for each partition
    subgraph_a = G.subgraph(partition_a).copy()
    subgraph_b = G.subgraph(partition_b).copy()

    return subgraph_a, subgraph_b

def mincut_maxflow(G, num_splits):
    partitions = []
    
    def split(G, i):
        if i == 0 or G.number_of_nodes() <= 1:
            partitions.append(G)
        else:
            source = np.random.choice([node for node in G.nodes if G.in_degree(node)==0])
            sink = np.random.choice([node for node in G.nodes if G.out_degree(node)==0])
            while source == sink:
                sink = np.random.choice([node for node in G.nodes if G.out_degree(node==0)])
            part_A, part_B = nx.minimum_cut(G, source, sink)[1]
            A = split(G.subgraph(part_A), i-1)
            B = split(G.subgraph(part_B), i-1)
    
    split(G, num_splits)
    return partitions

def KL(G, num_splits, max_iter=None):
    partitions = []
    
    def partition(G, i):
        if i == 0:
            partitions.append(G)
        else:
            if max_iter:
                part_A, part_B = kl_bisection(G, max_iter=max_iter)
            else:
                part_A, part_B = kl_bisection(G, max_iter=len(G.nodes)//2)

            A = partition(G.subgraph(part_A), i - 1)
            B = partition(G.subgraph(part_B), i - 1)

    partition(G, num_splits)
    return partitions

def multiple_fm(G, subgraphs, k, max_imbalance):
    
    if k == 0:
        subgraphs.append(G)
        return G
    
    p_a, p_b = fm_partition(G, k, max_imbalance)
    subgraph_a, subgraph_b = create_subgraphs(G, p_a, p_b)
    
    multiple_fm(subgraph_a, subgraphs, k - 1, max_imbalance)
    multiple_fm(subgraph_b, subgraphs, k - 1, max_imbalance)
    
    return subgraphs

def fm_partition(G, max_passes=10, max_imbalance=1):
    nodes = list(G.nodes())  # Convert Nodes to a list
    random.shuffle(nodes)  # Shuffle to ensure random partitioning
    
    partition_a, partition_b = set(nodes[:len(G)//2]), set(nodes[len(G)//2:])
    heap = []
    node_to_heap = {}

    # Initialize gains and heap
    for node in G:
        gain = calculate_initial_gain(G, node, partition_a if node in partition_a else partition_b)
        heap_entry = [-gain, node]  # Negate gain for max-heap
        heapq.heappush(heap, heap_entry)
        node_to_heap[node] = heap_entry

    for _ in range(max_passes):
        if not heap:
            break

        # Find a suitable node to move
        while heap:
            gain, node = heapq.heappop(heap)
            gain = -gain  # Correct the negated gain

            if node in partition_a and is_balanced(partition_a - {node}, partition_b | {node}, max_imbalance):
                partition_a.remove(node)
                partition_b.add(node)
                break
            elif node in partition_b and is_balanced(partition_a | {node}, partition_b - {node}, max_imbalance):
                partition_b.remove(node)
                partition_a.add(node)
                break

        if not heap:
            # No suitable node found
            break

        # Lock this node
        del node_to_heap[node]

        # Update gains of neighbors
        update_gains_and_queue(G, node, partition_a, partition_b, heap, node_to_heap)

    return partition_a, partition_b