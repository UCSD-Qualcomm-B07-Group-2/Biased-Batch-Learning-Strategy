import networkx as nx

def mincut_maxflow(G):
    return ... #G.minimum_cut()

def KL(G):
    return G.kernighan_lin_bisection(G, max_iter=1000)