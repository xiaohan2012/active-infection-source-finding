import math
import networkx as nx
import random
import numpy as np
from networkx.generators.classic import empty_graph


def kronecker_random_graph(k, P, seed=None, directed=True, n_edges=None):
    dim = len(P)
    
    errorstring = ("The initiator matrix must be a nonempty" +
                      (", symmetric," if not directed else "") +
                      " square matrix of probabilities.")

    if dim == 0:
        raise nx.NetworkXError(errorstring)
    for i, arr in enumerate(P):
        if len(arr) != dim:
            raise nx.NetworkXError(errorstring)
        for j, p in enumerate(arr):
            if p < 0 or p > 1:
                raise nx.NetworkXError(errorstring)
            if not directed and P[i][j] != P[j][i]:
                raise nx.NetworkXError(errorstring)

    if k < 1:
        return empty_graph(1)

    n = dim**k
    G = empty_graph(n)
    G = nx.DiGraph(G)

    acc = 0.0
    partitions = []
    for i in range(dim):
        for j in range(dim):
            if P[i][j] != 0:
                acc = acc+P[i][j]
                partitions.append([acc, i, j])
    psum = acc

    G.add_nodes_from(range(n))
    G.name = "kronecker2_random_graph(%s, %s)".format(n, P)

    if seed is not None:
        random.seed(seed)
    
    if n_edges is None:
        expected_edges = math.floor(psum**k)
    else:
        expected_edges = n_edges
    num_edges = 0
    while num_edges < expected_edges:
        multiplier = dim**k
        x = y = 0
        for i in range(k):
            multiplier = multiplier // dim
            r = c = -1
            p = random.uniform(0, psum)
            for n in range(len(partitions)):
                if partitions[n][0] >= p:
                    r = partitions[n][1]
                    c = partitions[n][2]
                    break
            x = x + r*multiplier
            y = y + c*multiplier

        if not G.has_edge(x, y):
            G.add_edge(x, y)
            num_edges = num_edges + 1

    if not directed:
        G = G.to_undirected()

    return G

P_peri = np.array([[0.9, 0.1], [0.1, 0.3]])
P_heir = np.array([[0.9, 0.1], [0.1, 0.9]])
P_rand = np.array([[0.5, 0.5], [0.5, 0.5]])
