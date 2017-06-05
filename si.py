import random
import numpy as np
from copy import copy
from graph_tool import Graph


def gen_cascade(g, p, source=None, stop_fraction=0.5):
    if source is None:
        source = random.choice(np.arange(g.num_vertices()))
    infected = {source}
    infection_times = np.ones(g.num_vertices()) * -1
    infection_times[source] = 0
    time = 0
    edges = []
    while np.count_nonzero(infection_times != -1) / g.num_vertices() <= stop_fraction:
        infected_nodes_until_t = copy(infected)
        time += 1
        for i in infected_nodes_until_t:
            for j in g.vertex(i).all_neighbours():
                j = int(j)
                if j not in infected and random.random() <= p:
                    infected.add(j)
                    infection_times[j] = time
                    edges.append((i, j))

    tree = Graph(directed=True)
    for _ in range(g.num_vertices()):
        tree.add_vertex()
    for u, v in edges:
        tree.add_edge(u, v)
    return source, infection_times, tree
