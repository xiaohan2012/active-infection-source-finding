import random
import numpy as np
from copy import copy
from graph_tool import GraphView


def gen_cascade(g, p, source=None, stop_fraction=0.5, ):
    if source is None:
        source = random.choice(np.arange(g.num_vertices()))
    infected = {source}
    infection_times = np.ones(g.num_vertices()) * -1
    infection_times[source] = 0
    time = 0
    edges = []
    while np.count_nonzero(infection_times != -1) / g.num_vertices() <= stop_fraction:
        infected_nodes_until_t = copy(infected)
        for i in infected_nodes_until_t:
            for j in g.vertex(i).all_neighbours():
                j = int(j)
                if j not in infected and random.random() <= p:
                    infected.add(j)
                    infection_times[j] = time
                    edges.append((i, j))
        time += 1
    efilt = g.new_edge_property('bool')
    efilt.a = False
    for u, v in edges:
        efilt[g.edge(g.vertex(u), g.vertex(v))] = True
        
    return source, infection_times, GraphView(g, directed=True, efilt=efilt)
