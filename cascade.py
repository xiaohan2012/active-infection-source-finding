import networkx as nx
import random
import numpy as np


def generate_cascade_old(g):
    """shit: just use Dijkstra!!
    """
    g = g.copy()
    source = random.choice(g.nodes())
    infected = {source}
    infected_times = {source: 0}
    iter_n = 0
    while True:
        iter_n += 1
        newly_infected = set()
        for u in infected:
            can_continue = False
            for v in g.neighbors(u):
                if v not in infected and not g[u][v].get('attempted', False):
                    # print('infected node: {}'.format(v))
                    can_continue = True
                    p_uv = 0.5  # proba of getting infected
                    if random.random() < p_uv:
                        newly_infected.add(v)
                        infected_times[v] = iter_n
                    g[u][v]['attempted'] = True
        infected |= newly_infected
        if not can_continue:
            break
    return infected_times



def generate_cascade(g):
    source = random.choice(g.nodes())

    rands = np.random.rand(g.number_of_edges())
    active_edges = [(u, v) for (u, v), r in zip(g.edges_iter(), rands) if g[u][v]['p'] >= r]
    induced_g = nx.Graph()
    induced_g.add_edges_from(active_edges)
    for u, v in induced_g.edges_iter():
        induced_g[u][v]['d'] = g[u][v]['d']
        
    if not induced_g.has_node(source):
        infection_times = {n: float('inf') for n in g.nodes_iter()}
        infection_times[source] = 0
    else:
        infection_times = nx.shortest_path_length(induced_g, source=source, weight='d')
        for n in g.nodes_iter():
            if n not in infection_times:
                infection_times[n] = float('inf')
    assert infection_times[source] == 0
    assert len(infection_times) == g.number_of_nodes()
    return infection_times
