import math
import networkx as nx
import numpy as np
import random

from graph_tool.all import shortest_path, Graph, GraphView

from utils import get_infection_time, MAXINT
from cascade import gen_nontrivial_cascade




def sample_graph_from_infection(g):
    rands = np.random.rand(g.number_of_edges())
    active_edges = [(u, v) for (u, v), r in zip(g.edges_iter(), rands) if g[u][v]['p'] >= r]
    induced_g = nx.Graph()
    induced_g.add_nodes_from(g.nodes())
    induced_g.add_edges_from(active_edges)
    for u, v in induced_g.edges_iter():
        induced_g[u][v]['d'] = g[u][v]['d']
    return induced_g


def make_full_cascade(g, source=None, is_sampled=False):
    """
    """
    if source is None:
        idx = np.arange(g.number_of_nodes())
        source = g.nodes()[np.random.choice(idx)]

    if not is_sampled:
        induced_g = sample_graph_from_infection(g)
    else:
        induced_g = g
        
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


def make_partial_cascade(g, fraction, sampling_method='uniform'):
    """simulate one IC cascade and return the source, infection times and infection tree"""
    tree = None  # compatibility reason
    infection_times = make_full_cascade(g)

    infected_nodes = [n for n in g.nodes_iter() if not np.isinf(infection_times[n])]
    cascade_size = len(infected_nodes)

    sample_size = math.ceil(cascade_size * fraction)
    
    if sampling_method == 'uniform':
        idx = np.arange(len(infected_nodes))
        sub_idx = np.random.choice(idx, sample_size, replace=False)
        obs_nodes = set([infected_nodes[i] for i in sub_idx])
    elif sampling_method == 'late_nodes':
        obs_nodes = set(sorted(infected_nodes, key=lambda n: -infection_times[n])[:sample_size])
    else:
        raise ValueError('unknown sampling methods')

    assert len(obs_nodes) > 0
    source = min(infection_times, key=lambda n: infection_times[n])

    return source, obs_nodes, infection_times, tree


def get_gvs(g, p, K):
    """g: graph_tool.Graph
    """
    rands2d = np.random.random((K, g.num_edges()))
    edge_masks2d = (rands2d <= p)
    
    gvs = []
    for i in range(K):
        p = g.new_edge_property('bool')
        p.set_2d_array(edge_masks2d[i, :])
        gvs.append(GraphView(g, efilt=p))
    return gvs


def sample_graph_by_p(g, p):
    """
    graph_tool version of sampling a graph
    mask the edge according to probability p and return the masked graph"""
    flags = (np.random.random(g.num_edges()) <= p)
    p = g.new_edge_property('bool')
    p.set_2d_array(flags)
    return GraphView(g, efilt=p)


def simulate_cascade(g, p, source=None, return_tree=False):
    """
    graph_tool version of simulating cascade
    return np.ndarray on vertices as the infection time in cascade
    uninfected node has dist -1
    """
    if source is None:
        source = random.choice(np.arange(g.num_vertices(), dtype=int))
    gv = sample_graph_by_p(g, p)

    times = get_infection_time(gv, source)
    if return_tree:
        all_edges = set()
        for target in np.nonzero(times != -1)[0]:
            path = shortest_path(gv, source=source, target=gv.vertex(target))[0]
            edges = set(zip(path[:-1], path[1:]))
            all_edges |= edges
        tree = Graph(directed=True)
        for _ in range(g.num_vertices()):
            tree.add_vertex()
        for u, v in all_edges:
            tree.add_edge(int(u), int(v))
        return source, times, tree
    else:
        return source, times
