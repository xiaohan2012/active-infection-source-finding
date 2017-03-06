import math
import networkx as nx
import numpy as np
from cascade import generate_cascade


# @Deprecated("function overlap with cascade.generate_cascade")
def sample_graph_from_infection(g):
    g_bone = g
    g = nx.Graph()
    g.add_nodes_from(g_bone.nodes())
    
    # sample active edges
    rands = np.random.random(g_bone.number_of_edges())
    
    probas = np.array([g_bone[i][j]['p'] for i, j in g_bone.edges_iter()])
    all_edges = g_bone.edges()
    active_edges = [all_edges[i] for i in (rands < probas).nonzero()[0]]

    g.add_edges_from(active_edges)
    
    for i, j in g.edges_iter():
        g[i][j]['tried'] = False
        g[i][j]['d'] = g_bone[i][j]['d']
        g[i][j]['p'] = g_bone[i][j]['p']
        
    return g


# In[4]:
# @Deprecated("function overlap with `cascade.generate_cascade`")
def simulate_IC(g, s=None, is_g_sampled=False, debug=False):
    """return dict of node to infected times
    edges in g should contain the follow information:

    - d: the transmission delay
    - p: infection probability
    """
    infection_time = {n: float('inf') for n in g.nodes_iter()}
    bp = {n: None for n in g.nodes_iter()}  # back tracker

    if not is_g_sampled:
        g = sample_graph_from_infection(g)
    else:
        g = g.copy()
    
    if debug:
        print(g.edges())
    if s is None:
        idx = np.arange(g.number_of_nodes())
        s = g.nodes()[np.random.choice(idx)]
        
    queue = [s]
    t = 0
    infection_time[s] = t
    while len(queue) > 0:
        outbreak_nodes = [u for u in queue if infection_time[u] <= t]
        queue = list(set(queue) - set(outbreak_nodes))
        infected_nodes = set()
        if debug:
            print('outbreak_nodes {}'.format(outbreak_nodes))
        for u in outbreak_nodes:
            for v in g.neighbors(u):
                if not g[u][v]['tried']:
                    if debug:
                        print('{} infects {}'.format(u, v))
                    g[u][v]['tried'] = True
                    if np.isinf(infection_time[v]):  # not infected yet
                        queue.append(v)

                    # update infection time
                    if (infection_time[u] + g[u][v]['d'] < infection_time[v]):
                        infection_time[v] = infection_time[u] + g[u][v]['d']
                        infected_nodes.add(v)
                        bp[v] = u
        if len(infected_nodes) == 0:
            break
        n = min(infected_nodes, key=lambda n: infection_time[n])
        t = infection_time[n]
    infection_tree = nx.Graph()
    infection_tree.add_nodes_from(g.nodes())
    del bp[s]
    infection_tree.add_edges_from(bp.items())
    return infection_time, infection_tree


def make_input(g, fraction, sampling_method='uniform'):
    """simulate one IC cascade and return the source, infection times and infection tree"""
    while True:
        infection_times = generate_cascade(g)
        tree = None  # compatibility reason

        cascade_size = np.count_nonzero(np.invert(np.isinf(list(infection_times.values()))))

        sample_size = math.ceil(cascade_size * fraction)
        infected_nodes = [n for n in g.nodes_iter() if not np.isinf(infection_times[n])]
        
        if len(infected_nodes) > sample_size:
            break
        
    if sampling_method == 'uniform':
        idx = np.arange(len(infected_nodes))
        sub_idx = np.random.choice(idx, sample_size)
        obs_nodes = set([infected_nodes[i] for i in sub_idx])
    elif sampling_method == 'late_nodes':
        obs_nodes = set(sorted(infected_nodes, key=lambda n: -infection_times[n])[:sample_size])
    else:
        raise ValueError('unknown sampling methods')

    assert len(obs_nodes) > 0
    source = min(infection_times, key=lambda n: infection_times[n])

    return source, obs_nodes, infection_times, tree
