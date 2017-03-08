import math
import networkx as nx
import numpy as np
from collections import defaultdict
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import hmean


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
    while True:
        infection_times = make_full_cascade(g)
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


def infection_time_estimation(g, n_rounds, mean_method='harmonic'):
    """
    estimate the harmonic mean of infection times given each node as source

    Returns:
    dict source to nodes' infection time:
    for each node as source, return the estimated infection times of all nodes
    """
    sampled_graphs = [sample_graph_from_infection(g)
                      for i in range(n_rounds)]
    s2t_len_list = Parallel(n_jobs=-1)(
        delayed(nx.shortest_path_length)(g, weight='d')
        for g in sampled_graphs)
    # 3D array
    s2n_times = defaultdict(lambda: defaultdict(list))

    for g, s2t_len in tqdm(zip(sampled_graphs, s2t_len_list)):
        for s in s2t_len:
            for n in g.nodes_iter():
                s2n_times[s][n].append(s2t_len[s].get(n, float('inf')))

    if mean_method == 'harmonic':
        def mean_func(times):
            times = np.array(times)
            times = times[np.nonzero(times)]
            if times.shape[0] >	0:
                return hmean(times)
            else:  # all zeros
                return 0
    elif mean_method == 'arithmetic':
        all_times = np.array([times
                              for n2times in s2n_times.values()
                              for times in n2times.values()])
        inf_value = all_times.max() + 1

        def mean_func(times):
            times = np.array(times)
            times[np.isinf(times)] = inf_value
            return times.mean()
            
    else:
        raise ValueError('{"harmoic", "arithmetic"} accepted')

    est = defaultdict(dict)
    for s, n2times in tqdm(s2n_times.items()):
        for n, times in n2times.items():
            est[s][n] = mean_func(times)
    return est, s2n_times
