import math
import networkx as nx
import numpy as np
from collections import defaultdict, Counter
from joblib import Parallel, delayed
from scipy.sparse import csr_matrix
from tqdm import tqdm


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


# @profile
def infection_time_estimation(g, n_rounds, return_node2id=False):
    """
    estimate the infection time distribution
    n_rounds of cascades for *each* node

    Returns:
    dict of source to 2D sparse matrices (node to infection time probabilities)
        can be viewed as 3D tensor:
        N x N x max(t), source to node to infection time
        dict source to nodes' infection time distribution
    """
    node2id = {n: i for i, n in enumerate(g.nodes_iter())}
    s2n_times_counter = defaultdict(lambda: defaultdict(Counter))
    inf = float('inf')
    # run in serial to save memory
    for i in tqdm(range(n_rounds)):
        sampled_g = sample_graph_from_infection(g)
        # this is using Dijkstra
        # s2t_len = nx.shortest_path_length(sampled_g, weight='d')

        # this is serial BFS
        s2t_len = nx.shortest_path_length(sampled_g)

        # this is parallel BFS (slow)
        # results = Parallel(n_jobs=-1)(
        #     delayed(nx.single_source_shortest_path_length)(g, n)
        #     for n in g.nodes_iter())
        # s2t_len = {n: t_len for n, t_len in zip(g.nodes_iter(), results)}
        
        for s in s2t_len:  # one cascade for each node
            for n in sampled_g.nodes_iter():
                time = s2t_len[s].get(n, inf)
                s2n_times_counter[s][n][time] += 1

    all_times = np.array([t
                          for n2times_counter in s2n_times_counter.values()
                          for counter in n2times_counter.values()
                          for t in counter.keys()])
    all_times = np.ravel(all_times)
    unique_values = np.unique(all_times)

    min_val, max_val = (int(unique_values.min()),
                        int(unique_values[np.invert(np.isinf(unique_values))].max()))
    n_times = max_val - min_val + 2
    d = dict()
    for s in tqdm(g.nodes_iter()):
        i = node2id[s]
        row = []  # infected node
        col = []  # infection time
        data = []  # probabilities
        for n in g.nodes_iter():
            j = node2id[n]
            cnts = s2n_times_counter[s][n]
            cnts[n_times-1] = cnts[float('inf')]
            del cnts[float('inf')]
            row += [j] * len(cnts)
            col_slice, cnts_list = map(list, zip(*cnts.items()))
            col += col_slice
            cnts_array = np.array(cnts_list, dtype=np.float)
            cnts_array /= cnts_array.sum()
            data += cnts_array.tolist()

        d[i] = csr_matrix((data, (row, col)), shape=(g.number_of_nodes(), n_times))
    if return_node2id:
        return d, node2id
    else:
        return d
