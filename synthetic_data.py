import numpy as np
import pickle as pkl
from collections import defaultdict, Counter
import networkx as nx
from networkx.generators.random_graphs import random_powerlaw_tree
from tqdm import tqdm
from joblib import Parallel, delayed
from graph_generator import kronecker_random_graph, grid_2d, \
    P_peri ,P_hier, P_rand, add_p_and_delta
from core import generate_sufficient_stats

KRONECKER_RAND = 'kronecker-rand'
KRONECKER_PERI = 'kronecker-peri'
KRONECKER_HIER = 'kronecker-hier'
GRID = 'grid'
PL_TREE = 'powerlaw_tree'
ER = 'erdos-renyi'
BARABASI = 'barabasi'
CLIQUE = 'clique'

COUNTER_FILE_SUFFIX = 'source2nodeid_counter'
TIMES_FILE_SUFFIX = 'source2times'
TIMES_MEAN_SUFFIX = 'source2time_mean'


def extract_larges_CC(g):
    nodes = max(nx.connected_components(g), key=len)
    return g.subgraph(nodes)


def gen_kronecker(P, k=8, n_edges=512):
    g = kronecker_random_graph(k, P, n_edges=n_edges, directed=False)
    return extract_larges_CC(g)


def load_data_by_gtype(gtype):
    source2nodeid_counter = pkl.load(open('data/{}/{}.pkl'.format(gtype, COUNTER_FILE_SUFFIX), 'rb'))
    times_by_source = pkl.load(open('data/{}/{}.pkl'.format(gtype, TIMES_FILE_SUFFIX), 'rb'))
    g = nx.read_gpickle('data/{}/graph.gpkl'.format(gtype))
    return g, times_by_source, source2nodeid_counter


if __name__ == "__main__":
    import os
    
    p = 0.7
    delta = 1
    
    gtype = BARABASI

    if gtype == KRONECKER_HIER:
        g = gen_kronecker(P=P_hier)
    elif gtype == KRONECKER_PERI:
        g = gen_kronecker(P=P_peri)
    elif gtype == KRONECKER_RAND:
        g = gen_kronecker(P=P_rand)
    elif gtype == PL_TREE:
        p = 0.88
        g = random_powerlaw_tree(100, tries=10000)
    elif gtype == ER:
        g = extract_larges_CC(nx.fast_gnp_random_graph(100, 0.2))
    elif gtype == BARABASI:
        g = extract_larges_CC(nx.barabasi_albert_graph(100, 5))
    elif gtype == GRID:
        g = grid_2d(10)
    elif gtype == CLIQUE:
        g = nx.complete_graph(10)
    else:
        raise ValueError('unsupported graph type {}'.format(gtype))

    g.remove_edges_from(g.selfloop_edges())
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    node2id = {n: i for i, n in enumerate(g.nodes_iter())}

    directory = 'data/{}'.format(gtype)
    if not os.path.exists(directory):
        os.makedirs(directory)

    print('graph type: {}'.format(gtype))
    g = add_p_and_delta(g, p, delta)
    nx.write_gpickle(g, 'data/{}/graph.gpkl'.format(gtype))

    cascade_number = 250
    print('generating {} cascades'.format(cascade_number))
    # list of list of (source, times, tree)
    stats = Parallel(n_jobs=-1)(delayed(generate_sufficient_stats)(g)
                                for i in range(cascade_number))

    print('generating times_by_source')
    # dict: node as source -> 2d matrix (infection time, node), K by N
    times_by_source = defaultdict(list)
    for stat in stats:
        for s, times, _ in stat:
            times_array = np.array([times[n] for n in g.nodes_iter()])
            times_by_source[s].append(times_array)
    times_by_source = {s: np.array(times2d)
                       for s, times2d in times_by_source.items()}

    pkl.dump(times_by_source,
             open('data/{}/{}.pkl'.format(gtype, TIMES_FILE_SUFFIX), 'wb'))

    print('generating {}'.format(TIMES_MEAN_SUFFIX))
    # N x N matrix
    # the ith row: using node i as the source, the mean of infection times for N nodes
    B = np.zeros((g.number_of_nodes(), g.number_of_nodes()))
    for source, times in times_by_source.items():
        s = node2id[source]
        means = np.mean(times, axis=0)
        assert len(means) == g.number_of_nodes()
        B[s, :] = means

    pkl.dump(B,
             open('data/{}/{}.pkl'.format(gtype, TIMES_MEAN_SUFFIX), 'wb'))
    
    print('generating source2nodeid_counter')
    # This "cache" file is mainly for better running performance
    source2nodeid_counter = defaultdict(dict)
    N = g.number_of_nodes()
    for src, times in tqdm(times_by_source.items()):
        for i in range(N):
            source2nodeid_counter[src][i] = Counter(times[:, i])
    
    pkl.dump(source2nodeid_counter,
             open('data/{}/{}.pkl'.format(gtype, COUNTER_FILE_SUFFIX), 'wb'))
