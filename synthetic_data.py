import numpy as np
import pickle as pkl
from collections import defaultdict, Counter
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from graph_generator import kronecker_random_graph, grid_2d, \
    P_peri ,P_hier, P_rand, add_p_and_delta
from core import generate_sufficient_stats

KRONECKER_RAND = 'kronecker-rand'
KRONECKER_PERI = 'kronecker-peri'
KRONECKER_HIER = 'kronecker-hier'
GRID = 'grid'

COUNTER_FILE_SUFFIX = 'source2nodeid_counter'
TIMES_FILE_SUFFIX = 'source2times'


def gen_kronecker(P, k=8, n_edges=512):
    g = kronecker_random_graph(k, P, n_edges=n_edges, directed=False)
    nodes = max(nx.connected_components(g), key=len)
    g = g.subgraph(nodes)
    return g


def load_data_by_gtype(gtype):
    source2nodeid_counter = pkl.load(open('outputs/{}_{}.pkl'.format(gtype, COUNTER_FILE_SUFFIX), 'rb'))
    times_by_source = pkl.load(open('outputs/{}_{}.pkl'.format(gtype, TIMES_FILE_SUFFIX), 'rb'))
    g = nx.read_gpickle('outputs/{}.gpkl'.format(gtype))
    return g, times_by_source, source2nodeid_counter


if __name__ == "__main__":    
    p = 0.7
    delta = 1
    
    gtype = GRID

    if gtype == KRONECKER_HIER:
        g = gen_kronecker(P=P_hier)
    elif gtype == KRONECKER_PERI:
        g = gen_kronecker(P=P_peri)
    elif gtype == KRONECKER_RAND:
        g = gen_kronecker(P=P_rand)
    elif gtype == GRID:
        g = grid_2d(100)
    else:
        raise ValueError('unsupported graph type {}'.format(gtype))

    print('graph type: {}'.format(gtype))
    g = add_p_and_delta(g, p, delta)
    nx.write_gpickle(g, 'outputs/{}.gpkl'.format(gtype))


    cascade_number = 250
    print('generating {} cascades'.format(cascade_number))
    # list of list of (source, times, tree)
    stats = Parallel(n_jobs=-1)(delayed(generate_sufficient_stats)(g)
                                for i in range(cascade_number))

    print('generating times_by_source')
    # dict: node as source -> 2d matrix (infection time, node)
    times_by_source = defaultdict(list)
    for stat in stats:
        for s, times, _ in stat:
            times_array = np.array([times[n] for n in g.nodes_iter()])
            times_by_source[s].append(times_array)
            times_by_source = {s: np.array(times2d)
                               for s, times2d in times_by_source.items()}

    pkl.dump(times_by_source,
             open('outputs/{}_{}.pkl'.format(gtype, TIMES_FILE_SUFFIX), 'wb'))

    print('generating source2nodeid_counter')
    # This "cache" file is mainly for better running performance
    source2nodeid_counter = defaultdict(dict)
    N = g.number_of_nodes()
    for src, times in tqdm(times_by_source.items()):
        for i in range(N):
            source2nodeid_counter[src][i] = Counter(times[:, i])
    
    pkl.dump(source2nodeid_counter,
             open('outputs/{}_{}.pkl'.format(gtype, COUNTER_FILE_SUFFIX), 'wb'))
