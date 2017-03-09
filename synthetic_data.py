import pickle as pkl
import networkx as nx
from networkx.generators.random_graphs import random_powerlaw_tree
from graph_generator import kronecker_random_graph, grid_2d, \
    P_peri ,P_hier, P_rand, add_p_and_delta
from ic import infection_time_estimation

KRONECKER_RAND = 'kronecker-rand'
KRONECKER_PERI = 'kronecker-peri'
KRONECKER_HIER = 'kronecker-hier'
GRID = 'grid'
PL_TREE = 'powerlaw_tree'
ER = 'erdos-renyi'
BARABASI = 'barabasi'
CLIQUE = 'clique'


INF_TIME_PROBA_FILE = 'inf_time_proba_matrix'
NODE2ID_FILE = 'node2id'
ID2NODE_FILE = 'id2node'

TIMES_FILE_SUFFIX = 'source2times'


def extract_larges_CC(g):
    nodes = max(nx.connected_components(g), key=len)
    return g.subgraph(nodes)


def gen_kronecker(P, k=8, n_edges=512):
    g = kronecker_random_graph(k, P, n_edges=n_edges, directed=False)
    return extract_larges_CC(g)


def load_data_by_gtype(gtype):
    g = nx.read_gpickle('data/{}/graph.gpkl'.format(gtype))
    time_probas = pkl.load(open('data/{}/{}.pkl'.format(gtype, INF_TIME_PROBA_FILE), 'rb'))
    node2id = pkl.load(open('data/{}/{}.pkl'.format(gtype, NODE2ID_FILE), 'rb'))
    id2node = pkl.load(open('data/{}/{}.pkl'.format(gtype, ID2NODE_FILE), 'rb'))
    return g, time_probas, node2id, id2node

if __name__ == "__main__":
    import os
    
    p = 0.7
    delta = 1
    
    gtype = GRID

    if gtype == KRONECKER_HIER:
        g = gen_kronecker(P=P_hier, k=10, n_edges=2048)
    elif gtype == KRONECKER_PERI:
        g = gen_kronecker(P=P_peri, k=10, n_edges=2048)
    elif gtype == KRONECKER_RAND:
        g = gen_kronecker(P=P_rand, k=10, n_edges=2048)
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

    directory = 'data/{}'.format(gtype)
    if not os.path.exists(directory):
        os.makedirs(directory)

    print('graph type: {}'.format(gtype))
    g = add_p_and_delta(g, p, delta)
    nx.write_gpickle(g, 'data/{}/graph.gpkl'.format(gtype))

    n_rounds = 100
    time_probas, node2id = infection_time_estimation(g, n_rounds,
                                                     return_node2id=True)
    pkl.dump(time_probas,
             open('data/{}/{}.pkl'.format(gtype, INF_TIME_PROBA_FILE), 'wb'))

    pkl.dump(node2id,
             open('data/{}/{}.pkl'.format(gtype, NODE2ID_FILE), 'wb'))
    id2node = {i: n for n, i in node2id.items()}
    pkl.dump(id2node,
             open('data/{}/{}.pkl'.format(gtype, ID2NODE_FILE), 'wb'))
