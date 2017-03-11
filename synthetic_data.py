
import argparse
import pickle as pkl
import networkx as nx
from networkx.generators.random_graphs import random_powerlaw_tree
from graph_generator import kronecker_random_graph, grid_2d, \
    P_peri ,P_hier, P_rand, add_p_and_delta
from ic import infection_time_estimation

KRONECKER_RAND = 'kr-rand'
KRONECKER_PERI = 'kr-peri'
KRONECKER_HIER = 'kr-hier'
GRID = 'grid'
PL_TREE = 'pl-tree'
ER = 'er'
BARABASI = 'barabasi'
CLIQUE = 'clique'
all_graph_types = [KRONECKER_RAND,
                   KRONECKER_PERI,
                   KRONECKER_HIER,
                   GRID,
                   PL_TREE,
                   ER,
                   BARABASI,
                   CLIQUE]


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


def load_data_by_gtype(gtype, size_param_str):
    g = nx.read_gpickle('data/{}/{}/graph.gpkl'.format(gtype, size_param_str))
    time_probas = pkl.load(open('data/{}/{}/{}.pkl'.format(gtype, size_param_str,
                                                           INF_TIME_PROBA_FILE), 'rb'))
    node2id = pkl.load(open('data/{}/{}/{}.pkl'.format(gtype, size_param_str,
                                                       NODE2ID_FILE), 'rb'))
    id2node = pkl.load(open('data/{}/{}/{}.pkl'.format(gtype, size_param_str,
                                                       ID2NODE_FILE), 'rb'))
    return g, time_probas, node2id, id2node

if __name__ == "__main__":
    import os
    
    p = 0.7
    delta = 1
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', choices=all_graph_types,
                        help='graph type')
    parser.add_argument('-s', '--size', type=int,
                        default=0,
                        help="size of graph")
    parser.add_argument('-e', '--size_exponent', type=int,
                        default=1,
                        help="exponent of the size")
    parser.add_argument('-b', '--exponent_base', type=int,
                        default=10,
                        help="base of the size exponent")
    parser.add_argument('-n', '--n_rounds', type=int,
                        default=100,
                        help="number of simulated cascades")

    args = parser.parse_args()
    gtype = args.type
    if args.size:
        size = args.size
        output_dir = 'data/{}/{}'.format(gtype, size)
    else:
        size = args.exponent_base ** args.size_exponent
        output_dir = 'data/{}/{}-{}'.format(gtype, args.exponent_base,
                                            args.size_exponent)
    if gtype == KRONECKER_HIER:
        g = gen_kronecker(P=P_hier, k=args.size_exponent, n_edges=2**args.size_exponent * 3)
    elif gtype == KRONECKER_PERI:
        g = gen_kronecker(P=P_peri, k=args.size_exponent, n_edges=2**args.size_exponent * 3)
    elif gtype == KRONECKER_RAND:
        g = gen_kronecker(P=P_rand, k=args.size_exponent, n_edges=2**args.size_exponent * 3)
    elif gtype == PL_TREE:
        p = 0.88
        g = random_powerlaw_tree(size, tries=999999)
    elif gtype == ER:
        g = extract_larges_CC(nx.fast_gnp_random_graph(size, 0.2))
    elif gtype == BARABASI:
        g = extract_larges_CC(nx.barabasi_albert_graph(size, 5))
    elif gtype == GRID:
        g = grid_2d(size)
    elif gtype == CLIQUE:
        g = nx.complete_graph(size)
    else:
        raise ValueError('unsupported graph type {}'.format(gtype))

    g.remove_edges_from(g.selfloop_edges())
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print('graph type: {}'.format(gtype))
    g = add_p_and_delta(g, p, delta)

    time_probas, node2id = infection_time_estimation(g, args.n_rounds,
                                                     return_node2id=True)
    
    nx.write_gpickle(g, '{}/graph.gpkl'.format(output_dir, gtype))

    pkl.dump(time_probas,
             open('{}/{}.pkl'.format(output_dir, INF_TIME_PROBA_FILE), 'wb'))

    pkl.dump(node2id,
             open('{}/{}.pkl'.format(output_dir, NODE2ID_FILE), 'wb'))
    id2node = {i: n for n, i in node2id.items()}
    pkl.dump(id2node,
             open('{}/{}.pkl'.format(output_dir, ID2NODE_FILE), 'wb'))
