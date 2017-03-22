# coding: utf-8
import os
import numpy as np
import arrow
import pandas as pd
import networkx as nx
from mwu import (MAX_MU, RANDOM, RAND_MAX_MU)
from edge_mwu import MEDIAN_NODE
from tree_binary_search import find_source as find_source_binary_search
from synthetic_data import (load_data_by_gtype, all_graph_types)
from experiment_utils import (experiment_node_mwu_multiple_rounds,
                              experiment_edge_mwu_multiple_rounds,
                              experiment_multiple_rounds,
                              experiment_dog_multiple_rounds, counts_to_stat)


def main(query_methods, n_rounds, gtype, size_params,
         fraction, epsilon, sampling_method,
         check_neighbor_threshold):
    rows = []
    indices = []
    for size_param in size_params:
        print(size_param)
        try:
            g, time_probas, dir_tbl, inf_tbl, sp_len, node2id, id2node = load_data_by_gtype(gtype, size_param)
            for s, m in time_probas.items():
                time_probas[s] = m.tolil()
        except IOError:
            print('fail to load {}/{}'.format(gtype, size_param))
            break

        def node_mwu_wrapper(method):
            return experiment_node_mwu_multiple_rounds(
                n_rounds, g, node2id, id2node, time_probas,
                fraction=fraction, epsilon=epsilon,
                sampling_method=sampling_method,
                query_selection_method=method,
                check_neighbor_threshold=check_neighbor_threshold,
                max_iter=g.number_of_nodes())

        def edge_mwu_wrapper(method):
            assert dir_tbl is not None
            assert inf_tbl is not None
            return experiment_edge_mwu_multiple_rounds(
                g, method,
                dir_tbl, inf_tbl,
                sp_len,
                check_neighbor_threshold=check_neighbor_threshold,
                fraction=fraction,
                sampling_method=sampling_method,
                rounds=n_rounds,
                max_iter=g.number_of_nodes())
        
        if MAX_MU in query_methods:
            print(MAX_MU)
            counts = node_mwu_wrapper(MAX_MU)
            rows.append(counts_to_stat(counts))
            indices.append((MAX_MU, g.number_of_nodes()))

        if RAND_MAX_MU in query_methods:
            print(RAND_MAX_MU)
            counts = node_mwu_wrapper(RAND_MAX_MU)
            rows.append(counts_to_stat(counts))
            indices.append((RAND_MAX_MU, g.number_of_nodes()))

        if RANDOM in query_methods:
            print(RANDOM)
            counts = node_mwu_wrapper(RANDOM)
            rows.append(counts_to_stat(counts))
            indices.append((RANDOM, g.number_of_nodes()))

        if 'dog' in query_methods:
            print('dog')
            for f in np.linspace(0, 1, 5):
                counts = experiment_dog_multiple_rounds(n_rounds, g, fraction, sampling_method,
                                                        query_fraction=f)
                rows.append(counts_to_stat(counts))
                indices.append(('dog-{:.2f}'.format(f), g.number_of_nodes()))

        if MEDIAN_NODE in query_methods:
            print(MEDIAN_NODE)
            counts = edge_mwu_wrapper(MEDIAN_NODE)
            rows.append(counts_to_stat(counts))
            indices.append((MEDIAN_NODE, g.number_of_nodes()))

        if 'binary_search' in query_methods:
            if not nx.is_tree(g):
                raise TypeError('g is not a tree')
            
            counts = experiment_multiple_rounds(
                find_source_binary_search,
                n_rounds, g,
                fraction,
                sampling_method)
            rows.append(counts_to_stat(counts))
            indices.append(('binary_search', g.number_of_nodes()))
            
        index = pd.MultiIndex.from_tuples(indices, names=('method', 'graph size'))
        df = pd.DataFrame.from_records(rows, index=index)

        path = 'data/{}/performance.pkl'.format(gtype)
        if os.path.exists(path):
            utc = arrow.utcnow()
            date = utc.to('Europe/Helsinki').format('YYYY-MM-DD')
            path = 'data/{}/performance-{}.pkl'.format(gtype, date)
        df.to_pickle(path)
        print('saving snapshot to {} done'.format(path))


if __name__ == '__main__':
    ALL_METHODS = (MAX_MU, RAND_MAX_MU, RANDOM, MEDIAN_NODE, 'dog', 'binary_search')

    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--fraction',
                        type=float, default=0.01)
    parser.add_argument('-e', '--epsilon',
                        type=float, default=0.7)
    parser.add_argument('--nbr_threshold',
                        type=float, default=0.01)
    parser.add_argument('-s', '--sampling_method',
                        choices=('late_nodes', 'uniform'),
                        default='late_nodes')
    parser.add_argument('query_methods', metavar='Q', nargs='+', type=str,
                        choices=ALL_METHODS)

    parser.add_argument('-n', '--n_rounds', default=100, type=int)
    parser.add_argument('-t', '--gtype', choices=all_graph_types, required=True)
    parser.add_argument('-b', '--base', type=int, default=2)
    parser.add_argument('--emin', type=int, default=2)
    parser.add_argument('--emax', type=int, default=10)

    args = parser.parse_args()
    size_params = ['{}-{}'.format(args.base, e) for e in range(args.emin, args.emax+1)]
    print(args.query_methods)
    main(args.query_methods,
         args.n_rounds,
         args.gtype,
         size_params,
         args.fraction,
         args.epsilon,
         args.sampling_method,
         args.nbr_threshold)
