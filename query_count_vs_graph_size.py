
# coding: utf-8

# In[14]:


import pandas as pd
from mwu import (MAX_MU, RANDOM)
from synthetic_data import (load_data_by_gtype, all_graph_types)
from experiment_utils import (experiment_mwu_multiple_rounds,
                              experiment_dog_multiple_rounds, counts_to_stat)


def main(n_rounds, gtype, size_params,
         fraction, epsilon, sampling_method,
         check_neighbor_threshold):
    rows = []
    indices = []
    for size_param in size_params:
        print(size_param)
        try:
            g, time_probas, node2id, id2node = load_data_by_gtype(gtype, size_param)
        except IOError:
            print('fail to load {}/{}'.format(gtype, size_param))
            break

        def mwu_wrapper(method):
            return experiment_mwu_multiple_rounds(n_rounds, g, node2id, id2node, time_probas,
                                                  fraction=fraction, epsilon=epsilon,
                                                  sampling_method=sampling_method,
                                                  query_selection_method=method,
                                                  check_neighbor_threshold=check_neighbor_threshold,
                                                  max_iter=g.number_of_nodes())
                
        counts = mwu_wrapper(MAX_MU)
        rows.append(counts_to_stat(counts))
        indices.append((MAX_MU, g.number_of_nodes()))

        counts = mwu_wrapper(RANDOM)
        rows.append(counts_to_stat(counts))
        indices.append((RANDOM, g.number_of_nodes()))
        
        counts = experiment_dog_multiple_rounds(n_rounds, g, fraction, sampling_method)
        rows.append(counts_to_stat(counts))
        indices.append(('dog', g.number_of_nodes()))

        index = pd.MultiIndex.from_tuples(indices, names=('method', 'graph size'))
        df = pd.DataFrame.from_records(rows, index=index)

        df.to_pickle('data/{}/performance.pkl'.format(gtype))
        print('saving snapshot done')


if __name__ == '__main__':
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

    parser.add_argument('-n', '--n_rounds', default=100, type=int)
    parser.add_argument('-t', '--gtype', choices=all_graph_types, required=True)
    parser.add_argument('-b', '--base', type=int, default=2)
    parser.add_argument('--emin', type=int, default=2)
    parser.add_argument('--emax', type=int, default=10)

    args = parser.parse_args()
    size_params = ['{}-{}'.format(args.base, e) for e in range(args.emin, args.emax+1)]
    
    main(args.n_rounds,
         args.gtype,
         size_params,
         args.fraction,
         args.epsilon,
         args.sampling_method,
         args.nbr_threshold)
