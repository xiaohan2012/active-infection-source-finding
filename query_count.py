# coding: utf-8
import os
from graph_tool.all import load_graph
from mwu import (MAX_MU, RANDOM, RAND_MAX_MU)
from noisy_binary_search import NOISY_BINARY_SEARCH
from experiment_utils import experiment_mwu_n_rounds, \
    experiment_noisy_bs_n_rounds, experiment_dog_n_rounds, \
    experiment_multiple_rounds, \
    counts_to_stat


def main(active_method,
         n_rounds,
         gtype, size_param,
         p, q, epsilon, sampling_method,
         mwu_reward_method='dist',
         dog_fraction=0.0,
         debug=False):
    g = load_graph('data/{}/{}/graph.gt'.format(gtype, size_param))

    n_nodes = g.num_vertices()
    print('|V| = {}'.format(n_nodes))

    def mwu_wrapper(active_method, reward_method):
        return experiment_mwu_n_rounds(
            n_rounds,
            g,
            p, q, epsilon,
            sampling_method,
            active_method,
            reward_method,
            seed=None)

    if active_method == MAX_MU:
        counts = mwu_wrapper(MAX_MU, mwu_reward_method)
        method_name = '{}-{}'.format(active_method, mwu_reward_method)
    elif active_method == RANDOM:
        counts = mwu_wrapper(RANDOM, mwu_reward_method)
        method_name = '{}-{}'.format(active_method, mwu_reward_method)
    elif active_method == 'dog':
        counts = experiment_dog_n_rounds(
            n_rounds, g, q, sampling_method,
            query_fraction=dog_fraction)
        method_name = 'dog-{:.1f}'.format(dog_fraction)
    elif active_method == NOISY_BINARY_SEARCH:
        counts = experiment_noisy_bs_n_rounds(
            g, sp_len,
            n_rounds,
            consistency_multiplier=0.9)
        method_name = NOISY_BINARY_SEARCH
    else:
        raise ValueError('unknown methoder')
            
    stat = counts_to_stat(counts)
    dirname = 'outputs/query_count/{}/{}'.format(method_name, gtype)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if not debug:
        stat.to_pickle('{}/{}.pkl'.format(dirname, size_param))
    else:
        print(stat)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--n_rounds', default=100, type=int)
    parser.add_argument('-p', '--p',
                        type=float, default=0.7)
    parser.add_argument('-q', '--q',
                        type=float, default=0.1)
    parser.add_argument('-e', '--epsilon',
                        type=float, default=0.2)
    parser.add_argument('-s', '--sampling_method',
                        default='uniform')
    parser.add_argument('-m', '--active_method', required=True)

    parser.add_argument('-t', '--gtype', required=True)
    parser.add_argument('--size_param', required=True)

    parser.add_argument('--mwu_reward_method', default='dist')
    parser.add_argument('--dog_fraction', default=0.0, type=float)

    parser.add_argument('-d', '--debug', action='store_true')
    
    args = parser.parse_args()
    main(args.active_method,
         args.n_rounds,
         args.gtype, args.size_param,
         args.p, args.q,
         args.epsilon, args.sampling_method,
         args.mwu_reward_method,
         args.dog_fraction,
         debug=args.debug)
