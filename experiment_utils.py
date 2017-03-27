import numpy as np
import pandas as pd
import random
import networkx as nx
from tqdm import tqdm


from ic import make_partial_cascade
from mwu import main_routine as mwu
from edge_mwu import mwu_by_infection_direction
from noisy_binary_search import noisy_binary_search
from baselines import random_dog
from joblib import Parallel, delayed


def experiment_node_mwu_multiple_rounds(rounds,
                                        g,
                                        node2id, id2node,
                                        s2n_probas,
                                        fraction, epsilon, sampling_method,
                                        query_selection_method,
                                        check_neighbor_threshold,
                                        max_iter=float('inf'),
                                        seed=None):
    np.random.seed(seed)
    random.seed(seed)
    results = []
    for i in tqdm(range(rounds)):
        source, obs_nodes, infection_times, tree = make_partial_cascade(
            g, fraction, sampling_method=sampling_method)
        r = mwu(g, node2id, id2node,
                source, obs_nodes, infection_times,
                s2n_probas,
                epsilon,
                query_selection_method=query_selection_method,
                debug=False,
                max_iter=max_iter,
                save_log=False)
        results.append(r)
    return results


def experiment_edge_mwu_multiple_rounds(g,
                                        query_method,
                                        dir_tbl, inf_tbl,
                                        sp_len,
                                        check_neighbor_threshold=0.01,
                                        fraction=0.05,
                                        sampling_method='late_nodes',
                                        rounds=100,
                                        max_iter=float('inf')):
    counts = []
    for i in tqdm(range(rounds)):
        source, obs_nodes, infection_times, tree = make_partial_cascade(
            g, fraction, sampling_method=sampling_method)
        query_count = mwu_by_infection_direction(
            g, query_method,
            obs_nodes, infection_times, source,
            direction_reward_table=dir_tbl,
            inf_reward_table=inf_tbl,
            sp_len=sp_len,
            check_neighbor_threshold=check_neighbor_threshold,
            max_iter=max_iter,
            save_logs=False)
        counts.append(query_count)
    return counts


def experiment_multiple_rounds(source_finding_method, rounds, g, fraction, sampling_method):
    """source finding method should be given
    """
    cnts = []
    for i in tqdm(range(rounds)):
        source, obs_nodes, infection_times, tree = make_partial_cascade(
            g, fraction, sampling_method=sampling_method)
        try:
            c = source_finding_method(g, obs_nodes, infection_times)
            cnts.append(c)
        except RecursionError:
            pass

    return cnts


def experiment_dog_multiple_rounds(rounds, g, fraction, sampling_method,
                                   query_fraction):
    cnts = []
    for i in range(rounds):
        source, obs_nodes, infection_times, tree = make_partial_cascade(
            g, fraction, sampling_method=sampling_method)
        c = random_dog(g, obs_nodes, infection_times, query_fraction)
        cnts.append(c)
    return cnts


def counts_to_stat(counts):
    s = pd.Series(list(filter(lambda c: c is not False, counts)))
    return s.describe().to_dict()


def noisy_bs_one_round(g, sp_len,
                       consistency_multiplier,
                       debug=False):
    source, obs_nodes, infection_times, _ = make_partial_cascade(g, 0.01)

    c = noisy_binary_search(g, source, infection_times,
                            obs_nodes,
                            sp_len,
                            consistency_multiplier=consistency_multiplier,
                            max_iter=g.number_of_nodes(),
                            debug=debug)
    return c


def experiment_noisy_bs_n_rounds(g, N,
                                 consistency_multiplier,
                                 parallelize=True):
    sp_len = nx.shortest_path_length(g)
    if parallelize:
        return Parallel(n_jobs=-1)(delayed(noisy_bs_one_round)(
            g, sp_len, consistency_multiplier)
                                   for i in tqdm(range(N)))
    else:
        cnts = []
        for i in tqdm(range(N)):
            c = noisy_bs_one_round(g, sp_len, consistency_multiplier)
            cnts.append(c)
        return cnts
