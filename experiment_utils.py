import numpy as np
import pandas as pd
import random
from ic import make_partial_cascade
from mwu import main_routine as mwu
from edge_mwu import mwu_by_infection_direction
from tqdm import tqdm
from baselines import baseline_dog_tracker


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
            max_iter=max_iter,
            save_logs=False)
        counts.append(query_count)
    return counts


def experiment_dog_multiple_rounds(rounds, g, fraction, sampling_method):
    cnts = []
    for i in range(rounds):
        source, obs_nodes, infection_times, tree = make_partial_cascade(
            g, fraction, sampling_method=sampling_method)
        c = baseline_dog_tracker(g, obs_nodes, infection_times)
        cnts.append(c)
    return cnts




def counts_to_stat(counts):
    s = pd.Series(list(filter(lambda c: c is not False, counts)))
    return s.describe().to_dict()