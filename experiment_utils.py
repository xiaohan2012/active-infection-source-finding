import numpy as np
import pandas as pd
import random
import networkx as nx
from tqdm import tqdm


from ic import gen_nontrivial_cascade, get_gvs
from mwu import mwu
from edge_mwu import mwu_by_infection_direction
from noisy_binary_search import noisy_binary_search
from baselines import random_dog
from joblib import Parallel, delayed


def experiment_mwu_n_rounds(rounds,
                            g,
                            p, q, epsilon,
                            sampling_method,
                            active_method,
                            reward_method,
                            n1=100,
                            seed=None):
    np.random.seed(seed)
    random.seed(seed)
    results = []
    gvs = get_gvs(g, p, n1)
    for i in tqdm(range(rounds)):
        infection_times, source, obs_nodes = gen_nontrivial_cascade(
            g, p, q)
        r = mwu(g, gvs,
                source, obs_nodes, infection_times, o2src_time=None,
                active_method=active_method,
                reward_method=reward_method,
                eps=0.2,
                max_iter=g.num_vertices(),
                use_uninfected=True,
                debug=False)
        if r > 0:
            results.append(r)
    return results


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


def experiment_dog_n_rounds(rounds, g, fraction, sampling_method,
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
    return s.describe()


def noisy_bs_one_round(g, sp_len,
                       sampled_graphs,
                       consistency_multiplier,
                       debug=False):
    source, obs_nodes, infection_times, _ = make_partial_cascade(g, 0.01)

    c = noisy_binary_search(g, source, infection_times,
                            obs_nodes,
                            sp_len,
                            consistency_multiplier=consistency_multiplier,
                            max_iter=g.number_of_nodes(),
                            sampled_graphs=sampled_graphs,
                            debug=debug)
    return c


def experiment_noisy_bs_n_rounds(g, sp_len,
                                 N,
                                 consistency_multiplier):
    cnts = []
    sampled_graphs = [sample_graph_from_infection(g)
                      for _ in range(100)]
    for i in tqdm(range(N)):
        c = noisy_bs_one_round(g, sp_len, sampled_graphs,
                               consistency_multiplier)
        cnts.append(c)
    return cnts
