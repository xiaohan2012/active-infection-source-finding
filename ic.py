import math
import networkx as nx
import numpy as np
import random
import itertools


from joblib import Parallel, delayed
from tqdm import tqdm
from utils import chunks


def sample_graph_from_infection(g):
    rands = np.random.rand(g.number_of_edges())
    active_edges = [(u, v) for (u, v), r in zip(g.edges_iter(), rands) if g[u][v]['p'] >= r]
    induced_g = nx.Graph()
    induced_g.add_nodes_from(g.nodes())
    induced_g.add_edges_from(active_edges)
    for u, v in induced_g.edges_iter():
        induced_g[u][v]['d'] = g[u][v]['d']
    return induced_g


def make_full_cascade(g, source=None, is_sampled=False):
    """
    """
    if source is None:
        idx = np.arange(g.number_of_nodes())
        source = g.nodes()[np.random.choice(idx)]

    if not is_sampled:
        induced_g = sample_graph_from_infection(g)
    else:
        induced_g = g
        
    if not induced_g.has_node(source):
        infection_times = {n: float('inf') for n in g.nodes_iter()}
        infection_times[source] = 0
    else:
        infection_times = nx.shortest_path_length(induced_g, source=source, weight='d')
        for n in g.nodes_iter():
            if n not in infection_times:
                infection_times[n] = float('inf')
    assert infection_times[source] == 0
    assert len(infection_times) == g.number_of_nodes()
    return infection_times


def make_partial_cascade(g, fraction, sampling_method='uniform'):
    """simulate one IC cascade and return the source, infection times and infection tree"""
    tree = None  # compatibility reason
    infection_times = make_full_cascade(g)

    infected_nodes = [n for n in g.nodes_iter() if not np.isinf(infection_times[n])]
    cascade_size = len(infected_nodes)

    sample_size = math.ceil(cascade_size * fraction)
    
    if sampling_method == 'uniform':
        idx = np.arange(len(infected_nodes))
        sub_idx = np.random.choice(idx, sample_size, replace=False)
        obs_nodes = set([infected_nodes[i] for i in sub_idx])
    elif sampling_method == 'late_nodes':
        obs_nodes = set(sorted(infected_nodes, key=lambda n: -infection_times[n])[:sample_size])0
    else:
        raise ValueError('unknown sampling methods')

    assert len(obs_nodes) > 0
    source = min(infection_times, key=lambda n: infection_times[n])

    return source, obs_nodes, infection_times, tree


def sp_len_dict_to_2d_array(n, sp_len, dtype=np.int16):
    d = np.zeros((n, n), dtype=dtype)
    for i in np.arange(n):
        d[i, :] = [sp_len[i].get(j, -1) for j in np.arange(n)]
    return d


def simulation_infection_time_all_sources(g):
    sg = sample_graph_from_infection(g)
    sp_len = nx.shortest_path_length(sg)
    n = g.number_of_nodes()
    return sp_len_dict_to_2d_array(n, sp_len)


def simulated_infection_time_3d(g, n_rounds):
    array_list = Parallel(n_jobs=-1)(delayed(simulation_infection_time_all_sources)(g)
                                     for i in tqdm(range(n_rounds)))
    return np.dstack(array_list)


def source_likelihood_1st_order(n_nodes, obs_nodes, inf_time_3d,
                                infection_times,
                                N2, eps=1):
    source_likelihood = np.ones(n_nodes, dtype=np.float64)
    for o in obs_nodes:
        single_probas = ((np.sum(inf_time_3d[:, o, :] == infection_times[o], axis=1) + eps)
                         / (N2 + eps))
        source_likelihood *= single_probas
        source_likelihood /= source_likelihood.sum()
    return source_likelihood


def source_likelihood_2nd_order(n_nodes, obs_nodes, inf_time_3d_by_p,
                                infection_times,
                                N2, eps=1):
    source_likelihood = np.ones(n_nodes, dtype=np.float64)
    obs_nodes = list(obs_nodes)
    random.shuffle(obs_nodes)
    pairs = chunks(obs_nodes, 2)
    for pr in pairs:
        if len(pr) == 1:
            break
        else:
            o1, o2 = pr
        bool_vect = np.logical_and(inf_time_3d_by_p[:, o1, :] == infection_times[o1],
                                   inf_time_3d_by_p[:, o2, :] == infection_times[o2])
        single_probas = ((np.sum(bool_vect, axis=1) + eps)
                         / (N2 + eps))
        source_likelihood *= single_probas
        source_likelihood /= source_likelihood.sum()
    return source_likelihood


def source_likelihood_merging_neighbor_pair(g, obs_nodes, inf_time_3d_by_p,
                                            infection_times,
                                            N2, eps=1):
    source_likelihood = np.ones(g.number_of_nodes(),
                                dtype=np.float64)
    node_pool = list(obs_nodes)
    pairs = []
    while len(node_pool) > 0:
        i = node_pool.pop()
        found_pair = False
        for j in node_pool:
            if g.has_edge(i, j):
                pairs.append((i, j))
                node_pool.remove(j)
                found_pair = True
                break
        if not found_pair:
            pairs.append((i, ))
    assert {i for tpl in pairs for i in tpl} == obs_nodes

    for pr in pairs:
        if len(pr) == 1:
            o = pr[0]
            single_probas = ((np.sum(inf_time_3d_by_p[:, o, :] == infection_times[o], axis=1) + eps)
                             / (N2 + eps))
        else:
            o1, o2 = pr
            bool_vect = np.logical_and(inf_time_3d_by_p[:, o1, :] == infection_times[o1],
                                       inf_time_3d_by_p[:, o2, :] == infection_times[o2])
            single_probas = ((np.sum(bool_vect, axis=1) + eps)
                             / (N2 + eps))
        source_likelihood *= single_probas
        source_likelihood /= source_likelihood.sum()
    return source_likelihood


def source_likelihood_1st_order_weighted_by_time(
        n_nodes, obs_nodes, inf_time_3d_by_p,
        infection_times,
        N2, eps=1,
        time_weight_func='linear'):
    """this method add more importance to earlier infected nodes
    """
    times = np.array([infection_times[o] for o in obs_nodes])
    t_max, t_min = times.max(), times.min()
    if time_weight_func == 'linear':
        def weight_func(t):
            if t_max == t_min:
                return 0
            else:
                return (t - t_min) / (t_max - t_min)
    elif time_weight_func == 'inverse':
        def weight_func(t):
            return 1 / (t_max - t + eps)
    else:
        raise ValueError('unsupported time weight func')

    source_likelihood = np.ones(n_nodes, dtype=np.float64)
    for o in obs_nodes:
        single_probas = ((np.sum(inf_time_3d_by_p[:, o, :] == infection_times[o], axis=1) + eps)
                         / (N2 + eps) +
                         weight_func(infection_times[o])) / 2
        source_likelihood *= single_probas
        source_likelihood /= source_likelihood.sum()
    return source_likelihood


def precondition_mask_and_count(o1, o2, inf_time_3d):
    sim_mask = np.invert(
        np.logical_or(
            inf_time_3d[:, o1, :] == -1,
            inf_time_3d[:, o2, :] == -1))
    counts = np.sum(sim_mask, axis=1)
    return sim_mask, counts


def source_likelihood_drs(n_nodes, obs_nodes, inf_time_3d,
                          infection_times,
                          N2,
                          use_time_weight=False,
                          use_preconditioning=True,
                          source=None,
                          debug=False,
                          eps=1e-3,
                          nan_proba=0.1):
    times = np.array([infection_times[o] for o in obs_nodes])
    t_max, t_min = times.max(), times.min()
    
    if use_time_weight:
        def weight_func(t):
            if t_max == t_min:
                return 0
            else:
                return (t - t_min) / (t_max - t_min)
    
    source_likelihood = np.ones(n_nodes, dtype=np.float64) / n_nodes
    obs_nodes = list(obs_nodes)
    for o1, o2 in itertools.combinations(obs_nodes, 2):
        t1, t2 = infection_times[o1], infection_times[o2]
        if use_preconditioning:
            sim_mask, counts = precondition_mask_and_count(o1, o2, inf_time_3d)

            probas = (np.sum(((inf_time_3d[:, o1, :] - inf_time_3d[:, o2, :]) == (t1 - t2)) * sim_mask,
                             axis=1)
                      / counts)
            probas[np.isnan(probas)] = nan_proba
        else:
            probas = (np.sum(
                (inf_time_3d[:, o1, :] - inf_time_3d[:, o2, :]) == (t2 - t1), axis=1)
                      / N2)
        if debug:
            print('t1={}, t2={}'.format(t1, t2))
            print('source reward: {:.2f}'.format(probas[source]))
            print('obs reward: {}'.format([probas[obs] for obs in set(obs_nodes)-{source}]))

        if use_time_weight:
            weight = weight_func(min(t1, t2))
            source_likelihood *= ((probas + weight) / 2 + eps)
        else:
            source_likelihood *= (probas + eps)
        source_likelihood /= source_likelihood.sum()
    return source_likelihood


def source_likelihood_pair_order(n_nodes, obs_nodes, inf_time_3d,
                                 infection_times,
                                 N2, eps=1e-5):
    source_likelihood = np.ones(n_nodes, dtype=np.float64) / n_nodes
    obs_nodes = list(obs_nodes)
    for o1, o2 in itertools.combinations(obs_nodes, 2):
        # assumes both o1 and o2 are **infected**
        t1, t2 = infection_times[o1], infection_times[o2]
        sim_mask, counts = precondition_mask_and_count(o1, o2, inf_time_3d)
        effective_matches = (((inf_time_3d[:, o1, :] < inf_time_3d[:, o2, :])
                             == (t1 < t2))
                             * sim_mask)
        probas = (np.sum(effective_matches, axis=1) / counts)
        probas[np.isnan(probas)] = 0
        source_likelihood *= (probas + eps)
        source_likelihood /= source_likelihood.sum()
    return source_likelihood


def source_likelihood_quad_time_difference(
        n_nodes, obs_nodes, inf_time_3d,
        infection_times,
        N2,
        sp_len,
        eps=1e-3,
        nan_proba=0.1,
        source=None,
        debug=False):
    source_likelihood = np.ones(n_nodes, dtype=np.float64) / n_nodes
    obs_nodes = list(obs_nodes)

    for o1, o2 in itertools.combinations(obs_nodes, 2):
        t1, t2 = infection_times[o1], infection_times[o2]
        sim_mask, counts = precondition_mask_and_count(o1, o2, inf_time_3d)
        diff_means = (np.sum((inf_time_3d[:, o1, :] - inf_time_3d[:, o2, :]) * sim_mask,
                             axis=1)
                      / counts)
        actual_diff = t1 - t2
        penalty = (np.power(actual_diff - diff_means, 2) /
                   (np.power(sp_len[:, o1], 2) + np.power(sp_len[:, o2], 2)))

        non_nan_penalty = penalty[np.invert(np.isnan(penalty))]
        if len(non_nan_penalty) == 0:
            continue

        max_value = non_nan_penalty.max()  # max of nan arrays gives nan
        
        probas = (1 - penalty / max_value)
        if debug:
            print('t1={}, t2={}'.format(t1, t2))
            print('diff_means', diff_means)
            print('source reward: {:.2f}'.format(probas[source]))
            print('obs reward: {}'.format([probas[obs] for obs in set(obs_nodes)-{source}]))
            
        # **questionabl**, what if no non-inf pair is found , does it mean we penalize all?
        probas[np.isnan(probas)] = nan_proba
        source_likelihood *= (probas + eps)

        source_likelihood /= source_likelihood.sum()
    return source_likelihood


def source_likelihood_quad_time_difference_normalized_by_dist_diff(
        n_nodes, obs_nodes, inf_time_3d,
        infection_times,
        N2,
        sp_len,
        eps=1e-5):
    source_likelihood = np.ones(n_nodes, dtype=np.float64) / n_nodes
    obs_nodes = list(obs_nodes)

    for o1, o2 in itertools.combinations(obs_nodes, 2):
        t1, t2 = infection_times[o1], infection_times[o2]
        sim_mask, counts = precondition_mask_and_count(o1, o2, inf_time_3d)

        diff_means = (np.sum((inf_time_3d[:, o1, :] - inf_time_3d[:, o2, :]) * sim_mask,
                             axis=1)
                      / counts)
        actual_diff = t1 - t2
        penalty = (np.power(actual_diff - diff_means, 2) /
                   (np.power(sp_len[:, o1] - sp_len[:, o2], 2) + 1.))

        probas = 1 - penalty / np.max(penalty)
        probas[np.isnan(probas)] = 0  # also questionable

        source_likelihood *= (probas + eps)
        source_likelihood /= source_likelihood.sum()

        if False:
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    pass
                except Warning as e:
                    pass
    return source_likelihood


def source_likelihood_abs_time_difference_normalized_by_dist_diff(
        n_nodes, obs_nodes, inf_time_3d,
        infection_times,
        N2,
        sp_len,
        eps=1e-5):
    source_likelihood = np.ones(n_nodes, dtype=np.float64) / n_nodes
    obs_nodes = list(obs_nodes)

    for o1, o2 in itertools.combinations(obs_nodes, 2):
        t1, t2 = infection_times[o1], infection_times[o2]
        sim_mask, counts = precondition_mask_and_count(o1, o2, inf_time_3d)

        diff_means = (np.sum((inf_time_3d[:, o1, :] - inf_time_3d[:, o2, :]) * sim_mask,
                             axis=1)
                      / counts)
        actual_diff = t1 - t2
        penalty = (np.absolute(actual_diff - diff_means) /
                   (np.absolute(sp_len[:, o1] - sp_len[:, o2]) + 1))
        probas = 1 - penalty / np.max(penalty)
        probas[np.isnan(probas)] = 0  # also questionable
        source_likelihood *= (probas + eps)
        source_likelihood /= source_likelihood.sum()
    return source_likelihood
