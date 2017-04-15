# coding: utf-8
import os
import sys
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from graph_tool.all import shortest_distance

from synthetic_data import load_data_by_gtype
from ic import sample_graph_from_infection, make_partial_cascade, simulated_infection_time_3d, \
    source_likelihood_1st_order, source_likelihood_2nd_order, \
    source_likelihood_merging_neighbor_pair, \
    source_likelihood_1st_order_weighted_by_time, \
    source_likelihood_drs, \
    source_likelihood_pair_order, \
    source_likelihood_quad_time_difference

from ic import simulate_cascade, observe_cascade, \
    get_o2src_time, source_likelihood_drs_gt, get_gvs
from graph_generator import add_p_and_delta
from utils import sp_len_2d


def source_likelihood_given_single_obs(g, o, t, N):
    matching_count = np.zeros(g.number_of_nodes(), dtype=np.float64)
    for i in range(N):
        sg = sample_graph_from_infection(g)
        sp_len = nx.shortest_path_length(sg, source=o)
        times = np.array([sp_len.get(n, float('inf')) for n in np.arange(g.number_of_nodes())])
        matching_count += (times == t)
    return matching_count / N


def source_likelihood_ratios_and_dists_gt(g,
                                          gvs, p, q, N1,
                                          estimation_method, debug=True):
    source_likelihood_array = []
    sources = []
    dist_array = []
    if debug:
        iters = tqdm(range(N1))
    else:
        iters = range(N1)
    for i in iters:
        source, infection_times = simulate_cascade(g, p)
        obs_nodes = observe_cascade(infection_times, q, method='uniform')
        source_estimation_params_gt = {
            'g': g,
            'obs_nodes': obs_nodes,
            'o2src_time': get_o2src_time(obs_nodes, gvs),
            'infection_times': infection_times,
        }
        sources.append(source)
        if estimation_method == 'drs-gt':
            source_likelihood = source_likelihood_drs_gt(
                **source_estimation_params_gt)
        else:
            raise ValueError()

        max_n = np.argmax(source_likelihood)
        dist_to_max_n = shortest_distance(g, source=source, target=max_n)
        dist_array.append(dist_to_max_n)
        source_likelihood_array.append(source_likelihood)

    source_likelihood_array = np.array(source_likelihood_array, dtype=np.float64)
    source_llh = np.array([source_likelihood_array[i, src]
                           for i, src in enumerate(sources)])
    ratios = source_llh / source_likelihood_array.max(axis=1)
    return {
        'ratio': pd.Series(ratios[np.invert(np.isnan(ratios))]).describe(),
        'dist': pd.Series(dist_array).describe(),
        'mu[s]': pd.Series(source_llh).describe(),
    }
        

def source_likelihood_ratios_and_dists(g, p, q, N1, N2,
                                       inf_time_3d,
                                       estimation_method,
                                       time_weight_func='linear',
                                       debug=True):
    g = add_p_and_delta(g, p, 1)
    source_likelihood_array = []
    sources = []
    dist_array = []

    if debug:
        iters = tqdm(range(N1))
    else:
        iters = range(N1)
    for i in iters:
        source, obs_nodes, infection_times, _ = make_partial_cascade(g, q, 'uniform')
        sources.append(source)

        source_estimation_params = {
            'n_nodes': g.number_of_nodes(),
            'obs_nodes': obs_nodes,
            'inf_time_3d': inf_time_3d,
            'infection_times': infection_times,
            'N2': N2}

        if estimation_method == '1st':
            source_likelihood = source_likelihood_1st_order(
                **source_estimation_params)
        elif estimation_method == '1st_time':
            source_likelihood = source_likelihood_1st_order_weighted_by_time(
                **source_estimation_params,
                time_weight_func=time_weight_func)
        elif estimation_method == 'drs':
            source_likelihood = source_likelihood_drs(
                **source_estimation_params)
        elif estimation_method == 'drs_time_early':
            pass
        elif estimation_method == 'pair_order':
            source_likelihood = source_likelihood_pair_order(
                **source_estimation_params)
        elif estimation_method == 'time_diff_dist_sum_quad':
            sp_len = sp_len_2d(g)
            source_likelihood = source_likelihood_quad_time_difference(
                **source_estimation_params,
                sp_len=sp_len)
        elif estimation_method == 'time_diff_dist_diff_quad':
            from ic import source_likelihood_quad_time_difference_normalized_by_dist_diff
            sp_len = sp_len_2d(g)
            source_likelihood = source_likelihood_quad_time_difference_normalized_by_dist_diff(
                **source_estimation_params,
                sp_len=sp_len)
        elif estimation_method == 'time_diff_dist_diff_abs':
            from ic import source_likelihood_abs_time_difference_normalized_by_dist_diff
            sp_len = sp_len_2d(g)
            source_likelihood = source_likelihood_abs_time_difference_normalized_by_dist_diff(
                **source_estimation_params,
                sp_len=sp_len)
            
        else:
            raise ValueError('unsupported source estimation method')
        
        max_n = np.argmax(source_likelihood)
        dist_to_max_n = nx.shortest_path_length(g, source=source, target=max_n)
        dist_array.append(dist_to_max_n)
        source_likelihood_array.append(source_likelihood)
    source_likelihood_array = np.array(source_likelihood_array, dtype=np.float64)
    source_llh = np.array([source_likelihood_array[i, src]
                           for i, src in enumerate(sources)])
    ratios = source_llh / source_likelihood_array.max(axis=1)
    return {
        'ratio': pd.Series(ratios[np.invert(np.isnan(ratios))]).describe(),
        'dist': pd.Series(dist_array).describe(),
        'mu-src': pd.Series(source_llh).describe(),
    }

if __name__ == '__main__':
    gtype = sys.argv[1]
    param = sys.argv[2]
    estimation_method = sys.argv[3]

    USE_GT = True
    DEBUG = False
    
    if DEBUG:
        N1 = 10
        N2 = 10
    else:
        N1 = 100  # experiment round
        N2 = 100  # simulation rounds
    if USE_GT:
        print('use graph_tool')
        from graph_tool.all import load_graph
        g = load_graph('data/{}/{}/graph.gt'.format(gtype, param))
        print('|V|={}'.format(g.num_vertices()))
    else:
        g = load_data_by_gtype(gtype, param)[0]
        print('|V|={}'.format(g.number_of_nodes()))

    ps = np.linspace(0.1, 1.0, 10)
    qs = np.linspace(0.1, 1.0, 10)

    if USE_GT:
        p2gvs = {p: get_gvs(g, p, N2) for p in ps}
    else:
        inf_time_3d_by_p = {p: simulated_infection_time_3d(add_p_and_delta(g, p, 1), N2) for p in ps}

    if not DEBUG:
        if USE_GT:
            rows = Parallel(n_jobs=-1)(delayed(source_likelihood_ratios_and_dists_gt)(
                g,
                p2gvs[p], p, q, N1,
                estimation_method=estimation_method,
                debug=False)
                                       for p in tqdm(ps) for q in qs)
        else:
            rows = Parallel(n_jobs=-1)(delayed(source_likelihood_ratios_and_dists)(
                g, p, q, N1, N2,
                inf_time_3d_by_p[p],
                estimation_method=estimation_method,
                debug=False)
                                   for p in tqdm(ps) for q in qs)
    else:
        if USE_GT:
            rows = [source_likelihood_ratios_and_dists_gt(
                g,
                p2gvs[p], p, q, N1,
                estimation_method=estimation_method,
                debug=False)
                for p in tqdm(ps) for q in qs]

        else:
            rows = [source_likelihood_ratios_and_dists(
                g, p, q, N1, N2,
                inf_time_3d_by_p[p],
                estimation_method=estimation_method,
                debug=False)
                for p in tqdm(ps) for q in qs]

    X, Y = np.meshgrid(ps, qs)
    ratio_median = np.array([r['ratio']['50%'] for r in rows]).reshape((len(ps), len(qs)))
    ratio_mean = np.array([r['ratio']['mean'] for r in rows]).reshape((len(ps), len(qs)))
    dist_median = np.array([r['dist']['50%'] for r in rows]).reshape((len(ps), len(qs)))
    dist_mean = np.array([r['dist']['mean'] for r in rows]).reshape((len(ps), len(qs)))

    dirname = 'outputs/source-likelihood-{}/{}'.format(
        estimation_method, gtype)

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    np.savez(dirname + '/{}'.format(param),
             X, Y, ratio_median, ratio_mean, dist_median, dist_mean)
