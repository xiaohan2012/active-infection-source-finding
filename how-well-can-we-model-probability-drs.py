
# coding: utf-8

# In[22]:

import sys
import os
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from copy import copy

from synthetic_data import load_data_by_gtype
from ic import simulated_infection_time_3d, make_partial_cascade
from graph_generator import add_p_and_delta


gtype = sys.argv[1]
param = sys.argv[2]
DEBUG = False

if DEBUG:
    N1 = 10
    N2 = 10
else:
    N1 = 500  # experiment round
    N2 = 500  # simulation rounds

g = load_data_by_gtype(gtype, param)[0]
print('|V|={}'.format(g.number_of_nodes()))


def source_likelihood_ratios_and_dists_by_obs_pairs(
        g, p, q, N1, N2, inf_time_3d, f_ref_nodes=0.1,
        eps=1e-6,
        debug=True):
    """inf_time_3d: shape = source x node x rounds"""
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
        source_likelihood = np.ones(g.number_of_nodes(), dtype=np.float64)
        
        n_ref_nodes = (int(len(obs_nodes) * f_ref_nodes) or 1)
        ref_nodes = np.random.permutation(np.array(list(obs_nodes)))[:n_ref_nodes]
        remaining_nodes = copy(obs_nodes)
        for ref_node in ref_nodes:
            remaining_nodes -= {ref_node}
            for o in remaining_nodes:
                t1, t2 = infection_times[o], infection_times[ref_node]
                probas = ((np.sum((inf_time_3d[:, o, :] - inf_time_3d[:, ref_node, :]) == (t1 - t2), axis=1)
                           + eps)
                          / (N2 + eps))
                source_likelihood *= probas

                source_likelihood /= source_likelihood.sum()
            # source_likelihood_given_single_obs(g, o, infection_times[o], N2)
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
        'dist': pd.Series(dist_array).describe()
    }


# In[28]:

ps = np.linspace(0.1, 1.0, 10)
qs = np.linspace(0.1, 1.0, 10)
if DEBUG:
    k = 1
else:
    k = 4

ref_nodes_fractions = np.linspace(0.1, 0.1*k, k)

inf_time_3d_by_p = {p: simulated_infection_time_3d(add_p_and_delta(g, p, 1), N2) for p in ps}


for f_ref_nodes in ref_nodes_fractions:
    print('f_ref_nodes={}'.format(f_ref_nodes))
    if not DEBUG:
        rows = Parallel(n_jobs=-1)(
            delayed(source_likelihood_ratios_and_dists_by_obs_pairs)(
                g, p, q, N1, N2, inf_time_3d_by_p[p],
                f_ref_nodes=f_ref_nodes,
                debug=False)
            for p in tqdm(ps) for q in qs)
    else:
        rows = [source_likelihood_ratios_and_dists_by_obs_pairs(
            g, p, q, N1, N2, inf_time_3d_by_p[p],
            f_ref_nodes=f_ref_nodes,
            debug=False)
            for p in tqdm(ps) for q in qs]

    X, Y = np.meshgrid(ps, qs)
    ratio_median = np.array([r['ratio']['50%'] for r in rows]).reshape((len(ps), len(qs)))
    ratio_mean = np.array([r['ratio']['mean'] for r in rows]).reshape((len(ps), len(qs)))
    dist_median = np.array([r['dist']['50%'] for r in rows]).reshape((len(ps), len(qs)))
    dist_mean = np.array([r['dist']['mean'] for r in rows]).reshape((len(ps), len(qs)))

    dirname = 'outputs/source-likelihood-drs/{}'.format(gtype)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    np.savez(dirname + '/{}-f_ref-{:.1f}'.format(param, f_ref_nodes),
             X, Y, ratio_median, ratio_mean, dist_median, dist_mean)
