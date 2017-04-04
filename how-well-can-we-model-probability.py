# coding: utf-8
import os
import sys
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

from synthetic_data import load_data_by_gtype
from ic import sample_graph_from_infection, make_partial_cascade, simulated_infection_time_3d, \
    source_likelihood_1st_order, source_likelihood_2nd_order, \
    source_likelihood_merging_neighbor_pair
from graph_generator import add_p_and_delta


gtype = sys.argv[1]
param = sys.argv[2]
estimation_method = sys.argv[3]

DEBUG = False

if DEBUG:
    N1 = 10
    N2 = 10
else:
    N1 = 500  # experiment round
    N2 = 500  # simulation rounds

g = load_data_by_gtype(gtype, param)[0]
print('|V|={}'.format(g.number_of_nodes()))


# In[41]:

def source_likelihood_given_single_obs(g, o, t, N):
    matching_count = np.zeros(g.number_of_nodes(), dtype=np.float64)
    for i in range(N):
        sg = sample_graph_from_infection(g)
        sp_len = nx.shortest_path_length(sg, source=o)
        times = np.array([sp_len.get(n, float('inf')) for n in np.arange(g.number_of_nodes())])
        matching_count += (times == t)
    return matching_count / N


def source_likelihood_ratios_and_dists(g, p, q, N1, N2,
                                       inf_time_3d_by_p,
                                       estimation_method,
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

        source_estimation_params = (g.number_of_nodes(),
                                    obs_nodes, inf_time_3d_by_p,
                                    infection_times,
                                    N2)
        if estimation_method == '1st':
            source_likelihood = source_likelihood_1st_order(
                *source_estimation_params)
        elif estimation_method == '2nd':
            source_likelihood = source_likelihood_2nd_order(
                *source_estimation_params)
        elif estimation_method == 'nbr_pair':
            source_likelihood = source_likelihood_merging_neighbor_pair(
                g,
                obs_nodes, inf_time_3d_by_p,
                infection_times,
                N2)
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
        'dist': pd.Series(dist_array).describe()
    }

ps = np.linspace(0.1, 1.0, 10)
qs = np.linspace(0.1, 1.0, 10)
inf_time_3d_by_p = {p: simulated_infection_time_3d(add_p_and_delta(g, p, 1), N2) for p in ps}


if not DEBUG:
    rows = Parallel(n_jobs=-1)(delayed(source_likelihood_ratios_and_dists)(
        g, p, q, N1, N2,
        inf_time_3d_by_p[p],
        estimation_method=estimation_method,
        debug=False)
                               for p in tqdm(ps) for q in qs)
else:
    rows = [source_likelihood_ratios_and_dists(
        g, p, q, N1, N2,
        inf_time_3d_by_p[p],
        estimation_method=estimation_method,
        debug=False)
            for p in tqdm(ps) for q in qs]


# In[38]:

# mpld3.enable_notebook()
X, Y = np.meshgrid(ps, qs)
ratio_median = np.array([r['ratio']['50%'] for r in rows]).reshape((len(ps), len(qs)))
ratio_mean = np.array([r['ratio']['mean'] for r in rows]).reshape((len(ps), len(qs)))
dist_median = np.array([r['dist']['50%'] for r in rows]).reshape((len(ps), len(qs)))
dist_mean = np.array([r['dist']['mean'] for r in rows]).reshape((len(ps), len(qs)))

# In[39]:

dirname = 'outputs/source-likelihood-{}/{}'.format(estimation_method, gtype)
if not os.path.exists(dirname):
    os.makedirs(dirname)

np.savez(dirname + '/{}'.format(param),
         X, Y, ratio_median, ratio_mean, dist_median, dist_mean)
