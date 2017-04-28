# coding: utf-8
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from graph_tool.all import shortest_distance

from ic import simulate_cascade, observe_cascade, \
    get_o2src_time, get_gvs, \
    sll_using_pairs
from utils import get_rank_index


def source_likelihood_stat(g,
                           gvs, p, q, N1,
                           estimation_method,
                           precond_method,
                           eps,
                           debug=True):
    sll_array = []
    sources = []
    dist_array = []

    if debug:
        iters = tqdm(range(N1))
    else:
        iters = range(N1)
        
    simulation_cache = {}
    for i in iters:
        while True:
            source, infection_times = simulate_cascade(g, p)
            obs_nodes = observe_cascade(infection_times, q, method='uniform')
            cascade_size = np.sum(infection_times != -1)

            if cascade_size >= 5:  # avoid small cascade
                break
        if debug:
            print('cascade size: {}'.format(cascade_size))

        # cache the simulation result
        o2src_time = get_o2src_time(set(obs_nodes) - set(simulation_cache.keys()),
                                    gvs,
                                    debug=debug)
        simulation_cache.update(o2src_time)

        source_estimation_params_gt = {
            'g': g,
            'obs_nodes': obs_nodes,
            'o2src_time': simulation_cache,
            'infection_times': infection_times,
            'method': estimation_method,
            'precond_method': precond_method,
            'eps': eps
        }

        sources.append(source)
        sll = sll_using_pairs(**source_estimation_params_gt)

        winner = np.argmax(sll)
        dist_to_max_n = shortest_distance(g, source=source, target=winner)
        dist_array.append(dist_to_max_n)
        sll_array.append(sll)

    source_likelihood_array = np.array(sll_array, dtype=np.float64)
    source_llh = np.array([source_likelihood_array[i, src]
                           for i, src in enumerate(sources)])
    ratios = source_llh / source_likelihood_array.max(axis=1)
    ranks = np.array([get_rank_index(source_likelihood_array[i, :], src)
                      for i, src in enumerate(sources)])

    return {
        'ratio': pd.Series(ratios[np.invert(np.isnan(ratios))]).describe(),
        'dist': pd.Series(dist_array).describe(),
        'mu[s]': pd.Series(source_llh).describe(),
        'rank': pd.Series(ranks).describe(),
    }
        

if __name__ == '__main__':
    import argparse
    from graph_tool.all import load_graph
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gtype', required=True)
    parser.add_argument('-p', '--param', default='2-6')
    parser.add_argument('-m', '--method', default='exact')
    parser.add_argument('-c', '--precond_method', default=None)
    parser.add_argument('-e', '--eps', type=float, default=0.2)
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--n1', type=int, help="simulation rounds for parameter esimation", default=100)
    parser.add_argument('--n2', type=int, help="experiment rounds", default=2)

    parser.add_argument('--p1', type=float, help="start of p", default=0.2)
    parser.add_argument('--p2', type=float, help="end of p", default=1.0)
    parser.add_argument('--ps', type=float, help="step size of p", default=0.1)
    
    parser.add_argument('--q1', type=float, help="start of q", default=0.2)
    parser.add_argument('--q2', type=float, help="end of q", default=1.0)
    parser.add_argument('--qs', type=float, help="step size of q", default=0.1)
    
    args = parser.parse_args()
    gtype = args.gtype
    param = args.param
    estimation_method = args.method
    precond_method = args.precond_method
    eps = args.eps
    
    DEBUG = args.debug
    
    if DEBUG:
        N1 = args.n1
        N2 = args.n2
    else:
        N1 = args.n1  # simulation rounds
        N2 = args.n2  # experiment round

    g = load_graph('data/{}/{}/graph.gt'.format(gtype, param))
    print('|V|={}'.format(g.num_vertices()))

    ps = np.arange(args.p1, args.p2 + 1e-10, args.ps)
    qs = np.arange(args.q1, args.q2 + 1e-10, args.qs)

    print('N1: {}'.format(N1))
    print('N2: {}'.format(N2))
    print('ps: {}'.format(ps))
    print('qs: {}'.format(qs))

    p2gvs = {p: get_gvs(g, p, N2) for p in ps}

    if not DEBUG:
        rows = Parallel(n_jobs=-1)(delayed(source_likelihood_stat)(
            g,
            p2gvs[p], p, q, N1,
            estimation_method=estimation_method,
            precond_method=precond_method,
            eps=eps,
            debug=DEBUG)
                                   for p in tqdm(ps) for q in qs)
    else:
        rows = [source_likelihood_stat(
            g,
            p2gvs[p], p, q, N1,
            estimation_method=estimation_method,
            precond_method=precond_method,
            eps=eps,
            debug=DEBUG)
            for p in tqdm(ps) for q in qs]

    X, Y = np.meshgrid(ps, qs)
    names = ['ratio', 'dist', 'mu[s]', 'rank']
    stats = ['50%', 'mean']
    data = [np.array([r[name][stat] for r in rows]).reshape((len(ps), len(qs)))
            for name in names for stat in stats]

    dirname = 'outputs/source-likelihood-{}-{}/{}'.format(
        estimation_method,
        precond_method,
        gtype)
    print(dirname)
    if DEBUG:
        from tempfile import NamedTemporaryFile
        f = NamedTemporaryFile(delete=False)
        path = f.name
    else:
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        path = dirname + '/{}'.format(param)
    np.savez(path,
             X, Y,
             *data)
