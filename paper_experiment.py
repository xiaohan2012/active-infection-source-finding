import pandas as pd
import time
import pickle as pkl
from tqdm import tqdm

from cascade import gen_nontrivial_cascade
from utils import earliest_obs_node

from steiner_tree_mst import steiner_tree_mst, build_closure
from steiner_tree_greedy import steiner_tree_greedy
from steiner_tree import get_steiner_tree
from temporal_bfs import temporal_bfs


DUMP_PERFORMANCE = False


def get_tree(g, infection_times, source, obs_nodes, method, verbose=False, debug=False):
    root = earliest_obs_node(obs_nodes, infection_times)
    if method == 'mst':
        tree = steiner_tree_mst(g, root, infection_times, source, obs_nodes, debug=debug,
                                closure_builder=build_closure,
                                strictly_smaller=False,
                                verbose=verbose)
    elif method == 'greedy':
        tree = steiner_tree_greedy(g, root, infection_times, source, obs_nodes,
                                   debug=debug,
                                   verbose=verbose)
    elif method == 'no-order':
        tree = get_steiner_tree(
            g, root, obs_nodes,
            debug=False,
            verbose=False,
        )
    elif method == 'tbfs':
        tree = temporal_bfs(g, root, infection_times, source, obs_nodes,
                            debug=debug,
                            verbose=verbose)
    return tree


def one_run(g, p, q, model, result_dir, i, verbose, debug):
    infection_times, source, obs_nodes, true_tree = gen_nontrivial_cascade(
        g, p, q, model=model,
        return_tree=True, source_includable=True)
    stime = time.time()
    tree = get_tree(g, infection_times, source, obs_nodes, method,
                    verbose=verbose,
                    debug=debug)

    # pickle cascade and pred_tree
    true_edges = [(int(u), int(v)) for u, v in true_tree.edges()]
    pred_edges = [(int(u), int(v)) for u, v in tree.edges()]
    pkl.dump((infection_times, source, obs_nodes, true_edges, pred_edges),
             open(result_dir + '/{}.pkl'.format(i), 'wb'))
    return time.time() - stime

    
# @profile
def run_k_rounds(g, p, q, model, method,
                 result_dir,
                 k=100,
                 do_parallel=False,
                 verbose=False, debug=False):
    iters = range(k)
    if verbose:
        iters = tqdm(iters)

    if not do_parallel:
        rows = []
        for i in iters:
            if verbose:
                print('{}th simulation'.format(i))
                print('gen cascade')
            time_cost = one_run(g, p, q, model, result_dir, i, verbose, debug)
            rows.append(time_cost)
    else:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=6)(delayed(one_run)(g, p, q, model, result_dir, i,
                                                    verbose, debug)
                                   for i in iters)

    df = pd.DataFrame(rows, columns=['time'])
    return df.describe()

if __name__ == '__main__':
    import argparse
    import os
    from graph_tool.all import load_graph
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gtype', required=True)
    parser.add_argument('--param', default='')
    parser.add_argument('-m', '--method', required=True)
    parser.add_argument('-l', '--model', required=True)
    parser.add_argument('-p', '--infection_proba', type=float, default=0.5)
    parser.add_argument('-q', '--report_proba', type=float, default=0.1)
    parser.add_argument('-k', '--repeat_times', type=int, default=100)
    parser.add_argument('-o', '--output_path', default='output/paper_experiment')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-d', '--debug', action='store_true')
    parser.add_argument('--parallel', action='store_true')

    args = parser.parse_args()
    gtype = args.gtype
    param = args.param
    p = args.infection_proba
    q = args.report_proba
    method = args.method
    model = args.model
    k = args.repeat_times
    output_path = args.output_path
    do_parallel = args.parallel

    print("""graph: {}
model: {}
p: {}
q: {}
k: {}
do_parallel: {}
method: {}""".format(gtype, model, p, q, k, do_parallel, method))

    g = load_graph('data/{}/{}/graph.gt'.format(gtype, param))

    dirname = os.path.dirname(output_path)

    result_dir = os.path.join(dirname, "{}".format(q))
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        
    stat = run_k_rounds(g, p, q, model, method,
                        result_dir=result_dir,
                        k=k,
                        do_parallel=do_parallel,
                        verbose=args.verbose,
                        debug=args.debug)

    if DUMP_PERFORMANCE:
        print('write result to {}'.format(output_path))

        stat.to_pickle(output_path)
    else:
        print('write result to {}'.format(output_path))
        stat.to_pickle(output_path)
        print('done')        
