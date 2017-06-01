import numpy as np
import pandas as pd
import time
import pickle as pkl
from graph_tool import GraphView
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from tqdm import tqdm
from cascade import gen_nontrivial_cascade
from utils import earliest_obs_node


def get_tree(g, infection_times, source, obs_nodes, method, verbose=False, debug=False):
    root = earliest_obs_node(obs_nodes, infection_times)
    if method == 'mst':
        from steiner_tree_mst import steiner_tree_mst, build_closure
        tree = steiner_tree_mst(g, root, infection_times, source, obs_nodes, debug=debug,
                                closure_builder=build_closure,
                                strictly_smaller=False,
                                verbose=verbose)
    elif method == 'truncated_mst':
        from steiner_tree_mst import steiner_tree_mst
        from mst_truncated import build_truncated_closure
        tree = steiner_tree_mst(g, root, infection_times, source, obs_nodes,
                                closure_builder=build_truncated_closure,
                                k=1,  # that's the difference
                                verbose=verbose,
                                debug=debug)
    elif method == 'greedy':
        from steiner_tree_greedy import steiner_tree_greedy
        tree = steiner_tree_greedy(g, root, infection_times, source, obs_nodes,
                                   debug=debug,
                                   verbose=verbose)
    elif method == 'tbfs':
        from steiner_tree_order import temporal_bfs
        tree = temporal_bfs(g, root, infection_times[root], infection_times, source, obs_nodes,
                            debug=debug)
    return tree


def run_k_rounds(g, p, q, model, method, k=100, verbose=False, debug=False):
    rows = []
    iters = range(k)
    if verbose:
        iters = tqdm(iters)

    for i in iters:
        if verbose:
            print('{}th simulation'.format(i))
            print('gen cascade')
        infection_times, source, obs_nodes = gen_nontrivial_cascade(
            g, p, q, model=model,
            return_tree=False, source_includable=True)
        stime = time.time()
        tree = get_tree(g, infection_times, source, obs_nodes, method,
                        verbose=verbose,
                        debug=debug)
        if tree:
            scores = evaluate_performance(g, tree, obs_nodes, infection_times)
            scores += (time.time() - stime, )
            rows.append(scores)
        else:
            if debug:
                print('fail to get tree')
                print('writing bad example')
                pkl.dump((g, infection_times, source, obs_nodes),
                         open('bad_examples/1.pkl', 'wb'))

    df = pd.DataFrame(rows, columns=['mmc', 'prec', 'rec', 'obj', 'time'])
    return df.describe()


def evaluate_performance(g, tree, obs_nodes, infection_times):
    tree = GraphView(tree)
    vfilt = tree.new_vertex_property('bool')
    vfilt.a = True
    tree.set_vertex_filter(vfilt)
    
    inferred_labels = np.zeros(g.num_vertices())
    true_labels = (infection_times != -1)
    for e in list(tree.edges()):
        u, v = map(int, e)
        inferred_labels[u] = 1
        inferred_labels[v] = 1
    
    idx = np.sort(list(set(np.arange(g.num_vertices())) - set(obs_nodes)))
    
    true_labels = true_labels[idx]
    inferred_labels = inferred_labels[idx]
    
    mmc = matthews_corrcoef(true_labels, inferred_labels)
    prec = precision_score(true_labels, inferred_labels)
    rec = recall_score(true_labels, inferred_labels)
    obj = tree.num_edges()
    return (mmc, prec, rec, obj)


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

    args = parser.parse_args()
    gtype = args.gtype
    param = args.param
    p = args.infection_proba
    q = args.report_proba
    method = args.method
    model = args.model
    k = args.repeat_times
    output_path = args.output_path

    print("""graph: {}
model: {}
p: {}
q: {}
k: {}
method: {}""".format(gtype, model, p, q, k, method))

    g = load_graph('data/{}/{}/graph.gt'.format(gtype, param))
    
    stat = run_k_rounds(g, p, q, model, method, k=k,
                        verbose=args.verbose,
                        debug=args.debug)

    print('write result to {}'.format(output_path))

    dirname = os.path.dirname(output_path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    stat.to_pickle(output_path)
