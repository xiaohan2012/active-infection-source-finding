import os
import numpy as np
import pandas as pd
import pickle as pkl
from graph_tool.all import load_graph, GraphView
from glob import glob
from tqdm import tqdm
from utils import edges2graph, get_infection_time, earliest_obs_node, to_directed, get_leaves, get_paths
from sklearn.metrics.pairwise import cosine_similarity


from graph_tool.topology import shortest_path
from scipy.stats import kendalltau


def get_rank_corrs(tree, root, true_tree_paths, debug=False):
    pred_leaves = get_leaves(tree)
    #     print(pred_leaves)
    corrs = []
    assert len(pred_leaves) > 0, 'empty predicted leaves'
    if debug:
        print('pred_leaves', pred_leaves)
    for o in pred_leaves:
        if o == root:
            continue
        true_root = next(v for v in tree.vertices() if v.in_degree() == 0 and v.out_degree() > 0)
        assert true_root == root
        path = shortest_path(tree, source=root, target=tree.vertex(o))[0]
        path = list(map(int, path))
        path_nodes = set(path)
        assert len(path) > 0
        if path_nodes:
            best_true_path = max(true_tree_paths,
                                 key=lambda true_path: jaccard_sim(set(true_path), path_nodes))

            # some similarity        
            if debug:
                print('path', path)
                print('best_true_path', best_true_path)
            common_nodes = path_nodes.intersection(set(best_true_path))
            if len(common_nodes) > 1:
                if debug:
                    print('common_nodes', common_nodes)
                n2rank_pred = {n: i for i, n in enumerate(path)}
                n2rank_true = {n: i for i, n in enumerate(best_true_path)}
                if debug:
                    print('n2rank_pred', n2rank_pred)
                    print('n2rank_true', n2rank_true)
                pred_rank = [n2rank_pred[n] for n in common_nodes]
                true_rank = [n2rank_true[n] for n in common_nodes]

                if debug:
                    print('pred_rank', pred_rank)
                    print('true_rank', true_rank)                
                corr = kendalltau(pred_rank, true_rank)
                
                if debug:
                    print('corr', corr[0])
                    print()
                corrs.append(corr[0])
    #             print()
    if len(corrs) == 0:
        # print('empty corrs')
        pass
        # raise ValueError('empty corrs list')
    return corrs


def jaccard_sim(s1, s2):
    return len(s1.intersection(s2)) / len(s1.union(s2))

    
def infer_infection_time_from_tree(t, source):
    b = t.new_vertex_property('bool')
    b.a = True
    t.set_vertex_filter(b)
    return get_infection_time(t, source=source)

# @profile
def evaluate_performance(g, root, source, pred_edges, obs_nodes, infection_times,
                         true_edges, convert_to_directed=False):
    true_nodes = {i for e in true_edges for i in e}
    pred_nodes = {i for e in pred_edges for i in e}
    
    # mmc = matthews_corrcoef(true_labels, inferred_labels)
    # n_prec = precision_score(true_labels, inferred_labels)
    # n_rec = recall_score(true_labels, inferred_labels)

    common_nodes = true_nodes.intersection(pred_nodes)
    n_prec = len(common_nodes) / len(pred_nodes)
    n_rec = len(common_nodes) / len(true_nodes)
    obj = len(pred_edges)

    pred_tree = edges2graph(g, pred_edges)
    true_tree = edges2graph(g, true_edges)

    if root is None:
        root = next(v for v in pred_tree.vertices() if v.in_degree() == 0 and v.out_degree() > 0)

    if convert_to_directed:
        # print('convert to directed')
        # assert not pred_tree.is_directed()
        pred_tree = to_directed(g,
                                GraphView(pred_tree, directed=False),
                                root)

    pred_times = infer_infection_time_from_tree(pred_tree, root)

    tree_nodes = list({u for e in pred_edges for u in e})
    cosine_sim = cosine_similarity([pred_times[tree_nodes]], [infection_times[tree_nodes]])[0, 0]

    common_edges = set(pred_edges).intersection(true_edges)
    e_prec = len(common_edges) / len(pred_edges)
    e_rec = len(common_edges) / len(true_edges)

    # leaves = get_leaves(true_tree)
    # true_tree_paths = get_paths(true_tree, source, leaves)
    # corrs = get_rank_corrs(pred_tree, root, true_tree_paths, debug=False)

    # return (n_prec, n_rec, obj, cosine_sim, e_prec, e_rec, np.mean(corrs))
    return (n_prec, n_rec, obj, cosine_sim, e_prec, e_rec)

def evaluate_from_result_dir(g, result_dir, qs):
    for q in tqdm(qs):
        rows = []
        for p in glob(result_dir + "/{}/*.pkl".format(q)):
            # print(p)
            # TODO: add root
            infection_times, source, obs_nodes, true_edges, pred_edges = pkl.load(open(p, 'rb'))
            
            convert_to_directed = ("no-order" in p)
            if convert_to_directed:
                # baseline does not care where the root is
                root = earliest_obs_node(obs_nodes, infection_times)
            else:
                root = None

            scores = evaluate_performance(g, root, source, pred_edges, obs_nodes,
                                          infection_times, true_edges,
                                          convert_to_directed=convert_to_directed)
            rows.append(scores)
        path = result_dir + "/{}.pkl".format(q)
        if rows:
            df = pd.DataFrame(rows, columns=['n.prec', 'n.rec', 'obj', 'cos-sim',
                                             'e.prec', 'e.rec'
            ])
            yield (path, df)
        else:
            if os.path.exists(path):
                os.remove(path)
            yield None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gtype', required=True)
    parser.add_argument('-l', '--model', required=True)
    parser.add_argument('-m', '--method', required=True)
    parser.add_argument('-q', '--qs', type=float, nargs="+")
    parser.add_argument('-o', '--output_dir', default='outputs/paper_experiment')

    args = parser.parse_args()
    gtype = args.gtype
    qs = args.qs
    method = args.method
    model = args.model
    output_dir = args.output_dir

    print("""graph: {}
model: {}
qs: {}
method: {}""".format(gtype, model, qs, method))

    result_dir = "{output_dir}/{gtype}/{model}/{method}/qs".format(
        output_dir=output_dir,
        gtype=gtype,
        model=model,
        method=method)

    g = load_graph('data/{}/graph.gt'.format(gtype))

    for r in evaluate_from_result_dir(g, result_dir, qs):
        if r:
            path, df = r
            print('writing to {}'.format(path))
            df.describe().to_pickle(path)
        else:
            print('not result.')
