import numpy as np
import pandas as pd
import pickle as pkl
from graph_tool.all import GraphView, load_graph
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from glob import glob
from utils import edges2graph, extract_edges


def evaluate_performance(g, pred_tree, obs_nodes, infection_times, true_tree):
    tree = GraphView(pred_tree)
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
    n_prec = precision_score(true_labels, inferred_labels)
    n_rec = recall_score(true_labels, inferred_labels)
    obj = tree.num_edges()

    # on edges
    pred_edges = set(extract_edges(pred_tree))
    true_edges = set(extract_edges(true_tree))

    common_edges = pred_edges.intersection(true_edges)
    e_prec = len(common_edges) / len(pred_edges)
    e_rec = len(common_edges) / len(true_edges)
                    
    return (mmc, n_prec, n_rec, obj, e_prec, e_rec, len(pred_edges), len(true_edges))


def evaluate_from_result_dir(graph_name, result_dir):
    g = load_graph('data/{}/graph.gt'.format(graph_name))
    rows = []
    for p in glob(result_dir + "/*.pkl"):
        infection_times, source, obs_nodes, true_edges, pred_edges = pkl.load(open(p, 'rb'))
        true_tree = edges2graph(g, true_edges)
        pred_tree = edges2graph(g, pred_edges)
        scores = evaluate_performance(g, pred_tree, obs_nodes, infection_times, true_tree)
        rows.append(scores)
    df = pd.DataFrame(rows, columns=['mmc', 'n.prec', 'n.rec', 'obj',
                                     'e.prec', 'e.rec', '|T\'|', '|T|'])
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gtype', required=True)
    parser.add_argument('-l', '--model', required=True)
    parser.add_argument('-m', '--method', required=True)
    parser.add_argument('-q', '--report_proba', type=float, default=0.1)
    parser.add_argument('-o', '--output_dir', default='outputs/paper_experiment')

    args = parser.parse_args()
    gtype = args.gtype
    q = args.report_proba
    method = args.method
    model = args.model
    output_dir = args.output_dir

    print("""graph: {}
model: {}
q: {}
method: {}""".format(gtype, model, q, method))

    result_dir = "{output_dir}/{gtype}/{model}/{method}/qs/{q}".format(
        output_dir=output_dir,
        gtype=gtype,
        model=model,
        method=method,
        q=q)
    
    df = evaluate_from_result_dir(gtype, result_dir)
    
    output_path = result_dir + '.pkl'
    
    print('writing to {}'.format(output_path))
    df.describe().to_pickle(output_path)
