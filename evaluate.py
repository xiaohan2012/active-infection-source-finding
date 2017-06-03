import numpy as np
import pandas as pd
import pickle as pkl
from graph_tool.all import GraphView, load_graph
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from glob import glob
from utils import edges2graph


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


def evaluate_from_result_dir(graph_name, result_dir):
    g = load_graph('data/{}/graph.gt'.format(graph_name))
    rows = []
    for p in glob(result_dir + "/*.pkl"):
        infection_times, source, obs_nodes, true_edges, pred_edges = pkl.load(open(p, 'rb'))
        pred_tree = edges2graph(g, pred_edges)
        scores = evaluate_performance(g, pred_tree, obs_nodes, infection_times)
        rows.append(scores)
    df = pd.DataFrame(rows, columns=['mmc', 'prec', 'rec', 'obj'])
    print(df.describe())

if __name__ == '__main__':
    evaluate_from_result_dir('p2p-gnutella08', 'outputs/paper_experiment/0.01/')
