import numpy as np
import pandas as pd
import pickle as pkl
from graph_tool.all import GraphView, load_graph
from sklearn.metrics import matthews_corrcoef, precision_score, recall_score
from glob import glob
from tqdm import tqdm


def evaluate_performance(g, pred_edges, obs_nodes, infection_times, true_edges):
    true_nodes = {i for e in true_edges for i in e}
    pred_nodes = {i for e in pred_edges for i in e}
    
    # mmc = matthews_corrcoef(true_labels, inferred_labels)
    # n_prec = precision_score(true_labels, inferred_labels)
    # n_rec = recall_score(true_labels, inferred_labels)

    common_nodes = true_nodes.intersection(pred_nodes)
    n_prec = len(common_nodes) / len(pred_nodes)
    n_rec = len(common_nodes) / len(true_nodes)
    obj = len(pred_edges)

    # on edges
    # pred_edges = set(extract_edges(pred_tree))
    # true_edges = set(extract_edges(true_tree))
    pred_edges = set(pred_edges)
    true_edges = set(true_edges)

    common_edges = pred_edges.intersection(true_edges)
    e_prec = len(common_edges) / len(pred_edges)
    e_rec = len(common_edges) / len(true_edges)
                    
    return (n_prec, n_rec, obj, e_prec, e_rec, len(pred_edges), len(true_edges))


def evaluate_from_result_dir(g, result_dir, qs):
    for q in tqdm(qs):
        rows = []
        for p in glob(result_dir + "/{}/*.pkl".format(q)):
            infection_times, source, obs_nodes, true_edges, pred_edges = pkl.load(open(p, 'rb'))
            # true_tree = edges2graph(g, true_edges)
            # pred_tree = edges2graph(g, pred_edges)
            scores = evaluate_performance(g, pred_edges, obs_nodes,
                                          infection_times, true_edges)
            rows.append(scores)
        df = pd.DataFrame(rows, columns=['n.prec', 'n.rec', 'obj',
                                         'e.prec', 'e.rec', '|T\'|', '|T|'])
        yield (result_dir + "/{}.pkl".format(q), df)


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

    for path, df in evaluate_from_result_dir(g, result_dir, qs):
        print('writing to {}'.format(path))
        df.describe().to_pickle(path)
