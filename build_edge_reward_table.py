import networkx as nx
import pickle as pkl
from tqdm import tqdm
from collections import defaultdict
from synthetic_data import load_data_by_gtype
from ic import sample_graph_from_infection


def build_reward_table(g, n_rounds=100):
    """
    return two tables.
    
    table 1 maps (s, u, v) to real number
    s: source
    u: earlier infected node
    v: later infected node
    
    table 2 maps (s, u) to real number, the probability that u is not infected given s
    """
    tbl1 = defaultdict(float)
    tbl2 = defaultdict(float)
    increase = 1 / n_rounds
    for i in tqdm(range(n_rounds)):
        sampled_g = sample_graph_from_infection(g)
        sp_path = nx.shortest_path(sampled_g)
        for s in g.nodes_iter():
            for q in g.nodes_iter():
                try:
                    path = sp_path[s][q]
                    if len(path) >= 2:
                        tbl1[(s, path[-2], q)] += increase
                except KeyError:
                    tbl2[(s, q)] += increase
    return tbl1, tbl2


def main(gtype, size_param):
    output_path = 'data/{}/{}/edge_reward_tables.pkl'.format(
        gtype, size_param)
    g = load_data_by_gtype(gtype, size_param)[0]
    tbl1, tbl2 = build_reward_table(g)
    print('writing to {}'.format(output_path))
    pkl.dump((tbl1, tbl2),
             open(output_path, 'wb'))

if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])
