import numpy as np
import networkx as nx
from synthetic_data import load_data_by_gtype


def main(gtype, size_param):
    output_path = 'data/{}/{}/sp_len.npz'.format(gtype, size_param)
    print(gtype, size_param)
    g = load_data_by_gtype(gtype, size_param)[0]
    sp_len = nx.shortest_path_length(g, weight='d')

    n = g.number_of_nodes()
    m = np.zeros((n, n), dtype=np.uint8)  # might be a bit dangerous
    for i in g.nodes_iter():
        for j in g.nodes_iter():
            m[i, j] = sp_len[i][j]
    print('writing to {}'.format(output_path))
    np.save(output_path, m)

if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])
