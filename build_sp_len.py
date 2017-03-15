import networkx as nx
import pickle as pkl
from synthetic_data import SP_LEN_NAME, load_data_by_gtype


def main(gtype, size_param):
    output_path = 'data/{}/{}/{}.pkl'.format(
        gtype, size_param, SP_LEN_NAME)
    g = load_data_by_gtype(gtype, size_param)[0]
    sp_len = nx.shortest_path_length(g, weight='d')
    print('writing to {}'.format(output_path))
    pkl.dump(sp_len,
             open(output_path, 'wb'))

if __name__ == '__main__':
    import sys
    main(sys.argv[1], sys.argv[2])
