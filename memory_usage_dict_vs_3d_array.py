import numpy as np
import pickle as pkl
from collections import defaultdict
from scipy.sparse import csr_matrix
from tqdm import tqdm


@profile
def main():
    dirname = 'data/grid/2-5'
    m = pkl.load(open('{}/inf_time_proba_matrix.pkl'.format(dirname), 'rb'))
    d = dict()
    nr, nc, nt = m.shape
    for i in tqdm(np.arange(nr)):
        row, col = np.nonzero(m[i, :, :])
        data = m[i, row, col]
        d[i] = csr_matrix((data, (row, col)), shape=(nc, nt))

    if False:
        for i in tqdm(np.arange(nr)):
            for j in range(nc):
                for k in range(nt):
                    assert m[i, j, k] == d[i][j, k]
    pkl.dump(d, open('{}/s2n_probas_sp.pkl'.format(dirname), 'wb'))
    del m
    del d
    print('done')
# if __name__ == "__main__":
main()

