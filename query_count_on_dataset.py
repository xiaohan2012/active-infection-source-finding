# coding: utf-8
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from noisy_binary_search import NOISY_BINARY_SEARCH
from synthetic_data import (load_data_by_gtype, add_p_and_delta)
from experiment_utils import (experiment_dog_multiple_rounds,
                              experiment_noisy_bs_n_rounds,
                              counts_to_stat)


def main(dataset,
         p=0.7,
         n_rounds=100,
         fraction=0.01,
         consistency_multiplier=0.9,
         sampling_method='late_nodes'):
    dumps = load_data_by_gtype(dataset, '')
    g, sp_len = dumps[0], dumps[4]
    g = add_p_and_delta(g, p, 1)
    print('|V|={}'.format(g.number_of_nodes()))
    print('|E|={}'.format(g.number_of_edges()))
    rows = []
    index = []

    # binary search
    counts = experiment_noisy_bs_n_rounds(
        g, sp_len,
        n_rounds, consistency_multiplier,
        parallelize=False)
    rows.append(counts_to_stat(counts))
    index.append(NOISY_BINARY_SEARCH)
    
    # dog
    for f in tqdm(np.linspace(0, 1, 5)):
        counts = experiment_dog_multiple_rounds(n_rounds, g, fraction, sampling_method,
                                                query_fraction=f)
        rows.append(counts_to_stat(counts))
        index.append('dog-{:.2f}'.format(f))

    df = pd.DataFrame.from_records(rows, index=index)
    df.to_pickle('data/{}/performance.pkl'.format(dataset))

if __name__ == '__main__':
    dataset = sys.argv[1]
    main(dataset)
