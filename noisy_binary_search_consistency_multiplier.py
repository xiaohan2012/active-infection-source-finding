import matplotlib
matplotlib.use('pdf')

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from synthetic_data import load_data_by_gtype
from experiment_utils import experiment_noisy_bs_n_rounds


gtypes = [('grid', '2-4'),
          ('er', '2-8'),
          ('barabasi', '2-8'),
          ('pl-tree', '2-8'),
          ('kr-hier', '10-10'),
          ('kr-peri', '10-10'),
          ('kr-rand', '10-10')]


for gtype, param in gtypes:
    print(gtype)
    g = load_data_by_gtype(gtype, param)[0]
    multipliers = [0.6, 0.7, 0.8, 0.9]
    means = []
    medians = []
    for mtp in multipliers:
        print(mtp)
        cnts = experiment_noisy_bs_n_rounds(g, 100, mtp)
        s = pd.Series(cnts).describe()
        means.append(s['mean'])
        medians.append(s['50%'])
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].plot(multipliers, means, 'o-')
    ax[0].set_title('Mean')
    ax[0].set_xlabel('consistency_multiplier')
    ax[0].set_ylabel('query count mean')

    ax[1].plot(multipliers, medians, 'o-')
    ax[1].set_title('Median')
    ax[1].set_xlabel('consistency_multiplier')
    ax[1].set_ylabel('query count median')
    fig.savefig('figs/noisy_binary_search_consistency_multiplier/{}.png'.format(gtype))
