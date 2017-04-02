import matplotlib
matplotlib.use('pdf')

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from synthetic_data import PL_TREE
from plot_utils import richify_line_style


def main(gtype, base=2):
    richify_line_style(plt)
    df = pd.read_pickle('data/{}/performance.pkl'.format(gtype))
    fig, ax = plt.subplots(1, 2, figsize=(20, 8))
    methods, x = df.index.levels
    methods, x = methods.tolist(), x.tolist()
    xaxis_len = min(len(df['50%'][m].tolist()) for m in sorted(methods))

    if gtype.startswith('kr-'):  # HACK
        x = np.power(2, np.arange(8, 13))

    x = x[:xaxis_len]
    methods = [m for m in methods
               if (not m.startswith('dog')) or
               (m.startswith('dog') and
                ('0.00' in m or '1.00' in m))]
    methods = ['dog-0.00', 'dog-1.00', 'max_mu', 'rand_max_mu', 'random', 'noisy_bs']
    for m in methods:
        if gtype == PL_TREE and m == 'random':
            continue
        if m not in df['50%']:
            continue
        medians = df['50%'][m].tolist()[:xaxis_len]
        means = df['mean'][m].tolist()[:xaxis_len]
        # print(x)
        # ax.errorbar(x, y, yerr=np.array([lb, ub]))
        ax[0].plot(x, medians, markersize=12)
        ax[1].plot(x, means, markersize=12)
    ax[0].set_xscale("log", nonposx='clip', basex=base)
    ax[0].set_xlim(np.min(list(x))-1, np.max(list(x))+1)
    ax[0].set_xlabel('graph size')
    ax[0].set_ylabel('query count')
    ax[0].set_title('median query count vs graph size on {} graphs'.format(gtype))
    ax[0].legend(methods, loc='upper left')

    ax[1].set_xscale("log", nonposx='clip', basex=base)
    ax[1].set_xlim(np.min(list(x))-1, np.max(list(x))+1)
    ax[1].set_xlabel('graph size')
    ax[1].set_ylabel('query count')
    ax[1].set_title('mean query count vs graph size on {} graphs'.format(gtype))
    ax[1].legend(methods, loc='upper left')

    output_dir = 'figs/{}'.format(gtype)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)
    fig.savefig(os.path.join(output_dir, 'query_count_vs_graph_size.pdf'))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
