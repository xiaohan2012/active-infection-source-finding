import matplotlib
matplotlib.use('pdf')

import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def main(gtype, base=2):
    df = pd.read_pickle('data/{}/performance.pkl'.format(gtype))
    fig, ax = plt.subplots(1, 1)
    methods, x = df.index.levels
    for m in methods:
        y = df['mean'][m]
        lb = df['25%'][m]
        ub = df['75%'][m]
        ax.errorbar(x, y, yerr=np.array([lb, ub]), fmt='o-')
        ax.set_xscale("log", nonposx='clip', basex=base)
        ax.set_xlim(np.min(list(x))-1, np.max(list(x))+1)
        ax.set_xlabel('graph size')
        ax.set_ylabel('query count')
        ax.set_title('mean query count vs graph size on {} graphs'.format(gtype))
        ax.legend(methods, loc='upper left')

    output_dir = 'figs/{}'.format(gtype)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)
    fig.savefig(os.path.join(output_dir, 'query_count_vs_graph_size.pdf'))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
