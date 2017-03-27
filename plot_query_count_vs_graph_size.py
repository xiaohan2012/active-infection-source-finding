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
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    methods, x = df.index.levels
    methods, x = methods.tolist(), x.tolist()
    for m in sorted(methods):
        if gtype == PL_TREE and m == 'random':
            continue

        y = df['50%'][m].fillna(0).tolist()
        lb = df['25%'][m].fillna(0).tolist()
        ub = df['75%'][m].fillna(0).tolist()

        # ax.errorbar(x, y, yerr=np.array([lb, ub]))
        ax.plot(x, y, markersize=12)
        ax.set_xscale("log", nonposx='clip', basex=base)
        ax.set_xlim(np.min(list(x))-1, np.max(list(x))+1)
        ax.set_xlabel('graph size')
        ax.set_ylabel('query count')
        ax.set_title('median query count vs graph size on {} graphs'.format(gtype))
        ax.legend(methods, loc='upper left')

    output_dir = 'figs/{}'.format(gtype)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(output_dir)
    fig.savefig(os.path.join(output_dir, 'query_count_vs_graph_size.pdf'))


if __name__ == '__main__':
    import sys
    main(sys.argv[1])
