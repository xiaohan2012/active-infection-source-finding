# coding: utf-8

# In[31]:
import matplotlib as mpl
mpl.use('pdf')

import sys
import os
import numpy as np
from matplotlib import pyplot as plt

param = '2-6'

graphs = ['balanced-tree', 'grid']
methods = [
    'exact-None',
    'exact-and',
    'order-and',
    'order-or',
    'dist-and'
]

dirnames = list(map(lambda m: 'source-likelihood-{}'.format(m),
                    methods))
output_dir = 'figs/source-likelihood-comparison-2d/'
legends = methods


def main(plot_type, dirnames, param, ps_as_y):
    """dirnames give methods
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    b = np.load('outputs/{}/{}/{}.npz'.format(dirnames[0], graphs[0], param))

    if ps_as_y:
        xs = b['arr_0'][0, :]
        zs = np.linspace(0.2, 1.0, 5)
        orig_zs = b['arr_1'][:, 0]
    else:
        xs = b['arr_1'][:, 0]
        zs = np.linspace(0.2, 1.0, 5)
        orig_zs = b['arr_0'][0, :]

    per_size, nrow, ncol = 5, len(graphs), len(zs)
    fig = plt.figure(figsize=(ncol * per_size,
                              (nrow+0.3) * per_size))

    for i, gtype in enumerate(graphs):
        bunches = [np.load('outputs/{}/{}/{}.npz'.format(dirname, gtype, param))
                   for dirname in dirnames]
        if plot_type == 'ratio_median':
            key = 'arr_2'
        elif plot_type == 'ratio_mean':
            key = 'arr_3'
        elif plot_type == 'dist_median':
            key = 'arr_4'
        elif plot_type == 'dist_mean':
            key = 'arr_5'
        elif plot_type == 'mu_median':
            key = 'arr_6'
        elif plot_type == 'mu_mean':
            key = 'arr_7'
        elif plot_type == 'rank_median':
            key = 'arr_8'
        elif plot_type == 'rank_mean':
            key = 'arr_9'
        else:
            raise ValueError('invalid plot_type')
        ms = [b[key] for b in bunches]

        for j, z in enumerate(zs):
            idx = i * ncol + j + 1
            lines = []
            ax = fig.add_subplot(nrow, ncol, idx)
            where = np.where(np.isclose(orig_zs, z))[0][0]
            for m in ms:
                if ps_as_y:
                    ys = m[:, where]
                else:
                    ys = m[where, :]
                l, = ax.plot(xs,
                             ys, 'o-')
                lines.append(l)
            if ps_as_y:
                title, xlabel = 'q={:.1f}'.format(z), 'p'
            else:
                title, xlabel = 'p={:.1f}'.format(z), 'q'
            ax.set_title(title)
            ax.set_xlabel(xlabel)

            if plot_type.startswith('ratio'):
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                
            if j == 0:
                ax.set_ylabel(gtype)
    fig.legend(lines, legends, loc='upper left')

    if ps_as_y:
        output_path = '{}/{}-by-q.pdf'.format(output_dir, plot_type)
    else:
        output_path = '{}/{}-by-p.pdf'.format(output_dir, plot_type)
    fig.tight_layout()
    fig.savefig(output_path)
    print('writing to {}'.format(output_path))

if __name__ == '__main__':
    from joblib import Parallel, delayed
    types = ['ratio_mean', 'ratio_median', 'dist_mean', 'dist_median',
             'mu_mean', 'mu_median', 'rank_mean', 'rank_median']
    Parallel(n_jobs=-1)(delayed(main)(plot_type, dirnames, param, flag)
                        for plot_type in types for flag in [True, False])
    # main('rank_mean', dirnames, param, False)
