import matplotlib as mpl
mpl.use('pdf')

import os
from matplotlib import pyplot as plt
from plot_utils import plot_source_likelihood_surface as plot_surface

params = ['2-6', '2-7', '2-8', '2-9']
# graphs = ['kr-hier', 'kr-peri', 'kr-rand', 'balanced-tree', 'er', 'barabasi']
graphs = ['kr-hier', 'kr-peri', 'kr-rand']
dirname = 'source-likelihood-1st'
fig_dirname = 'source-likelihood-on-graphs-and-sizes'


def main(plot_type):
    per_size = 5
    nrow, ncol = len(graphs), len(params)
    fig = plt.figure(figsize=(ncol * per_size, nrow * per_size))

    if plot_type.startswith('dist'):
        angle = (10, 45)
    else:
        angle = (15, 210)

    for i, gname in enumerate(graphs):
        for j, param in enumerate(params):
            idx = i * ncol + j + 1
            ax = fig.add_subplot(nrow, ncol, idx, projection='3d')
            plot_surface(gname, param, plot_type,
                         fig, ax=ax,
                         dirname=dirname,
                         angle=angle,
                         use_colorbar=False)
            ax.set_title('{}({})'.format(gname, param))
            plt.locator_params(axis='y', nbins=5)
            plt.locator_params(axis='x', nbins=5)

    fig_dir = 'figs/{}'.format(fig_dirname)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    figpath = '{}/{}.pdf'.format(fig_dir, plot_type)
    print(figpath)
    fig.savefig(figpath)

if __name__ == '__main__':
    for t in ['ratio_mean', 'ratio_median', 'dist_mean', 'dist_median']:
        main(t)
