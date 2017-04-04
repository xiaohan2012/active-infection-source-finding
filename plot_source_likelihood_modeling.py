import matplotlib as mpl
mpl.use('pdf')

import os
from matplotlib import pyplot as plt
from plot_utils import plot_source_likelihood_surface as plot_surface

param = '2-6'
graphs = ['er', 'barabasi', 'kr-hier', 'kr-peri', 'kr-rand', 'balanced-tree']
dirname = 'source-likelihood-2nd'
fig_dirname = 'source-likelihood-on-graphs-2nd-2-6'


def main(plot_type):
    nrow, ncol = 2, 3
    fig = plt.figure(figsize=(15, 10))

    if plot_type.startswith('dist'):
        angle = (10, 45)
    else:
        angle = (15, 210)

    for i, gname in enumerate(graphs):
        ax = fig.add_subplot(nrow, ncol, i+1, projection='3d')
        plot_surface(gname, param, plot_type,
                     fig, ax=ax,
                     dirname=dirname,
                     angle=angle,
                     use_colorbar=False)
        ax.set_title(gname)
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
