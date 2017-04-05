import matplotlib as mpl
mpl.use('pdf')

import os
from matplotlib import pyplot as plt
from plot_utils import plot_source_likelihood_surface as plot_surface

param = '2-6'
graphs = ['er', 'barabasi', 'kr-hier', 'kr-peri', 'kr-rand', 'balanced-tree']
methods = ['1st', '1st_time', 'drs']
dirnames = list(map(lambda m: 'source-likelihood-{}'.format(m),
                    methods))
fig_dirname = 'source-likelihood-on-graphs-2-6'


def main(plot_type):
    nrow, ncol = len(dirnames), len(graphs)
    per_size = 5
    fig = plt.figure(figsize=(per_size * ncol, per_size * nrow))

    if plot_type.startswith('dist'):
        angle = (10, 45)
    else:
        angle = (15, 210)

    for i, (method, dirname) in enumerate(zip(methods, dirnames)):
        for j, gname in enumerate(graphs):
                
            idx = i * ncol + j + 1
            ax = fig.add_subplot(nrow, ncol, idx, projection='3d')
            plot_surface(gname, param, plot_type,
                         fig, ax=ax,
                         dirname=dirname,
                         angle=angle,
                         use_colorbar=False)
            plt.locator_params(axis='y', nbins=5)
            plt.locator_params(axis='x', nbins=5)

            if i == 0:
                ax.set_title(gname)
            if j == 0:
                ax.set_zlabel(method, size='large')

    fig.tight_layout()
    fig_dir = 'figs/{}'.format(fig_dirname)
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)
    figpath = '{}/{}.pdf'.format(fig_dir, plot_type)
    print(figpath)
    fig.savefig(figpath)

if __name__ == '__main__':
    for t in ['ratio_mean', 'ratio_median', 'dist_mean', 'dist_median']:
        main(t)
