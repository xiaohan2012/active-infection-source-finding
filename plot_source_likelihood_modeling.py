import matplotlib as mpl
mpl.use('pdf')

from matplotlib import pyplot as plt
from plot_utils import plot_source_likelihood_surface as plot_surface


graphs_and_params = [
    ('er', '2-6'),
    ('barabasi', '2-6'),
    ('kr-hier', '10-6'),
    ('kr-peri', '10-6'),
    ('kr-rand', '10-6'),
    ('balanced-tree', '2-6')
]


def main(graph_type):
    nrow, ncol = 2, 3
    fig = plt.figure(figsize=(15, 10))

    if graph_type.startswith('dist'):
        angle = (10, 45)
    else:
        angle = (15, 210)

    for i, (gname, param) in enumerate(graphs_and_params):
        ax = fig.add_subplot(nrow, ncol, i+1, projection='3d')
        plot_surface(gname, graph_type,
                     fig, ax,
                     angle=angle,
                     use_colorbar=False)
        ax.set_title(gname)
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='x', nbins=5)
    fig.savefig('figs/source-likelihood-single-obs/{}.pdf'.format(graph_type))

if __name__ == '__main__':
    for t in ['ratio_mean', 'ratio_median', 'dist_mean', 'dist_median']:
        main(t)
