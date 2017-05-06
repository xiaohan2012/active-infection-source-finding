import numpy as np
import math
import matplotlib as mpl
import networkx as nx
from matplotlib import pyplot as plt, cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import axes3d
from cycler import cycler
from utils import infeciton_time2weight


def plot_snapshot(g, pos,
                  node2weight,
                  is_infection_time=True,
                  ax=None,
                  query_node=None,
                  queried_nodes=None,
                  source_node=None,
                  max_node_size=1000,
                  with_labels=False):
    if is_infection_time:
        node2weight = infeciton_time2weight(node2weight)

    weights = node2weight
    vmin, vmax = min(weights), max(weights)

    def node_color(n):
        return node2weight[n]
    
    def node_shape(n):
        if n == query_node:
            return '^'
        elif n == source_node:
            return '*'        
        elif len(queried_nodes) > 0 and n in queried_nodes:
            return 's'
        else:
            return 'o'
    node2shape = {n: node_shape(n) for n in g.nodes()}
    
    def node_size(n):
        if n == source_node:
            return max_node_size / 2
        elif n == query_node:
            return max_node_size
        elif len(queried_nodes) > 0 and n in queried_nodes:
            return max_node_size / 4
        elif node2weight[n] != 0:
            return max_node_size / 4
        else:
            return max_node_size / 10
    
    # draw by shape
    if ax is None:
        ax = None
    all_shapes = set(node2shape.values())
    for shape in all_shapes:
        nodes = [n for n in g.nodes() if node2shape[n] == shape]
        nx.draw_networkx_nodes(g, pos=pos,
                               ax=ax,
                               node_shape=shape,
                               node_color=list(map(node_color, nodes)),
                               node_size=list(map(node_size, nodes)),
                               nodelist=nodes,
                               with_labels=with_labels,
                               cmap='OrRd',
                               vmin=vmin,
                               vmax=vmax)
    nx.draw_networkx_edges(g, pos=pos, ax=ax)


def add_colorbar(cvalues, cmap='OrRd', ax=None):
    eps = np.maximum(0.0000000001, np.min(cvalues)/1000.)
    vmin = np.min(cvalues) - eps
    vmax = np.max(cvalues)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scm = mpl.cm.ScalarMappable(norm, cmap)
    scm.set_array(cvalues)
    if ax is None:
        plt.colorbar(scm)
    else:
        ax.colorbar(scm)


def plot_query_process(g, source, obs_nodes, infection_times,
                       mu_list, query_list,
                       check_neighbor_threshold,
                       pos,
                       ncols=2, width=12, max_plots=10,
                       max_node_size=750):
    """
    mu_list: the \mu *after* the ith iteration.
    query_list: sequence of queries (including observed nodes)
    """
    nrows = math.ceil((max_plots+2) / ncols)
    height = int(width * nrows / ncols)
        
    fig, ax = plt.subplots(nrows, ncols,
                           sharex=True, sharey=True,
                           figsize=(width, height))
    node2weight = infeciton_time2weight(infection_times)
    plot_snapshot(g, pos, node2weight, source_node=source,
                  queried_nodes=obs_nodes, ax=ax[0, 0],
                  max_node_size=max_node_size)
    ax[0, 0].set_title('ground truth')

    i, j = 0, 1

    plot_snapshot(g, pos, mu_list[len(obs_nodes)-1],
                  source_node=source, queried_nodes=obs_nodes,
                  ax=ax[i, j], max_node_size=max_node_size)
    ax[i, j].set_title('using observation')

    # truncate the list to only queries by our algorithm
    mu_list = mu_list[len(obs_nodes):]
    query_list = query_list[len(obs_nodes):]

    # query by the algorithm
    for iter_i, mu, query in zip(range(max_plots), mu_list, query_list):
        i, j = int((iter_i+2) / ncols), (iter_i+2) % ncols
        plot_snapshot(g, pos, mu, query_node=query,
                      source_node=source,
                      queried_nodes=set(obs_nodes) | set(query_list[:iter_i]),
                      ax=ax[i, j],
                      max_node_size=max_node_size)
        ax[i, j].set_title('iter_i={} mu(source)={:.5f} ({})'.format(iter_i,
                                                                     mu[source],
                                                                     check_neighbor_threshold))
    return fig, ax


def richify_line_style(plt):
    plt.style.use('fivethirtyeight')
    plt.rc('axes',
           prop_cycle=(
               cycler('color', ['r', 'r', 'g', 'g', 'g', 'b']) +
               cycler('linestyle', ['-', '--', ':', '-.', '--', '-']) +
               cycler('marker', ['o', 'v', 's', '*', 'o', '*'])
           ))


def plot_source_likelihood_surface(
        gtype, param, plot_type,
        fig, dirname,
        ax=None,
        angle=(15, 210), use_colorbar=True):
    bunch = np.load('outputs/{}/{}/{}.npz'.format(dirname, gtype, param))
    X, Y = bunch['arr_0'], bunch['arr_1']
    if plot_type == 'ratio_median':
        Z = bunch['arr_2']
    elif plot_type == 'ratio_mean':
        Z = bunch['arr_3']
    elif plot_type == 'dist_median':
        Z = bunch['arr_4']
    elif plot_type == 'dist_mean':
        Z = bunch['arr_5']
    elif plot_type == 'mu_median':
        Z = bunch['arr_6']
    elif plot_type == 'mu_mean':
        Z = bunch['arr_7']
    elif plot_type == 'rank_median':
        Z = bunch['arr_8']
    elif plot_type == 'rank_mean':
        Z = bunch['arr_9']
    else:
        raise ValueError('invalid plot_type')

    if ax is None:
        ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, np.transpose(Z),
                           rstride=1, cstride=1,
                           vmin=0, vmax=1,
                           cmap=cm.coolwarm)
    ax.set_zlim(0, Z.max())
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    if use_colorbar:
        fig.colorbar(surf, shrink=0.5, aspect=5)

    ax.set_xlabel('q')
    ax.set_ylabel('p')
    # ax.set_zlabel(plot_type)
    ax.view_init(*angle)
    return fig, ax
