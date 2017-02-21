import numpy as np
import matplotlib as mpl
import networkx as nx
from matplotlib import pyplot as plt


def plot_snapshot(g, pos,
                  node2weight,
                  ax=None,
                  query_node=None,
                  queried_nodes=None,
                  source_node=None,
                  max_node_size=1000):
    weights = node2weight.values()
    vmin, vmax = min(weights), max(weights)

    def node_color(n):
        return node2weight[n]
    
    def node_shape(n):
        if n == query_node:
            return '^'
        elif n == source_node:
            return '*'        
        elif queried_nodes and n in queried_nodes:
            return 's'
        else:
            return 'o'
    node2shape = {n: node_shape(n) for n in g.nodes()}
    
    def node_size(n):
        if n == source_node:
            return max_node_size / 2
        elif n == query_node:
            return max_node_size
        elif queried_nodes and n in queried_nodes:
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
                               with_labels=False,
                               cmap='OrRd',
                               vmin=vmin,
                               vmax=vmax)
    nx.draw_networkx_edges(g, pos=pos, ax=ax)
