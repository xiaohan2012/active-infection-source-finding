"""continous-time independent cascade model"""
import numpy as np
import random
from graph_tool.all import shortest_distance

from utils import edges2graph
from feasibility import is_arborescence


def gen_cascade(g, scale=1.0, source=None, stop_fraction=0.5, return_tree=True):
    rands = np.random.exponential(scale, g.num_edges())
    delays = g.new_edge_property('float')
    delays.set_2d_array(rands)

    if source is None:
        source = random.choice(np.arange(g.num_vertices()))

    dist, pred = shortest_distance(g, source=source, weights=delays, pred_map=True)

    q = stop_fraction * 100
    percentile = np.percentile(dist.a, q)
    infected_nodes = np.nonzero(dist.a <= percentile)[0]
    uninfected_nodes = np.nonzero(dist.a > percentile)[0]

    infection_times = np.array(dist.a)
    infection_times[uninfected_nodes] = -1

    rets = (source, infection_times)
    if return_tree:
        tree_edges = set()
        for n in infected_nodes:
            c = n
            while pred[c] != c:
                edge = ((pred[c], c))
                if edge not in tree_edges:
                    tree_edges.add(edge)
                else:
                    break
        tree = edges2graph(g, tree_edges)
        rets += (tree, )
    return rets
