import pytest
import numpy as np
import networkx as nx

from numpy.testing import assert_almost_equal as aae
from graph_tool.all import shortest_distance

from ic import simulate_cascade, MAXINT
from fixtures import grid_and_cascade, setup_function


def test_simulate_cascade(grid_and_cascade):
    g = grid_and_cascade[0]
    for p in np.arange(0.2, 1.0, 0.1):
        source, times, tree = simulate_cascade(
            g, p, source=None, return_tree=True)
        dist = shortest_distance(tree, source=source).a
        dist[dist == MAXINT] = -1
        aae(dist, times)
    
