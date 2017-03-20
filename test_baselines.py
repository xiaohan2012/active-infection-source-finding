import pytest
import numpy as np
import networkx as nx
from numpy.testing import assert_almost_equal as aae

from fixtures import line_infection, tree_infection
from baselines import random_dog


def test_random_dog(line_infection):
    g, _, _, _, _ = line_infection
    nodes = list(sorted(g.nodes()))
    infection_times = {n: n for n in nodes}
    # source = nodes[0]
    obs_nodes = {nodes[-1]}
    for fraction in np.arange(0, 1, 0.25):
        query_count, query_node_list = random_dog(g, obs_nodes, infection_times,
                                                  fraction, debug=True,
                                                  save_logs=True)
        assert query_count == g.number_of_nodes() - 1
        assert set(query_node_list) == set(range(0, 100))


def test_random_dog(tree_infection):
    """ just make sure it runs
    """
    g, (source, obs_nodes, infection_times, tree) = tree_infection

    for fraction in np.arange(0, 1, 0.25):
        query_count, query_node_list = random_dog(g, obs_nodes, infection_times,
                                                  fraction, debug=True,
                                                  save_logs=True)
