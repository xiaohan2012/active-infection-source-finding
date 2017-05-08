import pytest
import networkx as nx
from synthetic_data import load_data_by_gtype, GRID, PL_TREE
from graph_generator import add_p_and_delta
from ic import gen_nontrivial_cascade, get_gvs
from graph_tool.all import load_graph


@pytest.fixture
def simulated_cascade_summary():
    return load_data_by_gtype(GRID, '10')


@pytest.fixture
def partial_cascade():
    g = load_data_by_gtype(GRID, '10')[0]
    return make_partial_cascade(g, 0.05, 'late_nodes')


@pytest.fixture
def tree_infection():
    g = load_data_by_gtype(PL_TREE, '100')[0]
    return g, make_partial_cascade(g, 0.05, 'late_nodes')


@pytest.fixture
def line_infection():
    g = nx.path_graph(100)
    g = add_p_and_delta(g, 0.7, 1)
    return (g, ) + make_partial_cascade(g, 0.01, 'late_nodes')


@pytest.fixture
def tree_and_cascade():
    g = load_graph('data/balanced-tree/2-6/graph.gt')
    c, s, o = gen_nontrivial_cascade(g, 0.8, 0.2)
    return g, get_gvs(g, 0.5, 100), c, s, o


@pytest.fixture
def grid_and_cascade():
    g = load_graph('data/grid/2-6/graph.gt')
    c, s, o = gen_nontrivial_cascade(g, 0.8, 0.2)
    return g, get_gvs(g, 0.5, 100), c, s, o


def setup_module(module):
    import random
    import numpy as np
    seed = 123456
    random.seed(seed)
    np.random.seed(seed)
