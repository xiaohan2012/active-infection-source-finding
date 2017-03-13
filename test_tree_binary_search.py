import pytest
import numpy as np
import networkx as nx
from tree_binary_search import subtree_size, find_centroid, find_source
from fixtures import tree_infection


@pytest.fixture
def line_graph():
    return nx.path_graph(5)


@pytest.fixture
def star():
    return nx.star_graph(5)


def test_subtree_size(line_graph):
    cache = {}
    assert subtree_size(line_graph, 0, 1, cache) == 1
    assert subtree_size(line_graph, 1, 0, cache) == 4
    assert subtree_size(line_graph, 1, 2, cache) == 2
    assert subtree_size(line_graph, 2, 3, cache) == 3
    assert subtree_size(line_graph, 3, 4, cache) == 4


def test_subtree_size_cache(line_graph):
    cache = {}
    for u, v in line_graph.edges_iter():
        subtree_size(line_graph, u, v, cache)
        subtree_size(line_graph, v, u, cache)
    assert cache == {(0, 1): 1, (1, 0): 4,
                     (1, 2): 2, (2, 1): 3,
                     (2, 3): 3, (3, 2): 2,
                     (3, 4): 4, (4, 3): 1}

    
def test_subtree_size_star(star):
    cache = {}
    for u, v in star.edges_iter():
        subtree_size(star, u, v, cache)
        subtree_size(star, v, u, cache)

    assert cache == {(0, 1): 5, (1, 0): 1,
                     (0, 2): 5, (2, 0): 1,
                     (0, 3): 5, (3, 0): 1,
                     (0, 4): 5, (4, 0): 1,
                     (0, 5): 5, (5, 0): 1}
    

def test_find_centroid():
    g = nx.path_graph(10)
    assert find_centroid(g) == 4


def test_find_centroid_star(star):
    assert find_centroid(star) == 0


def test_find_source_line_1():
    g = nx.path_graph(10)
    source = 1
    obs_nodes = {5}
    infection_times = {0: 1, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4}
    query_count = find_source(g, obs_nodes, infection_times)
    assert query_count == 5


def test_find_source_line_2():
    g = nx.path_graph(10)
    source = 1
    obs_nodes = {0}
    infection_times = {0: 1, 1: 0, 2: 1, 3: float('inf'), 4: float('inf')}
    query_count = find_source(g, obs_nodes, infection_times)
    assert query_count == 3


def test_find_source(tree_infection):
    np.random.seed(123456)
    for i in range(100):
        g, (source, obs_nodes, infection_times, tree) = tree_infection
        query_count = find_source(g, obs_nodes, infection_times)
        max_query_count = np.log(g.number_of_nodes()) * max(list(g.degree().values()))
        assert query_count <= max_query_count
