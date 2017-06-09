import numpy as np
import pytest
from graph_tool.all import load_graph

from steiner_tree_greedy import steiner_tree_greedy
from steiner_tree_mst import steiner_tree_mst, build_closure
from steiner_tree import get_steiner_tree

from mst_truncated import build_truncated_closure
from temporal_bfs import temporal_bfs
from utils import earliest_obs_node
from feasibility import is_feasible
from cascade import gen_nontrivial_cascade

QS = np.linspace(0.1, 1.0, 10)
MODELS = ['si', 'ic', 'sp']

P = 0.5
K = 10


@pytest.fixture
def cascades_on_tree():
    cascades = []
    g = load_graph('data/grid/2-6/graph.gt')
    for model in MODELS:
        for q in QS:
            for i in range(K):
                ret = gen_nontrivial_cascade(
                    g, P, q, model=model, return_tree=True,
                    source_includable=True)
                ret = (g, ) + ret + (model, q, i)
                cascades.append(ret)  # g, infection_times, source, obs_nodes, true_tree
    return cascades
        

def test_greedy(cascades_on_tree):
    for g, infection_times, source, obs_nodes, true_tree, model, q, i in cascades_on_tree:
        print(model, q, i)
        root = earliest_obs_node(obs_nodes, infection_times)
        tree = steiner_tree_greedy(
            g, root, infection_times, source, obs_nodes,
            debug=False,
            verbose=True
        )
        assert is_feasible(tree, root, obs_nodes, infection_times)


def test_mst(cascades_on_tree):
    for g, infection_times, source, obs_nodes, true_tree, model, q, i in cascades_on_tree:
        print(model, q, i)
        root = earliest_obs_node(obs_nodes, infection_times)
        tree = steiner_tree_mst(
            g, root, infection_times, source, obs_nodes,
            closure_builder=build_closure,
            strictly_smaller=False,
            debug=False,
            verbose=False,
        )
        assert is_feasible(tree, root, obs_nodes, infection_times)


def test_mst_truncated(cascades_on_tree):
    for g, infection_times, source, obs_nodes, true_tree, model, q, i in cascades_on_tree:
        print(model, q, i)
        root = earliest_obs_node(obs_nodes, infection_times)
        tree = steiner_tree_mst(
            g, root, infection_times, source, obs_nodes,
            closure_builder=build_truncated_closure,
            k=1,
            debug=False,
            verbose=False,
        )
        assert is_feasible(tree, root, obs_nodes, infection_times)


def test_temporal_bfs(cascades_on_tree):
    for g, infection_times, source, obs_nodes, true_tree, model, q, i in cascades_on_tree:
        print(model, q, i)
        root = earliest_obs_node(obs_nodes, infection_times)
        tree = temporal_bfs(
            g, root, infection_times, source, obs_nodes,
            debug=False,
            verbose=False,
        )
        assert is_feasible(tree, root, obs_nodes, infection_times)


def test_vanilla_steiner_tree(cascades_on_tree):
    for g, infection_times, source, obs_nodes, true_tree, model, q, i in cascades_on_tree:
        print(model, q, i)
        pred_tree = get_steiner_tree(
            g, obs_nodes,
            debug=False,
            verbose=False,
        )
        # it's undirected, so test is a bit different
        assert np.all(np.array([v.out_degree() for v in pred_tree.vertices()]) > 0)
        assert np.sum([v.out_degree() for v in pred_tree.vertices()]) == (pred_tree.num_vertices() * 2)
        for o in obs_nodes:
            pred_tree.vertex(o)
