import random
import pytest
import numpy as np
from numpy.testing import assert_array_equal
from graph_tool.all import shortest_distance, shortest_path

from steiner_tree_exact import sample_consistent_cascade, max_infection_time, \
    all_simple_paths_of_length, best_tree_sizes
from fixtures import tree_and_cascade, grid_and_cascade, setup_function
from ic import get_infection_time, gen_nontrivial_cascade
from utils import get_rank_index


def test_paths_length_tree_no_path(tree_and_cascade):
    g = tree_and_cascade[0]
    with pytest.raises(StopIteration):
        next(all_simple_paths_of_length(
            g, 5, 2, length=3,
            forbidden_nodes={},
            debug=True))


def test_paths_length_tree(tree_and_cascade):
    g = tree_and_cascade[0]
    for i in range(10):
        s, t = np.random.permutation(g.num_vertices())[:2]
        length = shortest_distance(g, s, t)
        forbidden_nodes = {}
        for p in all_simple_paths_of_length(g, s, t, length,
                                            forbidden_nodes=forbidden_nodes,
                                            debug=True):
            correct_path = [int(v) for v in shortest_path(g, g.vertex(s),
                                                          g.vertex(t))[0]]
            assert correct_path == p

    for i in range(10):
        s, t = np.random.permutation(g.num_vertices())[:2]
        length = shortest_distance(g, s, t)
        if length > 2:
            forbidden_nodes = {int(random.choice(
                shortest_path(g, g.vertex(s), g.vertex(t))[0][1:-1]))}
            with pytest.raises(StopIteration):
                next(all_simple_paths_of_length(g, s, t, length,
                                                forbidden_nodes=forbidden_nodes,
                                                debug=True))

def test_paths_length_grid(grid_and_cascade):
    g = grid_and_cascade[0]
    for i in range(10):
        s, t = np.random.permutation(g.num_vertices())[:2]
        length = shortest_distance(g, s, t)
        forbidden_nodes = {}
        for p in all_simple_paths_of_length(g, s, t, length,
                                            forbidden_nodes=forbidden_nodes,
                                            debug=True):
            assert len(p) - 1 == length, '{} != {}'.format(len(p)-1, length)
            for u, v in zip(p[:-1], p[1:]):
                assert g.edge(u, v) is not None
                for u in p:
                    assert u not in forbidden_nodes
            assert p[0] == s
            assert p[-1] == t


def test_sample_consistent_cascade_tree(tree_and_cascade):
    g, _, infection_times, source, obs_nodes = tree_and_cascade
    if False:
        # for true source
        # DEBUGGING purpose
        gv = sample_consistent_cascade(g, obs_nodes, source, infection_times, debug=True)
        assert gv is not None
        ts_max = max_infection_time(g, infection_times, obs_nodes, source, debug=True)
        pred_inf_time = get_infection_time(gv, source)
        pred_inf_time += ts_max
        assert_array_equal(pred_inf_time[obs_nodes], infection_times[obs_nodes])
    else:
        # for other nodes as sources
        count = 0
        for cand_source in g.vertices():
            gv = sample_consistent_cascade(g, obs_nodes, cand_source, infection_times, debug=True)
            ts_max = max_infection_time(g, infection_times, obs_nodes, cand_source, debug=True)

            if cand_source == source:
                assert gv is not None

            if gv is not None:
                pred_inf_time = get_infection_time(gv, cand_source)
                pred_inf_time += ts_max
                assert_array_equal(pred_inf_time[obs_nodes], infection_times[obs_nodes])
                count += 1
        print('{} / {} are possible'.format(count, g.num_vertices()))


def test_sample_consistent_cascade_grid(grid_and_cascade):
    g, _, infection_times, source, obs_nodes = grid_and_cascade
    if False:
        # for true source
        # DEBUGGING purpose
        gv = sample_consistent_cascade(g, obs_nodes, source, infection_times, debug=True)
        assert gv is not None
        ts_max = max_infection_time(g, infection_times, obs_nodes, source, debug=True)
        pred_inf_time = get_infection_time(gv, source)
        pred_inf_time += ts_max
        assert_array_equal(pred_inf_time[obs_nodes], infection_times[obs_nodes])
    else:
        # for other nodes as sources
        count = 0
        for cand_source in g.vertices():
            print('source: {}'.format(source))
            gv = sample_consistent_cascade(g, obs_nodes, cand_source, infection_times, debug=True)
            ts_max = max_infection_time(g, infection_times, obs_nodes, cand_source, debug=True)

            if cand_source == source:
                assert gv is not None

            if gv is not None:
                pred_inf_time = get_infection_time(gv, cand_source)
                pred_inf_time += ts_max
                assert_array_equal(pred_inf_time[obs_nodes], infection_times[obs_nodes])
                count += 1
        print('{} / {} are possible'.format(count, g.num_vertices()))
        

def test_best_tree_sizes_tree(tree_and_cascade):
    g, _, infection_times, source, obs_nodes = tree_and_cascade
    scores = best_tree_sizes(g, obs_nodes, infection_times)
    print('|possible_nodes|={}'.format(np.sum(np.invert(np.isinf(scores)))))
    print(scores[source], scores.min())
    assert get_rank_index(scores, source) <= 3


def test_best_tree_sizes_grid(grid_and_cascade):
    g, _, infection_times, source, obs_nodes = grid_and_cascade
    scores = best_tree_sizes(g, obs_nodes, infection_times)
    assert get_rank_index(scores, source) <= 1


def test_full_observation_tree(tree_and_cascade):
    g = tree_and_cascade[0]
    for p in np.arange(0.2, 1.0, 0.1):
        infection_times, source, obs_nodes = gen_nontrivial_cascade(g, p, 1.0)
        scores = best_tree_sizes(g, obs_nodes, infection_times)
        assert get_rank_index(scores, source) == 0


def test_full_observation_grid(grid_and_cascade):
    g = grid_and_cascade[0]
    for p in np.arange(0.5, 1.0, 0.1):
        infection_times, source, obs_nodes = gen_nontrivial_cascade(g, p, 1.0)
        scores = best_tree_sizes(g, obs_nodes, infection_times)
        assert get_rank_index(scores, source) == 0
