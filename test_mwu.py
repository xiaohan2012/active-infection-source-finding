import numpy as np
import random
from fixtures import tree_and_cascade
from mwu import mwu, MAX_MU, RANDOM


def setup_module(module):
    random.seed(123456)
    np.random.seed(123456)

def run_mwu(method, tree_and_cascade):
    g, gvs, c, s, o = tree_and_cascade
    query_count = mwu(
        g, gvs,
        s, o, c, o2src_time=None,
        active_method=method,
        reward_method='dist',
        eps=0.2,
        max_iter=g.num_vertices(),
        debug=True,
        save_log=True)
    return query_count


def test_mwc_max_mu(tree_and_cascade):
    query_count, q_log, sll_log, is_nbr_log = run_mwu(
        MAX_MU, tree_and_cascade)
    assert query_count == 4
    assert len(q_log) == query_count
    assert len(q_log) == len(is_nbr_log)


def test_mwc_random(tree_and_cascade):
    query_count, q_log, sll_log, is_nbr_log = run_mwu(
        RANDOM, tree_and_cascade)
    assert query_count == 4
    assert len(q_log) == query_count
    assert len(q_log) == len(is_nbr_log)
