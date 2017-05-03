import numpy as np
from fixtures import tree_and_cascade
from mwu import mwu, MAX_MU


def setup_module(module):
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
        debug=True)
    return query_count


def test_mwc_max_mu(tree_and_cascade):
    query_count = run_mwu(
        MAX_MU, tree_and_cascade)
    assert query_count == 1
