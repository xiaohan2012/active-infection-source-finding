import pytest
from fixtures import simulated_cascade_summary
from core import penalty_using_distribution


def test_penalty_using_distribution(simulated_cascade_summary):
    g, s2n_ps, node2id, _ = simulated_cascade_summary
    q = (0, 0)
    o = 0
    n2p = penalty_using_distribution(q, o, s2n_ps, node2id)
    for n in g.nodes_iter():
        if n == q:
            assert n2p[n] == 0
        else:
            assert n2p[n] == 1
