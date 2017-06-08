import pytest
import numpy as np
from graph_tool import load_graph
from ctic import gen_cascade as ctic_gen
from feasibility import is_arborescence


@pytest.fixture
def g():
    return load_graph('data/{}/2-6/graph.gt'.format('grid'))


def _test_result(g, source, infection_times, tree, stop_fraction):
    assert infection_times[source] == 0
    assert is_arborescence(tree)
    np.testing.assert_almost_equal(np.count_nonzero(infection_times != -1) / g.num_vertices(),
                                   stop_fraction, decimal=1)


def test_ctic(g):
    for stop_fraction in np.arange(0.1, 1.0, 0.1):
        for i in range(10):
            source, infection_times, tree = ctic_gen(
                g, 1.0, source=None,
                stop_fraction=stop_fraction, return_tree=True)

            _test_result(g, source, infection_times, tree, stop_fraction)
