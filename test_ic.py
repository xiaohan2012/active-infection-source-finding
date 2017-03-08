import pytest
import numpy as np
import networkx as nx

from numpy.testing import assert_almost_equal as aae

from ic import infection_time_estimation
from graph_generator import add_p_and_delta


@pytest.fixture
def g_det():
    g = nx.path_graph(3)
    return add_p_and_delta(g, 1.0, 1)


@pytest.fixture
def g_prob():
    g = nx.path_graph(3)
    return add_p_and_delta(g, 0.7, 1)


def test_line_det(g_det):
    m = infection_time_estimation(g_det, 10)
    assert m.shape == (3, 3, 4)
    for i in range(3):
        for j in range(3):
            expected = np.zeros(4)
            expected[int(abs(i-j))] = 1
            assert np.allclose(m[i, j, :], expected)


def test_line_prob(g_prob):
    m = infection_time_estimation(g_prob, 5000)
    assert m.shape == (3, 3, 4)
    aae(m[0, 1, :], np.array([0, 0.7, 0, 0.3]), decimal=2)
    aae(m[0, 2, :], np.array([0, 0, 0.49, 0.51]), decimal=2)
    aae(m[2, 1, :], np.array([0, 0.7, 0, 0.3]), decimal=2)
    aae(m[2, 0, :], np.array([0, 0, 0.49, 0.51]), decimal=2)
    aae(m[1, 2, :], np.array([0, 0.7, 0, 0.3]), decimal=2)
    aae(m[1, 0, :], np.array([0, 0.7, 0, 0.3]), decimal=2)

