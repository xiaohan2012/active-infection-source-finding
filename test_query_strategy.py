import pytest
import numpy as np
from query_strategy import expected_infection_time
from synthetic_data import load_data_by_gtype, GRID


@pytest.fixture
def data():
    return load_data_by_gtype(GRID)


def test_expected_infection_time(data):
    g, probas, node2id = data
    mu = np.zeros(g.number_of_nodes())
    mu[0] = 0.5
    mu[1] = 0.5
    exp_probas = expected_infection_time(mu, probas)
    np.allclose(exp_probas, (probas[0, :, :] + probas[1, :, :]) / 2)
