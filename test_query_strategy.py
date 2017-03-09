import pytest
import numpy as np
from query_strategy import expected_infection_time, override_exp_times_by_obs
from synthetic_data import load_data_by_gtype, GRID
from ic import make_partial_cascade


@pytest.fixture
def data():
    return load_data_by_gtype(GRID)


@pytest.fixture
def mu(data):
    g = data[0]
    mu = np.zeros(g.number_of_nodes())
    mu[0] = 0.5
    mu[1] = 0.5
    return mu


def test_expected_infection_time(data, mu):
    g, probas, node2id = data

    exp_probas = expected_infection_time(mu, probas)
    np.allclose(exp_probas, (probas[0, :, :] + probas[1, :, :]) / 2)


def test_override_exp_times_by_obs(data, mu):
    g, probas, node2id = data
    exp_times = expected_infection_time(mu, probas)
    _, obs_nodes, infection_times, _ = make_partial_cascade(
        g, fraction=0.05, sampling_method='late_nodes'
    )
    exp_times = override_exp_times_by_obs(exp_times, obs_nodes, node2id, infection_times)
    for n in obs_nodes:
        i = node2id[n]
        nz_ids = exp_times[i, :].nonzero()[0]
        assert len(nz_ids) == 1
        assert exp_times[i, infection_times[n]] == 1
