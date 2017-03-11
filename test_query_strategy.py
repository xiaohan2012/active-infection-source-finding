import pytest
import numpy as np
from query_strategy import expected_infection_time, override_exp_times_by_obs
from synthetic_data import load_data_by_gtype, GRID
from ic import make_partial_cascade


@pytest.fixture
def data():
    return load_data_by_gtype(GRID, '10')


@pytest.fixture
def mu(data):
    g = data[0]
    mu = np.zeros(g.number_of_nodes())
    mu[0] = 0.5
    mu[1] = 0.5
    return mu
