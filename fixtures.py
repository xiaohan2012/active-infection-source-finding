import pytest
from synthetic_data import load_data_by_gtype, GRID
from ic import make_partial_cascade


@pytest.fixture
def simulated_cascade_summary():
    return load_data_by_gtype(GRID)


@pytest.fixture
def partial_cascade():
    g = load_data_by_gtype(GRID)[0]
    return make_partial_cascade(g, 0.05, 'late_nodes')
