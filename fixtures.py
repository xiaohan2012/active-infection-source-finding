import pytest
from synthetic_data import load_data_by_gtype, GRID, PL_TREE
from ic import make_partial_cascade


@pytest.fixture
def simulated_cascade_summary():
    return load_data_by_gtype(GRID, '10')


@pytest.fixture
def partial_cascade():
    g = load_data_by_gtype(GRID, '10')[0]
    return make_partial_cascade(g, 0.05, 'late_nodes')


@pytest.fixture
def tree_infection():
    g = load_data_by_gtype(PL_TREE, '100')[0]
    return g, make_partial_cascade(g, 0.05, 'late_nodes')
