import pytest
from synthetic_data import load_data_by_gtype, GRID
from ic import make_partial_cascade


@pytest.fixture
def simulated_cascade_summary():
    data = load_data_by_gtype(GRID)
    node2id = data[-1]
    id2node = {i: n for n, i in node2id.items()}
    
    return data + (id2node, )


@pytest.fixture
def partial_cascade():
    g = load_data_by_gtype(GRID)[0]
    return make_partial_cascade(g, 0.05, 'late_nodes')
