import pytest
from synthetic_data import load_data_by_gtype, GRID


@pytest.fixture
def simulated_cascade_summary():
    data = load_data_by_gtype(GRID)
    node2id = data[-1]
    id2node = {i: n for n, i in node2id.items()}
    
    return data + (id2node, )
