import numpy as np
from utils import get_rank_index


def test_get_rank_index():
    array = [0, 0, 0, 1, 0, 0]
    id_ = 2
    assert np.where(np.argsort(array)[::-1] == id_)[0][0] != 1
    assert get_rank_index(array, id_) == 1

