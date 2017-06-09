import pytest
from sorted_array import insert, remove

@pytest.fixture
def seq():
    return ['a', 'b']


@pytest.fixture
def keys():
    return [1, 2]

@pytest.fixture
def key_func():
    mapping = {chr(i+96): i for i in range(1, 27)}
    return mapping.__getitem__


def test_all(seq, keys, key_func):
    insert(seq, keys, 'c', key_func)
    insert(seq, keys, 'a', key_func)
    assert seq == ['a', 'a', 'b', 'c']
    assert keys == [1, 1, 2, 3]

    remove(seq, keys, 'a', key_func)
    assert seq == ['a', 'b', 'c']
    assert keys == [1, 2, 3]

    remove(seq, keys, 'c', key_func)
    assert seq == ['a', 'b']
    assert keys == [1, 2]

    remove(seq, keys, 'a', key_func)
    assert seq == ['b']
    assert keys == [2]

    remove(seq, keys, 'b', key_func)
    assert seq == []
    assert keys == []
