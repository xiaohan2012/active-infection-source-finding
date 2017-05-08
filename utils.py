import numpy as np
import networkx as nx
from collections import Counter
from functools import reduce



# @profile
def generalized_jaccard_similarity(a, b):
    """a and b should be list
    >>> a = [1, 1, 2]
    >>> b = [1, 2, 3]
    >>> generalized_jaccard_similarity(a, b)
    0.5
    >>> generalized_jaccard_similarity(a, a)
    1.0
    """
    # a, b = map(list, [a, b])
    if not isinstance(a, Counter):
        a = Counter(a)

    if not isinstance(b, Counter):
        b = Counter(b)

    all_elements = set(a.keys()) | set(b.keys())

    # method 0
    # numer, denom = 0, 0
    # for e in all_elements:
    #     x, y = a.get(e, 0), b.get(e, 0)
    #     if x < y:
    #         numer += x
    #         denom += y
    #     else:
    #         numer += y
    #         denom += x

    # method 1
    numer, denom = reduce(lambda v, tpl: (v[0] + tpl[0], v[1] + tpl[1]),
                          ((b.get(e, 0), a.get(e, 0))
                           if a.get(e, 0) > b.get(e, 0)
                           else (a.get(e, 0), b.get(e, 0))
                           for e in all_elements),
                          (0, 0))
    
    # method 2
    # sorted_count = np.sort(counts, axis=1, kind='heapsort')
    # numer, denom = sorted_count.sum(axis=0)
    # numer = np.sum(np.min(counts, axis=1))
    # denom = np.sum(np.max(counts, axis=1))
    return numer / denom


def generalized_jaccard_distance(a, b):
    return 1 - generalized_jaccard_similarity(a, b)


# @profile
def weighted_sample_with_replacement(pool, weights, N):
    assert len(pool) == len(weights)
    np.testing.assert_almost_equal(np.sum(weights), 1)
    cs = np.tile(np.cumsum(weights), (N, 1))
    rs = np.tile(np.random.rand(N)[:, None], (1, len(weights)))
    indices = np.sum(cs < rs, axis=1)
    return list(map(pool.__getitem__, indices))


def test_weighted_sample_with_replacement():
    pool = [1, 2, 3]
    ps = [0.2, 0.3, 0.5]
    samples = weighted_sample_with_replacement(pool, ps, 10000)
    cnt = Counter(samples)
    total = sum(cnt.values())
    cnt[1] /= total
    cnt[2] /= total
    cnt[3] /= total
    np.testing.assert_almost_equal(sorted(cnt.values()), ps, decimal=2)


def test_generalized_jaccard_similarity():
    a = [1, 1, 2]
    b = [1, 2, 3]
    assert generalized_jaccard_similarity(a, b) == 0.5
    assert generalized_jaccard_similarity(a, a) == 1.0


def infeciton_time2weight(ts):
    """invert the infection times so that earlier infected nodes have larger weight"""
    max_val = np.max(ts)
    ts[ts == -1] = max_val + 1
    return np.array(
        [(max_val - t + 1)
         for n, t in enumerate(ts)])


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def sp_len_2d(g, dtype=np.float64):
    n = g.number_of_nodes()
    d = np.zeros((n, n), dtype=dtype)
    sp_len = nx.shortest_path_length(g)
    for i in np.arange(n):
        d[i, :] = [sp_len[i][j] for j in np.arange(n)]
    return d


def get_rank_index(array, id_):
    val = array[id_]
    sorted_array = np.sort(array)[::-1]
    return np.nonzero(sorted_array == val)[0][0]
