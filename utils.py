import numpy as np
from collections import Counter


def generalized_jaccard_similarity(a, b):
    """a and b should be list
    >>> a = [1, 1, 2]
    >>> b = [1, 2, 3]
    >>> generalized_jaccard_similarity(a, b)
    0.5
    >>> generalized_jaccard_similarity(a, a)
    1.0
    """
    a, b = map(Counter, [a, b])
    all_elements = set(a.keys()) | set(b.keys())
    counts = [(a[e], b[e]) for e in all_elements]
    numer = np.sum(np.min(counts, axis=1))
    denom = np.sum(np.max(counts, axis=1))
    return numer / denom


def generalized_jaccard_distance(a, b):
    return 1 - generalized_jaccard_similarity(a, b)
