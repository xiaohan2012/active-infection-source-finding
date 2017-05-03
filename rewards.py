import numpy as np


def exact_rewards(t1, t2, dists1, dists2, mask):
    counts = mask.sum(axis=0)
    return (((dists1 - dists2) == (t1 - t2)) * mask).sum(axis=0) / counts


def order_rewards(t1, t2, dists1, dists2, mask):
    counts = mask.sum(axis=0)
    if t1 == t2:
        flags = (dists1 == dists2)
    else:
        flags = ((dists1 < dists2) == (t1 < t2))
    return (flags * mask).sum(axis=0) / counts


def dist_rewards(t1, t2, dists1, dists2, len1, len2, mask, debug=False, source=None):
    counts = mask.sum(axis=0)
    diff_means = (np.sum((dists1 - dists2) * mask,
                         axis=0)
                  / counts)
    actual_diff = t1 - t2
    if debug:
        print('actual_diff: {}'.format(actual_diff))
        print('diff means: {}'.format(diff_means[source]))
        print('dist normalization: {}'.format(np.absolute(len1 + len2)))
    penalty = (np.absolute(actual_diff - diff_means)
               / (np.absolute(len1 + len2)))
    
    max_penalty = penalty[np.invert(np.isnan(penalty))].max()
    if debug:
        print('max penalty {}'.format(max_penalty))
        
    penalty = penalty / max_penalty  # normalize to 1
    probas = 1 - penalty  # invert
    return probas
