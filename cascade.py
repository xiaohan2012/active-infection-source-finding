import math
import numpy as np


def observe_cascade(c, source, q, method='uniform', source_includable=False):
    """graph_tool version of make_partial_cascade
    """
    all_infection = np.nonzero(c != -1)[0]
    if not source_includable:
        all_infection = list(set(all_infection) - {source})
    num_obs = int(math.ceil(len(all_infection) * q))

    if num_obs < 2:
        num_obs = 2

    if method == 'uniform':
        return np.random.permutation(all_infection)[:num_obs]
    elif method == 'late':
        return np.argsort(c)[-num_obs:]


def gen_nontrivial_cascade(g, p, q, model='ic', source=None, return_tree=False, source_includable=False):
    assert model in {'ic', 'si', 'sp'}
    while True:
        if model == 'ic':
            from ic import simulate_cascade
            rts = simulate_cascade(g, p, source=source, return_tree=return_tree)
        elif model == 'si':
            from si import gen_cascade
            rts = gen_cascade(g, p, source=source)
        elif model == 'sp':
            from sp import gen_cascade
            rts = gen_cascade(g, source=source)
        
        source, c = rts[:2]
        if return_tree:
            tree = rts[2]
        obs_nodes = observe_cascade(c, source, q, method='uniform',
                                    source_includable=source_includable)
        cascade_size = np.sum(c != -1)
        
        if cascade_size >= 5:  # avoid small cascade
            break
    if return_tree:
        return c, source, obs_nodes, tree
    else:
        return c, source, obs_nodes
