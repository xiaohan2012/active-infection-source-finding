import numpy as np
from fixtures import simulated_cascade_summary, partial_cascade
from mwu import main_routine, MAX_MU, MAX_ADV, RAND_MAX_MU


def setup_module(module):
    np.random.seed(123456)


def run_mwu(method, simulated_cascade_summary, partial_cascade):
    g, s2n_probas, _, _, _, node2id, id2node = simulated_cascade_summary
    source, obs_nodes, infection_times, tree = partial_cascade
    query_count, mu_list, query_list = main_routine(
        g, node2id, id2node,
        source, obs_nodes, infection_times,
        s2n_probas,
        epsilon=0.7,
        check_neighbor_threshold=0.1,
        query_selection_method=method,
        debug=False,
        save_log=True)
    return query_count, mu_list, query_list


def test_mwc_max_mu(simulated_cascade_summary, partial_cascade):
    query_count, mu_list, query_list = run_mwu(
        MAX_MU, simulated_cascade_summary, partial_cascade)
    assert query_count == 6
    assert len(mu_list) > 5
    assert len(query_list) > 5
    

def test_mwc_rand_max_mu(simulated_cascade_summary, partial_cascade):
    query_count, mu_list, query_list = run_mwu(
        RAND_MAX_MU, simulated_cascade_summary, partial_cascade)
    assert query_count == 18
    assert len(mu_list) > 5
    assert len(query_list) > 5
