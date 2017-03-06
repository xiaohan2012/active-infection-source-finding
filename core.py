import numpy as np
from simulations import make_input
from simulations import sample_graph_from_infection, simulate_IC


def print_nodes_by_mu(g, mu, source, top_k=5):
    sorted_nodes = list(sorted(g.nodes(), key=lambda n: mu[n], reverse=True))
    for n in sorted_nodes[:top_k]:
        # print('{}: {:.5f}'.format(n, mu[n]))
        pass
    print('source ranks {} th: {:.2f}'.format(sorted_nodes.index(source)+1,
                                              mu[source]))
        

def accuracy_for_nodes(query, outcome, times_by_source, node2id):
    """
    for each node as source, calculate the fraction of cascades that matches the "outcome" on node "query"
    """
    id_ = node2id[query]
    acc = {}
    for s, m in times_by_source.items():
        acc[s] = np.count_nonzero(m[:, id_] == outcome) / m.shape[0]
    return acc


def normalize_mu(mu):
    total = sum(mu.values())
    return {n: v / total for n, v in mu.items()}


def mwu(g, source, obs_nodes, infection_times,
        times_by_source,
        epsilon,
        debug=False, plot=False,
        inspect_mu=False,
        max_plots=9):
    """
    times_by_source: sufficient statistics from each node (precomputed)
    """
    mu_of_source = []
    node2id = {n: i for i, n in enumerate(sorted(g.nodes()))}
    mu = {n: 1 for n in g.nodes()}
    mu = normalize_mu(mu)
    mu_of_source.append(mu[source])

    query_count = 0
    queried_nodes = set()

    # using the observations
    temp_i = 0
    for q in sorted(obs_nodes, key=lambda n: infection_times[n]):
        temp_i += 1
        outcome = infection_times[q]
        if debug:
            print('query {} -> {}'.format(q, outcome))

        acc = accuracy_for_nodes(q, outcome, times_by_source, node2id)
        for n, ac in acc.items():
            if debug:
                if temp_i <= 2 and n == source:
                    id_ = node2id[q]
                    # print('query id_', id_)
                    m = times_by_source[source]
                    c = np.count_nonzero(m[:, id_] == outcome) / m.shape[0]
                    # print('m[:, id_]', m[:, id_])
                    # print('type(array)', type(m[:, id_]))
                    # print('outcome', outcome)
                    # print('type(outcome)', type(outcome))
                    # print('accuracy for source {}'.format(ac))
            mu[n] *= np.power(1-epsilon, 1-ac)
        mu = normalize_mu(mu)
        if debug:
            print('mu(source)={}'.format(mu[source]))
        queried_nodes.add(q)

    if debug:
        print_nodes_by_mu(g, mu, source)

    max_iter = float('inf')
    iter_i = 0

    all_nodes = set(g.nodes())
    while iter_i < max_iter:
        iter_i += 1
        
        if len(all_nodes - queried_nodes) == 0:
            if debug:
                print('no more nodes to query')
            break
            
        q = max(all_nodes - queried_nodes, key=lambda n: mu[n])
        queried_nodes.add(q)
        query_count += 1

        if debug:
            print('query {}'.format(q))

        # if some node has very large mu
        # query its neighbors
        if not inspect_mu and max(mu.values()) > 0.1:
            s = max(mu, key=mu.__getitem__)
            if s not in queried_nodes:
                query_count += 1
                queried_nodes.add(s)
                
            is_source = np.all(list(map(lambda u: infection_times[s] < infection_times[u],
                                        g.neighbors(s))))
            unqueried_nodes = set(filter(lambda u: u not in queried_nodes, g.neighbors(s)))
            query_count += len(unqueried_nodes)
            queried_nodes |= unqueried_nodes
            if is_source:
                if debug:
                    print('**Found source')
                    print('used {} queries'.format(query_count))
                assert source == s
                break

        outcome = infection_times[q]
        acc = accuracy_for_nodes(q, outcome, times_by_source, node2id)
        for n, ac in acc.items():
            mu[n] *= np.power(1-epsilon, 1-ac)

        mu = normalize_mu(mu)

        mu_of_source.append(mu[source])
        if debug:
            print('iteration: {}'.format(iter_i))
            print_nodes_by_mu(g, mu, source)

    return query_count, mu_of_source


def experiment_multiple_rounds(rounds,
                               g, times_by_source,
                               infp, fraction, epsilon, sampling_method,
                               debug=False,
                               seed=None):
    np.random.seed(seed)
    results = []
    for i in range(rounds):
        source, obs_nodes, infection_times, tree = make_input(
            g, infp, fraction,
            sampling_method=sampling_method)
        r = mwu(g, source, obs_nodes, infection_times, times_by_source, epsilon,
                debug=debug)
        results.append(r)
    return results


def generate_sufficient_stats(g):
    """
    **Buggy**!! the sampled graph and IC process performs **two** runs of sampling

    given one graph, simulate **one** IC process from **each** node as the source on
    the **same** sampled graph
    
    Returns:
    for each node as source, return the infection times and tree
    list of (source, times, tree)
    """
    g1 = sample_graph_from_infection(g)
    return [(s, ) + simulate_IC(g1, s=s, is_g_sampled=True, debug=False)
            for s in g.nodes()]
