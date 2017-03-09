import random
import networkx as nx
import numpy as np
from copy import copy
from core import normalize_mu, print_nodes_by_mu, penalty_using_distribution
from query_strategy import centroid, maximal_adversarial_query


RANDOM = 'random'
MAX_MU = 'max_mu'
MAX_ADV = 'max_adversarial'
CENTROID = 'by_centroid'


def main_routine(g, node2id, id2node,
                 source, obs_nodes, infection_times,
                 s2n_probas,
                 epsilon,
                 check_neighbor_threshold=0.1,
                 query_selection_method=MAX_MU,
                 max_iter=float('inf'),
                 debug=False,
                 save_log=False):
    """
    infection_times: observation
    s2n_probas: 3D tensor, N x N x max(t)
        1st dim: source node s
        2nd dim: infected node u
        3rd dim: time
        value: probability on t(u | s)

    save_log: boolean,
    wheter save the querying process log (mu_list, query_list)
    """
    if query_selection_method == CENTROID:
        sp_len = nx.shortest_path_length(g, weight='d')
    obs_nodes = copy(obs_nodes)

    mu_list = []
    query_list = []
        
    mu = {n: 1 for n in g.nodes()}
    mu = normalize_mu(mu)
            
    queried_nodes = set()

    # using the observations
    for q in sorted(obs_nodes, key=lambda n: infection_times[n]):
        outcome = infection_times[q]

        penalty = penalty_using_distribution(q, outcome, s2n_probas, node2id)
        for n, p in penalty.items():
            mu[n] *= np.power(1-epsilon, p)
    
        mu = normalize_mu(mu)
        queried_nodes.add(q)

        if save_log:
            mu_list.append(mu)
            query_list.append(q)

    if debug:
        print_nodes_by_mu(g, mu, source)
            
    iter_i = 0

    all_nodes = set(g.nodes())
    while iter_i < max_iter:
        iter_i += 1
        queryable = all_nodes
        if len(queried_nodes) == g.number_of_nodes():
            print('no more nodes to query')
            break
        if query_selection_method == RANDOM:
            q = random.choice(list(all_nodes))
        elif query_selection_method == MAX_MU:
            q = max(all_nodes, key=lambda n: mu[n])
        elif query_selection_method == MAX_ADV:
            mu_array = np.zeros(g.number_of_nodes())
            for n, val in mu.items():
                mu_array[node2id[n]] = val
            _, node2penalty = maximal_adversarial_query(
                g, s2n_probas, mu_array, obs_nodes, infection_times, node2id, id2node)
            node_ids = [node2id[n] for n in node2penalty.keys()]
            qid = np.random.choice(node_ids,
                                   size=1,
                                   p=list(normalize_mu(node2penalty).values()))
            assert len(qid) == 1
            q = id2node[qid[0]]
        elif query_selection_method == CENTROID:
            q = centroid(queryable, g, mu, sp_len)
        else:
            raise ValueError('available query methods are {}'.format(RANDOM, MAX_MU, CENTROID))
    
        queried_nodes.add(q)

        if debug:
            print('query {}'.format(q))

        # update weight
        outcome = infection_times[q]
        penalty = penalty_using_distribution(q, outcome, s2n_probas, node2id)
        for n, p in penalty.items():
            mu[n] *= np.power(1-epsilon, p)

        mu = normalize_mu(mu)

        # save log if necessary
        if save_log:
            mu_list.append(mu)
            query_list.append(q)

        # if some node has very large mu
        # query its neighbors
        if max(mu.values()) > check_neighbor_threshold:
            s = max(mu, key=mu.__getitem__)
            queried_nodes |= set(g.neighbors(s) + [s])
                    
            is_source = np.all(list(map(lambda u: infection_times[s] < infection_times[u],
                                        g.neighbors(s))))
            if is_source:
                if debug:
                    print('**Found source')
                    print('used {} queries'.format(len(queried_nodes - set(obs_nodes))))
                assert source == s
                break
            else:
                mu[s] = 0  # important
            
        if debug:
            print('iteration: {}'.format(iter_i))
            print_nodes_by_mu(g, mu, source)

    query_count = len(queried_nodes - set(obs_nodes))
    if save_log:
        return query_count, mu_list, query_list
    else:
        return query_count

