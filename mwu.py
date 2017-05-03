import random
import networkx as nx
import numpy as np
from copy import copy
from scipy.sparse import isspmatrix_csr
from graph_tool.all import shortest_distance
from tqdm import tqdm


from core import normalize_mu, print_nodes_by_mu, penalty_using_distribution
from rewards import exact_rewards, order_rewards, dist_rewards
from ic import get_o2src_time, sll_using_pairs, get_infection_time
from utils import get_rank_index

RANDOM = 'random'
MAX_MU = 'max_mu'
RAND_MAX_MU = 'rand_max_mu'


def mwu(g, gvs,
        source, obs_nodes, infection_times, o2src_time=None,
        active_method=MAX_MU,
        reward_method='exact',
        eps=0.2,
        max_iter=float('inf'),
        use_uninfected=True,
        debug=False,
        save_log=False):
    if save_log:
        query_log = []
        sll_log = []
        is_nbr_log = []
    if o2src_time is None:
        o2src_time = get_o2src_time(obs_nodes, gvs, debug=debug)

    if reward_method == 'dist':
        sp_len_dict = {o: shortest_distance(g, source=o).a for o in obs_nodes}
    else:
        sp_len_dict = None
    # init
    sll = sll_using_pairs(
        g,
        obs_nodes,
        infection_times,
        o2src_time,
        sp_len_dict=sp_len_dict,
        source=source,
        method=reward_method,
        eps=eps,
        precond_method='and',
        return_cascade=False,
        debug=debug)
        
    iter_i = 0
    all_nodes = set(np.arange(g.num_vertices()))
    unqueried_nodes = all_nodes - set(obs_nodes)

    obs_nodes = copy(obs_nodes)

    queried_nodes = set()

    # reference nodes to use for MWU,
    # required to be **infected**
    ref_nodes = set(obs_nodes)

    nodes_to_use = []  # nodes coming from querying the neighbors

    while iter_i < max_iter:
        iter_i += 1
        if len(unqueried_nodes) == 0:
            print('no more nodes to query')
            break
        if len(nodes_to_use) == 0:
            if active_method == MAX_MU:
                q = max(unqueried_nodes, key=lambda n: sll[n])
            elif active_method == RANDOM:
                q = random.choice(list(unqueried_nodes))
            else:
                raise ValueError('available query methods are {}'.format(MAX_MU))

            if debug:
                print('query {}'.format(q))
            queried_nodes.add(q)
            unqueried_nodes.remove(q)
            if save_log:
                query_log.append(q)
                is_nbr_log.append(False)
        else:
            if debug:
                print('using node from nodes_to_use')
            q = nodes_to_use.pop()
        q = int(q)
        if infection_times[q] == -1 and use_uninfected:
            # the query is uninfected
            pass
        else:
            # the query is infected
            if reward_method == 'dist':
                sp_len_dict[q] = shortest_distance(g, source=q).a

            o2src_time[q] = np.array([get_infection_time(gv, q) for gv in gvs])

            for o in ref_nodes:
                tq, to = infection_times[q], infection_times[o]
                dists_q, dists_o = o2src_time[q], o2src_time[o]
                mask = np.logical_and(dists_q != -1, dists_o != -1)

                if reward_method == 'exact':
                    probas = exact_rewards(tq, to, dists_q, dists_o, mask)
                elif reward_method == 'order':
                    probas = order_rewards(tq, to, dists_q, dists_o, mask)
                elif reward_method == 'dist':
                    try:
                        probas = dist_rewards(
                            tq, to,
                            dists_q, dists_o,
                            sp_len_dict[q], sp_len_dict[o],
                            mask)
                    except ValueError:
                        # zero-size array to reduction operation maximum which has no identity
                        # ignore this iteration
                        continue
                else:
                    raise ValueError('methoder is unknown')

                probas[np.isnan(probas)] = 0
                sll *= (eps + (1-eps) * probas)
                if np.isclose(sll.sum(), 0):
                    print('warning: sll.sum() close to 0')
                    sll = np.ones(g.num_vertices()) / g.num_vertices()
                else:
                    sll /= sll.sum()

            # when q is used for updating sll, add it to reference list
            ref_nodes.add(q)

            # if the query node infection time is no smaller than
            # the current known earliest infection,
            # it cannot be the source
            min_inf_t = min(infection_times[n] for n in ref_nodes)
            if infection_times[q] >= min_inf_t:
                sll[q] = 0

        if save_log:
            sll_log.append(sll)

        if debug:
            print('add q to ref_nodes (#nodes={})'.format(len(ref_nodes)))
            print('source current rank = {}'.format(get_rank_index(sll, source)))
        
        # if some node has very large mu
        # query its neighbors
        winners = np.nonzero(sll == sll.max())[0]
        for w in winners:
            nbrs = set(map(int, g.vertex(w).all_neighbours()))
            unqueried_neighbors = nbrs - queried_nodes
            nodes_to_use += list(unqueried_neighbors)
            queried_nodes |= unqueried_neighbors

            if save_log:
                query_log += list(unqueried_neighbors)
                is_nbr_log += [True] * len(unqueried_neighbors)

            if infection_times[w] != -1:
                is_source = np.all([(infection_times[w] < infection_times[int(u)])
                                    for u in nbrs])
            else:
                is_source = False
                continue

            if debug:
                print('checking source {} with winner {}'.format(source, w))
                print('winner\'s time {}'.format(infection_times[w]))
                print('winner\'s nbr infection time {}'.format([infection_times[int(u)] for u in nbrs]))

            if is_source:
                query_count = len(queried_nodes)
                if debug:
                    print('**Found source and used {} queries'.format(query_count))
                assert source == w
                if save_log:
                    return query_count, query_log, sll_log, is_nbr_log
                else:
                    return query_count
            else:
                sll[w] = 0

    query_count = len(queried_nodes)
    if save_log:
        return query_count, query_log, sll_log, is_nbr_log
    else:
        return query_count


# @profile
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

    Returns False if fails to find the source
    """
    if isspmatrix_csr(next(iter(s2n_probas.values()))):
        for s, m in tqdm(s2n_probas.items()):
            s2n_probas[s] = m.tolil()
        
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
            # mu[n] *= np.power(1-epsilon, p)
            mu[n] *= p
    
        mu = normalize_mu(mu)
        queried_nodes.add(q)

        if save_log:
            mu_list.append(mu)
            query_list.append(q)

    if debug:
        print_nodes_by_mu(g, mu, source)
            
    iter_i = 0
    success = False
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
        elif query_selection_method == RAND_MAX_MU:
            node_ids = [node2id[n] for n in all_nodes]
            weights = [mu[n] for n in all_nodes]
            q_id = np.random.choice(node_ids, size=1, p=weights)[0]
            q = id2node[q_id]
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
            raise ValueError('available query methods are {}'.format(RANDOM,
                                                                     MAX_MU,
                                                                     CENTROID,
                                                                     RAND_MAX_MU))
    
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
                success = True
                assert source == s
                break
            else:
                mu[s] = 0  # important
                mu = normalize_mu(mu)
        if debug:
            print('iteration: {}'.format(iter_i))
            print_nodes_by_mu(g, mu, source)

    if success:
        query_count = len(queried_nodes - set(obs_nodes))
        if save_log:
            return query_count, mu_list, query_list
        else:
            return query_count
    else:
        return False

if __name__ == '__main__':
    import sys
    from synthetic_data import load_data_by_gtype
    from ic import make_partial_cascade
    g, time_probas, _, _, _, node2id, id2node = load_data_by_gtype(sys.argv[1], sys.argv[2])
    if len(sys.argv) > 3:
        method = sys.argv[3]
    else:
        method = MAX_MU
    source, obs_nodes, infection_times, tree = make_partial_cascade(g, 0.05, 'late_nodes')
    
    query_count = main_routine(g, node2id, id2node,
                               source, obs_nodes, infection_times,
                               time_probas,
                               epsilon=0.7,
                               check_neighbor_threshold=0.1,
                               query_selection_method=method,
                               max_iter=float('inf'),
                               debug=False,
                               save_log=False)
    print('query count: {}'.format(query_count))
