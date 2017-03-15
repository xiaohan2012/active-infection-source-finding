import networkx as nx
import numpy as np

from ic import sample_graph_from_infection
from core import normalize_mu


MEDIAN_NODE = 'edge_mwu_median_node'


def reward_by_infection_direction(g, q, u, n_rounds=100):
    """for each node as source, calculate
    the fraction of cascades in which cascade goes from u to q"""
    reward = {n: 0 for n in g.nodes_iter()}

    for i in range(n_rounds):
        sampled_g = sample_graph_from_infection(g)
        for s in g.nodes_iter():
            try:
                for path in nx.all_shortest_paths(sampled_g, source=s, target=q):
                    if u in path:
                        reward[s] += 1
                        break
            except nx.NetworkXNoPath:
                pass

    for n in g.nodes_iter():
        reward[n] /= n_rounds
    return reward


def reward_by_uninfected_node(g, q, n_rounds=100):
    reward = {n: 0 for n in g.nodes_iter()}

    for i in range(n_rounds):
        sampled_g = sample_graph_from_infection(g)
        for s in g.nodes_iter():
            try:
                nx.shortest_path(sampled_g, source=s, target=q)
            except nx.NetworkXNoPath:  # uninfected
                reward[s] += 1
    for n in g.nodes_iter():
        reward[n] /= n_rounds
    return reward


def median_node(g, mu, sp_len):
    def sum_of_weighted_dist(q):
        mus = np.array([mu[v] for v in g.nodes_iter()])
        lens = np.array([sp_len[q][v] for v in g.nodes_iter()])
        return np.sum(mus * lens)

    return min(g.nodes_iter(), key=sum_of_weighted_dist)


def mwu_by_infection_direction(g,
                               query_method,
                               obs_nodes, infection_times, source,
                               direction_reward_table,
                               inf_reward_table,
                               max_iter=float('inf'),
                               save_logs=False,
                               debug=False):
    mu = {n: 1 for n in g.nodes_iter()}
    sp_len = nx.shortest_path_length(g, weight='d')
    centroids = []
    queried_nodes = set(obs_nodes)
    i = 0
    while i < max_iter:
        i += 1
        if len(queried_nodes) == g.number_of_nodes():
            print("no more queries to go")
            break

        if query_method == MEDIAN_NODE:
            q = median_node(g, mu, sp_len)
        else:
            raise ValueError('unsuportted methods {}'.format(query_method))

        queried_nodes.add(q)
        
        if debug:
            print('query node: {}'.format(q))
        
        if save_logs:
            centroids.append(q)
        found_source = True
        if np.isinf(infection_times[q]):  # uninfected
            found_source = False
            reward = {n: inf_reward_table[(n, q)]
                      for n in g.nodes_iter()}
        else:
            for u in g.neighbors(q):
                queried_nodes.add(u)
                if infection_times[u] < infection_times[q]:
                    reward = {n: direction_reward_table[(n, u, q)]
                              for n in g.nodes_iter()}
                    found_source = False
                    break
        if found_source:
            assert source == q
            break
            
        for n in g.nodes_iter():
            mu[n] *= reward[n]
        mu = normalize_mu(mu)
    return len(queried_nodes - obs_nodes)
