import numpy as np
import networkx as nx

from copy import copy
from ic import sample_graph_from_infection
from core import normalize_mu


def median_node(g, mu, sp_len):
    def sum_of_weighted_dist(q):
        mus = np.array([mu[v] for v in g.nodes_iter()])
        lens = np.array([sp_len[q][v] for v in g.nodes_iter()])
        return np.sum(mus * lens)

    return min(g.nodes_iter(), key=sum_of_weighted_dist)


def noisy_binary_search(g,
                        source,
                        infection_times,
                        obs_nodes,
                        sp_len,
                        consistency_multiplier,
                        max_iter,
                        uninfected_simulation_rounds=100,
                        debug=False,
                        save_log=False):
    # observed from the plot in infection_probability_vs_fraction_of_shortest_path.ipynb
    mu = {n: 1 for n in g.nodes_iter()}
    for n in obs_nodes:
        mu[n] = 0

    queried_nodes = copy(obs_nodes)
    querie_nodes_log = []
    for i in range(max_iter):
        if debug:
            print('source\'s mu: {:.2f}'.format(mu[source]))
        
        if len(queried_nodes) == g.number_of_nodes():
            break
        q = median_node(g, mu, sp_len)
        queried_nodes.add(q)

        if save_log:
            querie_nodes_log.append(q)
        if debug:
            print('query node: {}'.format(q))

        if np.isinf(infection_times[q]):
            if debug:
                print('query is not infected')
            # estimate the fraction of simulations that n is not infected
            reward = {n: 0 for n in g.nodes_iter()}
            for i in range(uninfected_simulation_rounds):
                sg = sample_graph_from_infection(g)
                sp_len_prime = nx.shortest_path_length(sg, source=q)
                for n in g.nodes_iter():
                    if n not in sp_len_prime:
                        reward[n] += 1
            for n in g.nodes_iter():
                mu[n] *= (reward[n] / uninfected_simulation_rounds)
                
            mu = normalize_mu(mu)
        else:
            # check if q is source
            found_source = True
            for n in g.neighbors_iter(q):
                if infection_times[q] > infection_times[n]:
                    found_source = False

            if found_source:
                assert q == source, '{} != {} ({} and {})'.format(
                    q, source,
                    infection_times[q],
                    infection_times[source])
                break

            possible_ancestors = []
            if False:
                for n in g.neighbors_iter(q):
                    queried_nodes.add(n)
                    if save_log:
                        querie_nodes_log.append(n)
                    if infection_times[q] - infection_times[n] == g[n][q]['d']:
                        possible_ancestors.append(n)
            else:
                # using just one ancestor node
                for n in g.neighbors_iter(q):
                    queried_nodes.add(n)
                    if save_log:
                        querie_nodes_log.append(n)
                    if infection_times[q] - infection_times[n] == g[n][q]['d']:
                        possible_ancestors.append(n)
                        break
                
            if debug:
                print('ancestor nodes: {}'.format(possible_ancestors))

            if len(possible_ancestors) > 0:
                for a in possible_ancestors:
                    for n in g.nodes_iter():
                        if sp_len[n][q] == (sp_len[n][a] + g[a][q]['d']):
                            mu[n] *= consistency_multiplier
                        else:
                            mu[n] *= (1 - consistency_multiplier)
                    mu = normalize_mu(mu)

    query_count = len(queried_nodes - obs_nodes)
    if debug:
        print('used {} queries to find the source'.format(query_count))
    if save_log:
        return query_count, queried_nodes
    else:
        return query_count
