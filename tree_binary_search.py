import networkx as nx
import numpy as np


def subtree_size(g, u, v, cache):
    """size of u's subtree excluding the v part
    """
    if (u, v) in cache:
        return cache[(u, v)]
    
    u_nbrs = g.neighbors(u)
    if len(u_nbrs) == 1:
        assert u_nbrs[0] == v
        cache[(u, v)] = 1
        return 1

    v_nbrs = g.neighbors(v)
    if len(v_nbrs) == 1:
        assert v_nbrs[0] == u
        cache[(u, v)] = g.number_of_nodes() - 1
        return cache[(u, v)]
        
    cache[(u, v)] = 1 + sum(subtree_size(g, n, u, cache)
                            for n in g.neighbors(u)
                            if n != v)
    return cache[(u, v)]


def subtree_size_iterative(g, i, j, cache):
    stack = [(i, j)]
    
    while len(stack) > 0:
        u, v = stack[0]
        u_nbrs = g.neighbors(u)
        if len(u_nbrs) == 1:
            assert u_nbrs[0] == v
            cache[(u, v)] = 1
            stack.pop(0)
            continue
            
        v_nbrs = g.neighbors(v)
        if len(v_nbrs) == 1:
            assert v_nbrs[0] == u
            cache[(u, v)] = g.number_of_nodes() - 1
            stack.pop(0)
            continue

        pushed = False
        for n in g.neighbors(u):
            if n != v:
                if (n, u) not in cache:
                    pushed = True
                    stack.insert(0, (n, u))
        if not pushed:  # every thing ready for u, v
            cache[(u, v)] = 1 + sum(cache[n, u]
                                    for n in g.neighbors(u)
                                    if n != v)
            stack.pop(0)
    return cache[(i, j)]

    
def find_centroid(g):
    cache = {}
    half_size = g.number_of_nodes() / 2
    for u in g.nodes_iter():
        sizes = np.array([subtree_size_iterative(g, n, u, cache)
                          for n in g.neighbors(u)])
        if (sizes <= half_size).all():
            return u


def get_infected_subgraph(g, q, queried_nodes):
    g = g.copy()
    g.remove_node(q)
    sg = g.subgraph(next(comp
                         for comp in nx.connected_components(g)
                         if comp.intersection(queried_nodes)))
    return sg


def find_source(g, obs_nodes, infection_times,
                debug=False):
    # print(infection_times)
    # print(obs_nodes)
    queried_nodes = set(obs_nodes)
    sg = g.copy()
    while True:
        q = find_centroid(sg)
        queried_nodes.add(q)
        if debug:
            print('query', q)

        # uninfected
        if np.isinf(infection_times[q]):
            if debug:
                print('uninfected')
            sg = get_infected_subgraph(sg, q, queried_nodes)
        else:
            if debug:
                print('nbr', g.neighbors(q))
            found_source = True
            for n in g.neighbors(q):
                queried_nodes.add(n)
                if infection_times[n] < infection_times[q]:
                    if debug:
                        print('earlier nbr', n)
                        print('their time', infection_times[n], infection_times[q])
                    found_source = False
                    sg = get_infected_subgraph(sg, q, {n})
                    break
            if found_source:
                true_source = min(infection_times, key=lambda n: infection_times[n])
                assert q == true_source
                break
        if debug:
            print('new subgraph size {}'.format(sg.number_of_nodes()))
    if debug:
        print(queried_nodes)
    return len(queried_nodes - obs_nodes)
