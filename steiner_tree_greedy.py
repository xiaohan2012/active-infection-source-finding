import numpy as np
from graph_tool import Graph
from graph_tool.search import cpbfs_search
from steiner_tree_mst import init_visitor, extract_edges_from_pred

# @profile
def steiner_tree_greedy(
        g, root, infection_times, source, obs_nodes,
        debug=False,
        verbose=True):
    # root = min(obs_nodes, key=infection_times.__getitem__)
    sorted_obs = list(sorted(obs_nodes, key=infection_times.__getitem__))[1:]
    tree_nodes = {root}
    tree_edges = set()
    for u in sorted_obs:
        # connect u to the tree
        vis = init_visitor(g, u)
        if debug:
            print('connect {} to tree'.format(u))
            print('nodes connectable: {}'.format(tree_nodes))
        forbidden_nodes = list(set(obs_nodes) - tree_nodes)
        cpbfs_search(g, u, visitor=vis,
                     terminals=list(tree_nodes),
                     forbidden_nodes=forbidden_nodes,
                     count_threshold=1)

        # add edge
        reachable_nodes = set(np.nonzero(vis.dist > 0)[0]).intersection(tree_nodes)

        if debug:
            print('reachable_nodes: {}'.format(reachable_nodes))

        assert len(reachable_nodes) > 0
        sorted_ancestors = sorted(reachable_nodes, key=vis.dist.__getitem__)
        ancestor = sorted_ancestors[0]

        if debug:
            print('ancestor: {}'.format(ancestor))
            print('dist to reachable: {}'.format(vis.dist[sorted_ancestors]))

        new_edges = extract_edges_from_pred(g, u, ancestor, vis.pred)

        if debug:
            print('new_edges: {}'.format(new_edges))

        tree_edges |= set(new_edges)
        tree_nodes |= {v for e in new_edges for v in e}

    t = Graph(directed=True)
    for _ in range(g.num_vertices()):
        t.add_vertex()

    vfilt = t.new_vertex_property('bool')
    vfilt.a = False
    for v in tree_nodes:
        vfilt[t.vertex(v)] = True

    for u, v in tree_edges:
        t.add_edge(t.vertex(u), t.vertex(v))

    t.set_vertex_filter(vfilt)

    return t

