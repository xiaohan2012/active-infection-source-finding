from graph_tool import Graph
from steiner_tree_mst import init_visitor, extract_edges_from_pred
from graph_tool.search import pbfs_search

# @profile
def temporal_bfs(g, root, infection_times, source, terminals,
                 debug=False,
                 verbose=True):
    terminals = set(terminals)
    visited = {root}
    edges = []

    terminals_sorted = list(sorted(terminals,
                                   key=lambda t: (infection_times[t], (t not in visited))))
    while len(terminals_sorted) > 0:
        n = terminals_sorted.pop(0)
        queue = [n]
        while len(queue) > 0:
            u = queue.pop(0)

            for v in g.vertex(u).all_neighbours():
                v = int(v)
                if v not in visited:
                    edges.append((u, v))
                    visited.add(v)
                    if v not in terminals:
                        # continue with non-terminal node
                        queue.append(v)
        # needs to sort the terminals because `visited` changed
        terminals_sorted = list(
            sorted(terminals_sorted,
                   key=lambda t: (infection_times[t], (t not in visited))))

    assert len(visited.intersection(terminals)) == len(terminals)
    
    # build the tree
    t = Graph(directed=True)

    for _ in range(g.num_vertices()):
        t.add_vertex()

    for (u, v) in edges:
        t.add_edge(u, v)

    if True:
        # mask out redundant edges
        vis = init_visitor(t, root)
        pbfs_search(t, source=root, terminals=list(terminals), visitor=vis)

        minimum_edges = {e
                         for u in terminals
                         for e in extract_edges_from_pred(t, root, u, vis.pred)}
        # print(minimum_edges)
        efilt = t.new_edge_property('bool')
        efilt.a = False
        for u, v in minimum_edges:
            efilt[u, v] = True
        t.set_edge_filter(efilt)

        vfilt = t.new_vertex_property('bool')
        vfilt.a = False
        tree_nodes = {u for e in minimum_edges for u in e}
        for v in tree_nodes:
            vfilt[v] = True
        t.set_vertex_filter(vfilt)
    
    return t
