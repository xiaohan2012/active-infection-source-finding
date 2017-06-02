from collections import Counter

from graph_tool import Graph
from graph_tool.search import pbfs_search
from steiner_tree_mst import init_visitor, extract_edges_from_pred
from utils import build_minimum_tree


def temporal_bfs_old(g, root, infection_times, source, terminals,
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

@profile
def temporal_bfs(g, root, infection_times, source, terminals,
                 debug=False,
                 verbose=True):
    terminals = set(terminals)
    visited = {root}
    edges = []

    processed_by_time = Counter()
    for v in terminals:
        processed_by_time[infection_times[v]] += 1

    terminals_sorted = list(sorted(
        terminals,
        key=lambda t: (infection_times[t], (t not in visited))))

    all_times_sorted = list(sorted(map(infection_times.__getitem__, terminals)))
    tmin = infection_times[root]
    tmin_idx = 0
    processed_by_time[tmin] -= 1

    # update tmin
    if processed_by_time[tmin] == 0:
        tmin_idx += 1
        tmin = all_times_sorted[tmin_idx]

    queue = [root]
    delayed = set()
    while len(queue) > 0:
        u = queue.pop(0)
        for v in g.vertex(u).all_neighbours():
            v = int(v)
            if v not in visited:
                edges.append((u, v))
                visited.add(v)
                if v in terminals:
                    delayed.add(v)
                    processed_by_time[infection_times[v]] -= 1
                else:
                    queue.append(v)
        # update tmin
        while processed_by_time[tmin] == 0 and tmin_idx < len(all_times_sorted)-1:
            tmin_idx += 1
            tmin = all_times_sorted[tmin_idx]

        # re-enqueue delayed terminal nodes
        for v in terminals_sorted:
            if v in delayed:
                if infection_times[v] > tmin:
                    break
                else:
                    delayed.remove(v)
                    queue.append(v)
    
    return build_minimum_tree(g, root, terminals, edges)
