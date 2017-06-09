from collections import Counter

from utils import build_minimum_tree
from sorted_array import insert


# @profile
def temporal_bfs(g, root, infection_times, source, terminals,
                 debug=False,
                 verbose=True):
    terminals = set(terminals)
    visited = {root}
    edges = []

    processed_by_time = Counter()
    for v in terminals:
        processed_by_time[infection_times[v]] += 1

    all_times_sorted = list(sorted(map(infection_times.__getitem__, terminals)))
    tmin = infection_times[root]
    tmin_idx = 0
    processed_by_time[tmin] -= 1

    # update tmin
    if processed_by_time[tmin] == 0:
        tmin_idx += 1
        tmin = all_times_sorted[tmin_idx]

    queue = [root]
    delayed = []
    delayed_keys = []

    while len(queue) > 0:
        u = queue.pop(0)
        for v in g.vertex(u).all_neighbours():
            v = int(v)
            if v not in visited:
                edges.append((u, v))
                visited.add(v)
                if v in terminals:
                    insert(delayed, delayed_keys, v, infection_times.__getitem__)
                    processed_by_time[infection_times[v]] -= 1
                else:
                    queue.append(v)
        # update tmin
        while processed_by_time[tmin] == 0 and tmin_idx < len(all_times_sorted)-1:
            tmin_idx += 1
            tmin = all_times_sorted[tmin_idx]

        # re-enqueue delayed terminal nodes
        i = 0
        for v in delayed:
            if infection_times[v] > tmin:
                break
            else:
                i += 1
                queue.append(v)
        if i > 0:
            delayed = delayed[i:]
            delayed_keys = delayed_keys[i:]
        
    return build_minimum_tree(g, root, terminals, edges)
