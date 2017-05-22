import numpy as np
from graph_tool.all import GraphView, shortest_path
from collections import defaultdict
from utils import extract_edges


def temporal_bfs(g, r, infection_times, source, obs_nodes, debug=False):
    """return the tree covering obs_nodes"""
    queue = [r]
    t_lower = np.ones(g.num_vertices(), dtype=np.int32) * -1  # hidden nodes has lower bound -1
    t_lower[obs_nodes] = infection_times[obs_nodes]
    t_lower[r] = infection_times[obs_nodes].min() - 1
    visited = g.new_vertex_property('bool')
    visited.set_2d_array(np.zeros(g.num_vertices(), dtype=bool))
    tree = []
    while len(queue) > 0 and np.any(visited.a[obs_nodes] == 0):
        v = queue.pop(0)

        if debug:
            print('visiting {}'.format(v))

        visited[g.vertex(v)] = True
        for u in g.vertex(v).all_neighbours():
            if debug:
                print('trying its nbr {}'.format(u))

            if visited[u] == 0:
                if debug:
                    print('{} is not visited'.format(u))
                    print('t_l[u]={}, t_l[v]={}'.format(t_lower[u], t_lower[v]))
                u = int(u)
                visitable = False

                if t_lower[u] > t_lower[v]:
                    if debug:
                        print('first case')
                    visitable = True

                if t_lower[u] == -1:
                    if debug:
                        print('second case')
                    visitable = True
                    t_lower[u] = t_lower[v]

                if visitable:
                    if debug:
                        print('add {} to queue'.format(u))
                    queue.append(u)
                    tree.append((v, u))
                    visited[g.vertex(u)] = True
    if np.any(visited.a[obs_nodes] == 0):
        # some terminal is uncovered
        return False
    else:
        efilt = g.new_edge_property('bool')
        for u, v in tree:
            efilt[g.edge(u, v)] = True
        
        min_tree_efilt = g.new_edge_property('bool')
        tree = GraphView(g, efilt=efilt)
        for o in obs_nodes:
            _, edge_list = shortest_path(tree, source=tree.vertex(r), target=tree.vertex(o))
            for e in edge_list:
                min_tree_efilt[e] = True
        min_tree = GraphView(g, efilt=min_tree_efilt)
        return min_tree


def tree_sizes_by_roots(g, obs_nodes, infection_times, source, method='sync_tbfs'):
    """
    use temporal BFS to get the scores for each node in terms of the negative size of the inferred tree
    thus, the larger the better
    """
    assert method in {'sync_tbfs', 'tbfs'}
    cand_sources = set(np.arange(g.num_vertices())) - set(obs_nodes)

    tree_sizes = np.ones(g.num_vertices()) * float('inf')
    for r in cand_sources:
        if method == 'tbfs':
            tree = temporal_bfs(g, r, infection_times, source, obs_nodes, debug=False)
        else:
            tree = temporal_bfs_sync(g, r, infection_times, source, obs_nodes, debug=False)
        if tree:
            tree_sizes[r] = tree.num_edges()

    return -tree_sizes


def remove_redundant_edges_from_tree(g, tree, r, terminals):
    """given a set of edges, a root, and terminals to cover,
    return a new tree with redundant edges removed"""
    efilt = g.new_edge_property('bool')
    for u, v in tree:
        efilt[g.edge(u, v)] = True
    tree = GraphView(g, efilt=efilt)

    # remove redundant edges
    min_tree_efilt = g.new_edge_property('bool')
    min_tree_efilt.set_2d_array(np.zeros(g.num_edges()))
    for o in terminals:
        _, edge_list = shortest_path(tree, source=tree.vertex(r), target=tree.vertex(o))
        assert len(edge_list) > 0, 'unable to reach {} from {}'.format(o, r)
        for e in edge_list:
            min_tree_efilt[e] = True
    min_tree = GraphView(g, efilt=min_tree_efilt)
    return min_tree


def temporal_bfs_sync(g, r, infection_times, source, obs_nodes, debug=False):
    queue = [r]
    t_lower = np.ones(g.num_vertices(), dtype=np.int32) * -1  # hidden nodes has lower bound -1
    t_lower[obs_nodes] = infection_times[obs_nodes]
    t_lower[r] = infection_times[obs_nodes].min() - 1
    visited = g.new_vertex_property('bool')
    visited.set_2d_array(np.zeros(g.num_vertices(), dtype=bool))
    tree = []

    obs_by_time = defaultdict(list)
    for o in obs_nodes:
        obs_by_time[infection_times[o]].append(o)
    obs_times = list(sorted(set(infection_times[obs_nodes])))

    success = True
    for cur_t in obs_times:
        banned_nodes = {v for v in obs_nodes if infection_times[v] != cur_t}
        target_nodes = [v for v in obs_nodes if infection_times[v] == cur_t]
        
        if debug:
            print('current time = {}'.format(cur_t))
            print('targets {}'.format(target_nodes))
        targets_covered = False
        # cover nodes of level t
        while len(queue) > 0:
            if np.all(visited.a[target_nodes] == 1):
                if debug:
                    print('covered all targets')
                targets_covered = True
                break
            v = queue.pop(0)
            for u in g.vertex(v).all_neighbours():
                if u not in banned_nodes and visited[u] == 0:
                    if debug and u in target_nodes:
                        print('cover target {}'.format(u))
                    if debug:
                        print('add edge {}'.format((v, int(u))))
                    tree.append((v, int(u)))
                    visited[u] = True
                    queue.append(int(u))
        if targets_covered:
            if False:
                if debug and False:
                    print('update tree and visited table')
                # remove redundant edges
                # construct the tree from used edges
                terminals = [o for o in obs_nodes if infection_times[o] <= cur_t]
                if debug:
                    print('terminals to cover: {}'.format(terminals))

                min_tree = remove_redundant_edges_from_tree(g, tree, r, terminals)
                if debug:
                    print('size of min tree: {}'.format(min_tree.num_edges()))
                tree = extract_edges(min_tree)
                if debug:
                    print('current tree edges {}'.format(tree))

                # update visited table
                visited.set_2d_array(np.zeros(g.num_vertices()))
                covered_nodes = {u for nodes in tree for u in nodes}
                for v in covered_nodes:
                    visited[g.vertex(v)] = 1
                queue = list(covered_nodes)
            continue
        else:
            if debug:
                print('failed to cover targets')
            success = False
            break
    if success:
        return remove_redundant_edges_from_tree(g, tree, r, obs_nodes)
    else:
        return None
