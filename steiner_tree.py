
import numpy as np
from graph_tool.all import shortest_distance, GraphView


def all_simple_paths_of_length(g, source, o, length,
                               forbidden_nodes=set(),
                               debug=False):
    if length < 1:
        return
    
    visited = [int(source)]
    stack = [g.vertex(source).all_neighbours()]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if debug:
            print('child: {}'.format(child))
        if child is None:
            if debug:
                print('no more child')
            stack.pop()
            visited.pop()
        elif child in forbidden_nodes:
            if debug:
                print('got forbidden node {}'.format(child))
            pass
        elif len(visited) == length:
            if debug:
                print('correct length {}'.format(len(visited)))
                print('child ({}) = o ({})? {}'.format(child, o, child == o))
            if child == o and child not in visited:
                if debug:
                    print('found o')
                yield visited + [int(o)]
                stack.pop()
                visited.pop()
        elif len(visited) > length:
            stack.pop()
            visited.pop()
        else:  # still more edges to go
            if debug:
                print('len(visited) length {}'.format(len(visited)))
            if child not in visited:
                visited.append(int(child))
                stack.append(g.vertex(child).all_neighbours())


def max_infection_time(g, infection_times, obs_nodes, cand_source, debug):
    t_min = min(infection_times[obs_nodes])
    earliest_node = min(obs_nodes, key=infection_times.__getitem__)
    if debug:
        print('candidate {}'.format(cand_source))
        print('earliest node: {} (t={})'.format(earliest_node, t_min))

    # maximum infection time of source assuming cand_source is source
    # consider only latest infection time
    # can be generalized to other times
    return t_min - shortest_distance(g, source=cand_source, target=earliest_node)


def sample_consistent_cascade(g, obs_nodes, cand_source, infection_times, debug=False):
    tree_paths = []
    ts_max = max_infection_time(g, infection_times, obs_nodes, cand_source, debug)
    
    if debug:
        print('observed infection times {}'.format({o: infection_times[o] for o in obs_nodes}))
        print('max(t_s) = {}'.format(ts_max))

    # ranked by infection time in ascending order
    pred_infected_nodes = {cand_source}
    pred_infection_time = {cand_source: ts_max}
    for o in obs_nodes:
        pred_infection_time[o] = infection_times[o]
    
    for o in sorted(obs_nodes, key=infection_times.__getitem__):
        if debug:
            print('o={}'.format(o))
        succeed = False
        # try node from late to early
        # in order to maximize path re-use
        for op in sorted(pred_infected_nodes,
                         key=pred_infection_time.__getitem__,
                         reverse=True):
            if pred_infection_time[op] >= infection_times[o]:
                if debug:
                    print('t(op) >= t(o): {} >= {}\ntry next...'.format(
                        pred_infection_time[op], infection_times[o]))
                continue
            if op == cand_source:
                length = infection_times[o] - ts_max
            else:
                length = infection_times[o] - pred_infection_time[op]

            if debug:
                print('try connecting {} and {} with length {}'.format(op, o, length))

            d = shortest_distance(g, source=op, target=o)

            if d > length:
                if debug:
                    print('however d({}, {})={} > {}: impossible'.format(o, op, d, length))
                continue

            # cannot visit later nodes and itself
            forbidden_nodes = {u for u in obs_nodes
                               if infection_times[u] >= infection_times[o] and u != o}
            
            # cannot visit nodes on accumulated paths
            forbidden_nodes |= {u for p in tree_paths
                                for u in p
                                if u != op and u != o}
            
            paths = all_simple_paths_of_length(g, op, o,
                                               length=length,
                                               forbidden_nodes=forbidden_nodes,
                                               debug=False)
            try:
                path = next(paths)
                if debug:
                    # assert len(path) - 1 == length, "{} != {}".format(len(path) - 1, length)
                    # pred_inf_time = ts_max + length + infection_times[op]
                    # assert pred_inf_time == infection_times[o], \
                    #     "{} != {}".format(pred_inf_time, infection_times[o])
                    print('connect {} and {} via {}'.format(op, o, path))
                succeed = True
                break
            except StopIteration:
                # continue trying
                if debug:
                    print('unable to find such path')
                pass
        if succeed:
            tree_paths.append(path)

            # update predicted infection time
            for l, u in enumerate(path):
                if u in pred_infection_time:
                    assert pred_infection_time[u] == pred_infection_time[op] + l, \
                        'update t({}): {} != {} + {}'.format(
                            u,
                            pred_infection_time[u],
                            pred_infection_time[op],
                            l)
                pred_infection_time[u] = pred_infection_time[op] + l
            pred_infected_nodes |= set(path)
        else:
            # failed to find a path
            return None
        
    edges = set([(u, v) for p in tree_paths for u, v in zip(p[:-1], p[1:])])
    efilt = np.array([(((int(u), int(v)) in edges) or ((int(v), int(u)) in edges))
                      for u, v in g.edges()],
                     dtype=bool)
    gv = GraphView(g, efilt=efilt)

    if debug:
        print(obs_nodes)
    return gv
