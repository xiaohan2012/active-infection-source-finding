import random
from copy import copy


def random_dog(g, obs_nodes, infection_times, fraction,
               debug=False, save_logs=False):
    """fraction: the fraciton of neighbors to query.
    the parameter that controls how dog operetes
    """
    queried_nodes = copy(obs_nodes)
    if save_logs:
        query_node_list = []
    q = min(obs_nodes, key=lambda n: infection_times[n])
    while True:
        if debug:
            print('query node: {}'.format(q))
        queried_nodes.add(q)
        if save_logs:
            query_node_list.append(q)
            
        found_source = True
        nbs = g.neighbors(q)
        random.shuffle(nbs)
        limit = int(round(fraction * len(nbs)))

        if debug:
            print('limit: {}'.format(limit))

        nodes_to_query = nbs[:limit]

        min_time = infection_times[q]
        node_to_follow = None

        # query the fraction
        for u in nodes_to_query:
            if debug:
                print('query nbr node (fraction): {}'.format(u))
            if save_logs:
                query_node_list.append(u)

            queried_nodes.add(u)
            if infection_times[u] < min_time:
                found_source = False
                min_time = infection_times[u]
                node_to_follow = u

        # if no node is earlier,
        # continue querying until finding one
        if node_to_follow is None:
            for u in nbs[limit:]:
                if debug:
                    print('query nbr node (have to): {}'.format(u))
                if save_logs:
                    query_node_list.append(u)

                queried_nodes.add(u)
                if infection_times[u] < min_time:
                    found_source = False
                    q = u
                    break

            if found_source:
                expected = min(infection_times, key=lambda n: infection_times[n])
                assert q == expected, '{} != {} (expected)'.format(q, expected)
                break
        else:
            # follow it
            q = node_to_follow

    query_count = len(queried_nodes - set(obs_nodes))
    if save_logs:
        return query_count, query_node_list
    else:
        return query_count


def baseline_dog_tracker(g, obs_nodes, infection_times):
    query_count = 0
    q = min(obs_nodes, key=lambda n: infection_times[n])
    while True:
        if q not in obs_nodes:
            query_count += 1
            obs_nodes.add(q)
            
        found_source = True
        nbs = g.neighbors(q)
        random.shuffle(nbs)
        for u in nbs:
            if u not in obs_nodes:  # need to query it
                obs_nodes.add(u)
                query_count += 1

            if infection_times[q] > infection_times[u]:
                # q later than u
                found_source = False
                # found the direction
                if infection_times[q] - infection_times[u] == g[q][u]['d']:
                    q = u
                    break
        if found_source:
            source = q
            expected = min(infection_times, key=lambda n: infection_times[n])
            assert source == expected
            break
    return query_count
