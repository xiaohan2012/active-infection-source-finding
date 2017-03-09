import random


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
            # print('**Found** source: {}'.format(source))
            assert source == expected
            break
    return query_count


def random_dog_tracker(g, obs_nodes, infection_times, max_degree):
    pass
