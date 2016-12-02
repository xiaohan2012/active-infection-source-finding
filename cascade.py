import random


def generate_cascade(g):
    g = g.copy()
    source = random.choice(g.nodes())
    infected = {source}
    infected_times = {source: 0}
    iter_n = 0
    while True:
        iter_n += 1
        newly_infected = set()
        for u in infected:
            can_continue = False
            for v in g.neighbors(u):
                if v not in infected and not g[u][v].get('attempted', False):
                    # print('infected node: {}'.format(v))
                    can_continue = True
                    p_uv = 0.5  # proba of getting infected
                    if random.random() < p_uv:
                        newly_infected.add(v)
                        infected_times[v] = iter_n
                    g[u][v]['attempted'] = True
        infected |= newly_infected
        if not can_continue:
            break
    return infected_times
