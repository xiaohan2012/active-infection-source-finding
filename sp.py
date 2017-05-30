"""infection model following shortest path"""

import numpy as np
import random
from graph_tool.all import shortest_distance


def gen_cascade(g, source=None, fraction=0.5):
    if source is None:
        source = random.choice(np.arange(g.num_vertices()))

    length = shortest_distance(g, source=source).a
    t = 1
    while (np.count_nonzero(length <= t) / g.num_vertices()) <= fraction:
        t += 1
    infection_times = np.array(length)
    infection_times[infection_times > t] = -1
    return source, infection_times
