"""infection model following shortest path"""

import numpy as np
from ic import simulate_cascade


def gen_cascade(g, source=None, fraction=0.5):
    source, infection_times, tree = simulate_cascade(
        g, 1.0, source=source, return_tree=True)

    t = 1
    while (np.count_nonzero(infection_times <= t) / g.num_vertices()) <= fraction:
        t += 1
    infection_times[infection_times > t] = -1

    filtered_nodes = np.nonzero(infection_times == -1)[0]
    vfilt = tree.new_vertex_property('bool')
    vfilt.a = True
    for n in filtered_nodes:
        vfilt[n] = False
    tree.set_vertex_filter(vfilt)
    return source, infection_times, tree
