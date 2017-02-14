import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt


def get_display_kwargs(g, infected_times, obs_nodes=set(), max_node_size=1000):
    diff = 1
    times = np.array(list(infected_times.values()))
    times = times[np.invert(np.isinf(times))]
    upper_bound = times.max() + diff
    node_colors = [((upper_bound - infected_times[n] + diff) if not np.isinf(infected_times[n]) else 0)
                   for n in g.nodes()]

    def node_size(n):
        if n in obs_nodes:
            return max_node_size
        elif not np.isinf(infected_times[n]):
            return max_node_size / 4
        else:
            return max_node_size / 10
    node_sizes = list(map(node_size, g.nodes()))
    return {'node_size': node_sizes,
            'node_color': node_colors,
            'cmap': 'OrRd'}


# In[6]:

def add_colorbar(cvalues, cmap='OrRd'):
    eps = np.maximum(0.0000000001, np.min(cvalues)/1000.)
    vmin = np.min(cvalues) - eps
    vmax = np.max(cvalues)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    scm = mpl.cm.ScalarMappable(norm, cmap)
    scm.set_array(cvalues)
    plt.colorbar(scm)    
