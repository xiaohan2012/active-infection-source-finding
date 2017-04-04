# coding: utf-8

import matplotlib as plt
plt.use('pdf')
from matplotlib import pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm

from synthetic_data import load_data_by_gtype
from ic import make_partial_cascade
from graph_generator import add_p_and_delta
from plot_utils import richify_line_style


# In[58]:

ps = np.linspace(0.1, 1.0, 19)


# In[60]:

def experiment(g, p, N):
    numer, denum = 0, 0
    g = add_p_and_delta(g, p, 1)
    for i in range(N):
        source, _, infection_times, _ = make_partial_cascade(g, 0.01)
        sp_len = nx.shortest_path_length(g, source=source)

        total_infection = sum(1 for l in infection_times.values() if not np.isinf(l))
        n_matches = sum(1 for n, l in infection_times.items()
                        if not np.isinf(l) and sp_len[n] == l)
        numer += n_matches
        denum += total_infection
    return numer, denum


# In[61]:

gtypes = [('grid', '2-4'),
          ('er', '2-8'), 
          ('barabasi', '2-8'),
          ('pl-tree', '2-8'),
          ('kr-hier', '10-10'),
          ('kr-peri', '10-10'),
          ('kr-rand', '10-10')]
result = {}
for gtype, param in tqdm(gtypes):
    g = load_data_by_gtype(gtype, param)[0]
    rows = Parallel(n_jobs=-1)(delayed(experiment)(g, p, 100) for p in ps)
    result[gtype] = np.array(rows)


# In[66]:

result['grid']


# In[65]:

richify_line_style(plt)
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
for g, m in result.items():
    ax.plot(ps, m[:, 0] / m[:, 1])
ax.legend(list(result.keys()), loc='lower right')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_title('infection probability vs fraction of shortest path')
ax.set_xlabel('probability')
ax.set_ylabel('fraction')

fig.savefig('figs/p-vs-sp-path-fraction.pdf')
