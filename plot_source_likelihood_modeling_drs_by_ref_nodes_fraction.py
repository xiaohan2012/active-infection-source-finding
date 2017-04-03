
# coding: utf-8

# In[31]:
import matplotlib as mpl
mpl.use('pdf')

import sys
import os
import numpy as np
from matplotlib import pyplot as plt


# In[36]:

gtype = sys.argv[1]
param = '2-6'
drs_dirname = 'source-likelihood-drs'
single_dirname = 'source-likelihood'
output_dir = 'figs/source-likelihood-drs-by-ref-nodes-fraction/{}'.format(gtype)


# In[34]:

fs = np.linspace(0.1, 0.4, 4)
data_list = [np.load('outputs/{}/{}/{}-f_ref-{:.1f}.npz'.format(drs_dirname,
                                                                gtype,
                                                                param, f))
             for f in fs]

# result from single observation method
data_list.append(np.load('outputs/{}/{}/{}.npz'.format(single_dirname,
                                                       gtype,
                                                       param)))
X, Y = data_list[0]['arr_0'], data_list[0]['arr_1']
legends = ['pair(f={:.1f})'.format(f) for f in fs]
legends.append('single obs')

to_plot_data = {
    'ratio_median': [d['arr_2'] for d in data_list],
    'ratio_mean': [d['arr_3'] for d in data_list],
    'dist_median': [d['arr_4'] for d in data_list],
    'dist_mean': [d['arr_5'] for d in data_list]
}


for name, ms in to_plot_data.items():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    per_size, nrow, ncol = 5, 5, 2
    fig = plt.figure(figsize=(ncol * per_size, (nrow+1) * per_size))

    for i, p in enumerate(np.linspace(0.1, 1.0, 10)):
        lines = []
        ax = fig.add_subplot(nrow, ncol, i+1)
        for m in ms:
            l, = ax.plot(np.linspace(0.1, 1.0, 10),
                         m[i, :], 'o-')
            lines.append(l)
        
        ax.set_title('p={:.1f}'.format(p))
        ax.set_xlabel('q')
        ax.set_ylabel(name)
        if name.startswith('ratio'):
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])

    fig.legend(lines, legends, loc='upper left')
    fig.savefig('{}/{}-by-q.pdf'.format(output_dir, name))
    
    per_size, nrow, ncol = 5, 5, 2
    fig = plt.figure(figsize=(ncol * per_size, nrow * per_size))
    for i, q in enumerate(np.linspace(0.1, 1.0, 10)):
        lines = []
        ax = fig.add_subplot(nrow, ncol, i+1)
        for m in ms:
            l, = ax.plot(np.linspace(0.1, 1.0, 10),
                         m[:, i], 'o-')
            lines.append(l)
        ax.set_title('q={:.1f}'.format(q))
        ax.set_xlabel('p')
        ax.set_ylabel(name)
        if name.startswith('ratio'):
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
    fig.legend(lines, legends, loc='upper left')
    fig.savefig('{}/{}-by-p.pdf'.format(output_dir, name))
print('saved to {}'.format(output_dir))
