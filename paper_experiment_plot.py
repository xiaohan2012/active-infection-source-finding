
# coding: utf-8

# In[13]:

get_ipython().magic('matplotlib notebook')
import matplotlib as mpl
mpl.use('pdf')
# mpl.rcParams.update({'font.size': 26})

import os
import pandas as pd
import numpy as np
from glob import glob
from matplotlib import pyplot as plt
from itertools import product, takewhile
from collections import defaultdict
from plot_utils import richify_line_style


# In[14]:

from cycler import cycler
plt.rc('axes',
       prop_cycle=(
           cycler('color', ['b', 'g', 'r', 'y', 'g', 'b']) +
           cycler('linestyle', ['-', '-', '-', '-', '--', '-']) +
           cycler('marker', ['', '', '', '', 'o', '*'])
       ))


# In[35]:

def plot_performance(datasets, models, methods, qs, column_names, savefig=False, savelegend=False):
    dirname_template = "outputs/paper_experiment/{dataset}/{model}/{method}/qs/{q}.pkl"
    result = {}
    for dataset, model in product(datasets, models):
        key = (dataset, model)
        result[key] = {}
        for method in methods:
            result[key][method] = []
            for q in qs_str:
                path = dirname_template.format(dataset=dataset, model=model, method=method, q=q)
                try:
                    result[key][method].append(pd.read_pickle(path))
                except FileNotFoundError:
                    dummy = defaultdict(lambda :defaultdict(lambda: None))
                    result[key][method].append(dummy)

    nrow = (len(datasets) if len(datasets) > 1 else len(models))
    per_width, per_height, ncol = 3, 2.2, len(column_names)
    for i, (dataset, model) in enumerate(product(datasets, models)):
        key = (dataset, model)
        # one plot
        for j, column in enumerate(column_names):
            if i == 0:
                lines = []
            idx = i * ncol + j + 1
            fig = plt.figure(figsize=(per_width, per_height))
            ax = fig.add_subplot(111)
            for method in methods:
                # one legend
                try:
                    ys = [df[column]['mean'] for df in result[key][method]]                
                except KeyError:
                    print(dataset, method, model)
                    raise KeyError
                lst = list(zip(*takewhile(lambda tpl: tpl[1] is not None, zip(qs, ys))))
                if lst:
                    correct_qs, correct_ys = lst
                    l, = ax.plot(correct_qs, correct_ys, markersize=7.5, alpha=0.75)
                    if i == 0:
                        lines.append(l)
                else:
                    continue        
            # ax.set_title(column)
            column_map = {'n.prec': 'n.precision', 'n.rec': 'n.recall', 'cos-sim': 'cosine sim',
                          'e.prec': 'e.precision', 'e.rec': 'e.recall',
                          'obj': '|edges|', 'rank-corr': 'order corr'}
            ax.set_ylabel(column_map.get(column, column))
            ax.locator_params(axis='y', nbins=4)

            ax.set_xlabel('prop. of reports')
            ax.locator_params(axis='x', nbins=4)                

            # ylim = (0, 1)
            ylim = None
            if column in {"n.prec"}:
                ylim  = (0.5, 1.05)
            if column in {'obj'}:
                ylim = None
                
            if ylim is not None:
                ax.set_ylim(*ylim)
                pass

            if column == 'obj':
                yticks = ax.get_yticks()
                if yticks.max() > 100:
                    ax.set_yticklabels(list(map(lambda v: "{:.1f}k".format(v / 1000), yticks)))
            fig.tight_layout()
            import os
            dirname = 'figs/paper_experiment/'
            if len(datasets) > 1:
                dirname += 'by_datasets/'
                path = dirname + '{}-{}.pdf'.format(dataset, column_map.get(column, column).replace('|', '').replace(' ', '-'))
            else:
                dirname += '{}_by_models/'.format(datasets[0])
                path = dirname + '{}-{}.pdf'.format(model, column_map.get(column, column).replace('|', '').replace(' ', '-'))
            print(dirname)
            print(path)
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            if savefig:
                fig.savefig(path)
# name_mapping = {'no-order': 'baseline', 'mst': 'closure', 'tbfs': 'delay-bfs', 'greedy': 'greedy'}
# legends = list(map(lambda m: name_mapping[m], methods))

# plt.tight_layout()

# fig.legend(lines, legends, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01), frameon=True)

# if len(datasets) > 1:
#     fig.savefig('figs/paper_experiment/measure_by_datasets.pdf')
# else:
#     print('figs/paper_experiment/{}_measure_by_models.pdf'.format(datasets[0]))
#     fig.savefig('figs/paper_experiment/{}_measure_by_models.pdf'.format(datasets[0]))

    figlegend = plt.figure(figsize=(2 * len(methods), 0.5))
    ax = fig.add_subplot(111)

    name_mapping = {'no-order': 'baseline', 'mst': 'closure', 'tbfs': 'delay-bfs', 'greedy': 'greedy'}
    legends = list(map(lambda m: name_mapping[m], methods))

    figlegend.legend(lines, legends, 'center', ncol=len(legends))
    fig.show()
    figlegend.show()
    
    if savelegend:
        figlegend.savefig(dirname + 'legend.pdf')
    return result


# In[41]:

if False:
    datasets = ['arxiv-hep-th', 'enron-email']
    models = ['si']
    methods = ["greedy", "tbfs"] 
else:
    # datasets = ['p2p-gnutella08']
    datasets = ['barabasi-64']
    models = ['si', 'ct']
    methods = ["mst", "tbfs", "no-order"] 
if datasets[0] not in {'barabasi-64', 'grid-64'}:
    qs = np.linspace(0.005, 0.1, 20)
else:
    qs = np.linspace(0.1, 1.0, 19)
qs_str = list(map(str, qs))
print(qs_str)
column_names = []
# column_names += ['n.prec', 'n.rec', 'obj']
# column_names = ['obj', 'rank-corr', 'n.prec', 'n.rec']
# column_names = ['rank-corr']
# column_names = ['n.prec-t', 'n.rec-t']
# column_names += ['e.prec', 'e.rec']
column_names += ['rank-corr', 'order accuracy']

result = plot_performance(datasets, models, methods, qs, column_names, savefig=True)

