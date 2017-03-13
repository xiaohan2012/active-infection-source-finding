
# coding: utf-8

# In[14]:

import networkx as nx
import pandas as pd
from tqdm import tqdm
from mwu import main_routine as mwu, MAX_MU, RANDOM
from synthetic_data import (load_data_by_gtype, GRID, KRONECKER_HIER, KRONECKER_PERI, KRONECKER_RAND,
                            PL_TREE, ER, BARABASI)
from experiment_utils import experiment_mwu_multiple_rounds, experiment_dog_multiple_rounds


# In[11]:

def counts_to_stat(counts):
    s = pd.Series(list(filter(lambda c: c != False, counts)))
    return s.describe().to_dict()


# In[42]:

n_rounds = 100
size_params = ['2-{}'.format(i) for i in range(2, 8)]
gtype = GRID
fracition = 0.01
epsilon = 0.7
sampling_method = 'late_nodes'
check_neighbor_threshold = 0.01

rows = []
indices = []
for size_param in size_params:
    print(size_param)
    try:
        g, time_probas, node2id, id2node = load_data_by_gtype(gtype, size_param)
    except IOError:
        print('fail to load {}/{}'.format(gtype, size_param))
        break
    def mwu_wrapper(method):
        return experiment_mwu_multiple_rounds(n_rounds, g, node2id, id2node, time_probas,
                                              fraction=fracition, epsilon=epsilon,
                                              sampling_method=sampling_method,
                                              query_selection_method=method, 
                                              check_neighbor_threshold=check_neighbor_threshold,
                                              max_iter=g.number_of_nodes())
    
    
    counts = mwu_wrapper(MAX_MU)
    rows.append(counts_to_stat(counts))    
    indices.append((MAX_MU, g.number_of_nodes()))

    counts = mwu_wrapper(RANDOM)
    rows.append(counts_to_stat(counts))    
    indices.append((RANDOM, g.number_of_nodes()))
    
    counts = experiment_dog_multiple_rounds(n_rounds, g, fracition, sampling_method)
    rows.append(counts_to_stat(counts))
    indices.append(('dog', g.number_of_nodes()))
    
index = pd.MultiIndex.from_tuples(indices, names=('method', 'graph size'))
df = pd.DataFrame.from_records(rows, index=index)


# In[ ]:

df.to_pickle('data/{}/performance.pkl'.format(gtype))


# In[43]:

df

