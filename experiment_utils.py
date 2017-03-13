import numpy as np
import random
from ic import make_partial_cascade
from mwu import main_routine as mwu
from tqdm import tqdm
from baselines import baseline_dog_tracker


def experiment_mwu_multiple_rounds(rounds, g,
								   node2id, id2node,
								   s2n_probas,
								   fraction, epsilon, sampling_method,
								   query_selection_method,
								   check_neighbor_threshold,
								   max_iter=float('inf'),
								   seed=None):
    np.random.seed(seed)
    random.seed(seed)
    results = []
    for i in tqdm(range(rounds)):
        source, obs_nodes, infection_times, tree = make_partial_cascade(
			g, fraction, sampling_method=sampling_method)
        r = mwu(g, node2id, id2node,
                source, obs_nodes, infection_times,
                s2n_probas,
                epsilon,
                query_selection_method=query_selection_method,
                debug=False,
                max_iter=max_iter,
                save_log=False)
        results.append(r)
    return results

def experiment_dog_multiple_rounds(rounds, g, fraction, sampling_method):
	cnts = []
	for i in range(rounds):
		source, obs_nodes, infection_times, tree = make_partial_cascade(
			g, fraction, sampling_method=sampling_method)
		c = baseline_dog_tracker(g, obs_nodes, infection_times)
		cnts.append(c)
	return cnts
