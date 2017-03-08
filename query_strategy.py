import numpy as np
import pickle as pkl
from collections import Counter
from utils import generalized_jaccard_similarity, weighted_sample_with_replacement


# @profile
def consensus_score(q, times_by_source, mu, node2id, source2nodeid_counter):
    """return the consensus value from different sources on node q"""
    qid = node2id[q]
    M, N = times_by_source[next(iter(times_by_source.keys()))].shape
    
    np.testing.assert_almost_equal(sum(mu.values()), 1.0)
    
    nodes = list(node2id.keys())
    # node_indices = list(node2id.values())
    weights = np.asarray([mu[n] for n in nodes])
    sampled_experts = weighted_sample_with_replacement(nodes, weights, M)
    # [nodes[np.random.choice(node_indices, p=weights)] for _ in range(M)]

    # the "centroid" belief on q's infection time
    centroid = np.zeros(M)
    for i, e in enumerate(sampled_experts):
        time = np.random.choice(times_by_source[e][:, qid])
        centroid[i] = time

    centroid = Counter(centroid)
    similarity_scores = np.zeros(N)

    for i, n in enumerate(nodes):
        similarity_scores[i] = generalized_jaccard_similarity(
            centroid, source2nodeid_counter[n][qid])
    return np.mean(similarity_scores * weights)  # weighted mean


def centroid(cand_nodes, g, mu, sp_len):
    """return the centroid node:
    argmin_{q \in queryable} sum_{i \in V} mu[i] sp_len[q][i]
    """
    def sum_of_weighted_dist(q):
        mus = np.array([mu[v] for v in g.nodes_iter()])
        lens = np.array([sp_len[q][v] for v in g.nodes_iter()])
        return np.sum(mus * lens)
    
    return min(cand_nodes, key=sum_of_weighted_dist)


def expected_infection_time(mu, s2n_probas):
    """
    mu: probability distribution over experts
    s2n_probas: 3D tensor, source x node x infection time (probability distribution)

    Returns:
    2D: node x infection time distribution
    """
    n2s_probas = np.swapaxes(s2n_probas, 0, 1)
    n2s_probas.shape
    return np.tensordot(mu, n2s_probas, axes=1)
    

if __name__ == "__main__":
    from graph_generators import grid_2d
    
    g = grid_2d(10, 0.7)
    times_by_source = pkl.load(open('outputs/time_by_source.pkl', 'rb'))
    source2nodeid_counter = pkl.load(open('outputs/source2nodeid_counter.pkl', 'rb'))
    node2id = {n: i for i, n in enumerate(g.nodes_iter())}
    
    mu = {n: 1 / g.number_of_nodes() for n in g.nodes_iter()}
    for _ in range(10):
        consensus_score((0, 3), times_by_source, mu, node2id, source2nodeid_counter)
