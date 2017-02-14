# active infection source finding

## Pagerank vs BFS

`bfs_source_finding.ipynb` and `pagerank_source_finding.ipynb`

Stopping criteria: when the current query node is the source.

Pagerank is better than BFS. For example, the mean query ratio against cascade size is 0.2 and 0.32 for PR and BFS respectively.

### Issues

1. In reality, the stopping criteria is not realistic
2. Why pagerank approach is better than BFS-approach?
   - one possibility: PR uses information of uninfected nodes while BFS doesn't. What if BFS also uses this information?
3. What's the intuition of pagerank?

## Cascade size and source degree

The larger the source degree, the larger the cascade size.
This seems obvious.

Check out `cascade_size_vs_source_infected_neighbors.ipynb`

Why this is useful?



## Core number vs source's neighbors' infection time mean and standard deviation.

The larger the source core number, the higher the mean and std of its neighbors infection time.

To generalize, any node with high core number has high mean and std.

Why is this useful?


Check out `cascade_size_vs_source_infected_neighbors.ipynb`

# Where are we?

- Performance metric priority:
  - correctness
    - able to find the source
  - number of queries should be minimized
  - simplicity and applicable to various models
    - a simple method can be applied to a variety of models
    - single p
    - different p one edges
  - robustness to
    - different node sampling methods
  - computation speed
    - if online fashion (new observations are generated on the fly), then speed is important

# Possible directions:

Bounding the likelihood:

- Can we derive some *lower bound* on the likelihood of on-edge nodes?
- Can we derive some *upper bound* on the likelihood of off-edge nodes?

Or some mean if the likelihood function is exponentially distributed

Particle filter:

- Can we use it here?

# Problems

- ZeroDivisionError: all mu drops to zero so total is zero
- Why the same input produces different querying strategy?
- Does the sampling method converge?


# Possible improvements:

- consider structure when calculating p_mu, not just p
- neighbor query order: query node should be close the earlier infected nodes
- different insitializations on mu
  - easier but faster way?
- baseline starts with the node with highest mu?
  - need to justify the multiplicative weight algorithm.
- uninfected nodes
  - nearby nodes decrease mu?
- another baseline:
  - iteratively adding new observations
  - and infer the new source likelihood


# Different settings

- sampling by:
  - node degree: high degree nodes are more likely to be sampled
  - infection time: later nodes are more likely to be sampled
  - uniform

# Better visualization

- Source node (square or star)
- Query and response (different colors) with an arrow from query to response
- Path nodes bigger
- observed nodes (circle with tick inside it)