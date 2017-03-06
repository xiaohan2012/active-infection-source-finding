# active infection source finding

# Todo (optimization)

- `s2n_times` using *3D tensor* to save memory and *parallellize* for speedup




# TODO (code structure)

- done: `simulations.py` and `cascade.py` should be merged into "ic.py"
- `synthetic_data.py` and `core.py`


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


# infection\_time\_distribution

Note: *uninfected nodes* are excluded from the following analysis.

when `p=0.7`

- correlation betweenlength of shortest path and ratio of infected nodes: weakly positive
- infection time distribution, the shape looks like Poisson distribution

When `p=0.4`

- the correlation is strong
- Again, the distribution looks like Poission distribution

# Query strategy: select the node with maximum weight

Using late nodes sampling method, 
epsilon needs to be large enough (>0.6) to beat the baseline.

When eps = 0.8, the query count is almost half.

However, there are certain cases there it queries almost **every** node. 

The reason is:

Because of the stochasticity, it's possible that some non-source node explains  the cascade better than the actual source.
Note, the process is random, thus a source might produce a cascade that is unlikely to happen.

To make it more robust, we can combine baseline algorithm with this method. 

# Query selection strategy: source likelihood convergence speed

# Query strategy running time

To query all nodes (grid):

- Random: 1s
- Min consensus: ~20s

For Kronecker graph, it is:

- Min consensus: 2min 37s

# Findings

- For cliques, you need to query all the nodes.
- Remember to set mu to zero.


# Other issues

- wheter deciding if node is source, the queried neighbors can be used to update mu as well.
- efficient implementation of generalized Jaccard similarity
- `networkx` `nodes\_iter` and `Parallel`
   - the `nodes\_iter` order is inconsistent with and without Parallel

