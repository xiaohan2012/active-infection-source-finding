# active infection source finding

# pre-processing

snap -> gml -> gt:

1. `convert_snap_network.py`: to gml
2. `convert_graphml_to_gt.py`: to gt

# debugging/profiling

`scripts/test_paper_experiment.sh`

# scripts for ordered steiner tree

- `/home/cloud-user/documents/order-steiner-tree/new_result.tex` (result document)
  - [url](http://193.166.24.212/steiner-new-result.pdf)
- `scripts/gen_paper_experiment_cmds.sh` (needs to be edited)
- `scripts/gen_eval_cmds.sh` (needs to be edited)
- `paper_experiment_plot.ipynb`: to plot
- `crop.sh`: to crop figures

done:

- (`q` varies): P2P, axiv-hep

# Important scripts

- `query_count_vs_graph_size.py`
- `query_count_on_dataset.py`

## for source likelihood modeling

simulation:

- single obs: `how-well-can-we-model-probability.py`
- DRS: `how-well-can-we-model-probability-drs.py`

plots:

  - cutting plane plot (comparing different modeling): `plot_source_likelihood_modeling_comparison_2d.py`
  - surface plot by graph types and sizes: `plot_source_likelihood_modeling_by_graphs_and_sizes.py`
  - surface plot by graph types: `plot_source_likelihood_modeling.py`



# Todo

- `edge_mwu: why it` sucks when setting `mu[q]=0` for query node that is not source.
- `edge_mwu: when doing neighborhood querying, why updating at each neighbor sucks?
- `max_mu`: why it's so unstable?
- `tree_binary_search`: maximum recursion exceeded

# Todo (optimization)

- `edge_mwu`

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

# Using Harmonic mean

Penalty definition: `abs(hmean - outcome)`

# Other issues

- wheter deciding if node is source, the queried neighbors can be used to update mu as well.
- efficient implementation of generalized Jaccard similarity
- `networkx` `nodes\_iter` and `Parallel`
   - the `nodes\_iter` order is inconsistent with and without Parallel


# preprocessing issue

# parallel processing for large graph

- each job just load what it needs. Otherwise, data loading can be time consuming
- parallel appending to the same file is fine. When the appended content is small (under `PIPE_BUF`), [no need to use file lock](http://stackoverflow.com/questions/1154446/is-file-append-atomic-in-unix)


# Installing `graph-tool`

- `sudo apt install -y libcgal-dev`
- `sudo apt install -y libcairo2-dev`
- `sudo apt install -y libcairomm-1.0`
- `sudo apt install -y libcairomm-1.0-dev`
- `sudo apt install -y python3-cairo`
- `sudo apt install -y python3-cairo-dev`
- `sudo apt install -y libsparsehash-dev`
- also `python3-gi python3-click python3-gi-cairo python3-cairo gir1.2-gtk-3.0`

Then:

- `./configure  CXXFLAGS="-std=gnu++14"`
- [https://git.skewed.de/count0/graph-tool/issues/359](the reason with the flag)