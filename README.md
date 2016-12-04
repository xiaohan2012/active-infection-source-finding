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