import numpy as np
from tqdm import tqdm
from steiner_tree_mst import init_visitor, get_edges
from graph_tool import Graph
from graph_tool.search import cpbfs_search


def build_truncated_closure(g, cand_source, terminals, infection_times,
                            k=-1,
                            debug=False,
                            verbose=False,
                            **kawrgs):
    """
    build a clojure graph in which cand_source + terminals are all connected to each other.
    the number of neighbors of each node is determined by k

    the larger the k, the denser the graph"""
    r2pred = {}
    edges = {}
    terminals = list(terminals)

    # from cand_source to terminals
    vis = init_visitor(g, cand_source)

    cpbfs_search(g, source=cand_source, visitor=vis, terminals=terminals,
                 forbidden_nodes=terminals,
                 count_threshold=-1)  # k=-1 here because root connects to all other nodes
    r2pred[cand_source] = vis.pred
    for u, v, c in get_edges(vis.dist, cand_source, terminals):
        edges[(u, v)] = c
    if debug:
        print('cand_source: {}'.format(cand_source))
        print('#terminals: {}'.format(len(terminals)))
        print('edges from cand_source: {}'.format(edges))

    if verbose:
        terminals_iter = tqdm(terminals)
        print('building closure graph')
    else:
        terminals_iter = terminals

    # from terminal to other terminals
    # every temrinal should connetct to at least one earlier terminal
    # in this way, connectivity is ensured
    for root in terminals_iter:
        if root == cand_source:
            continue            
        # connect from some earlier node to root

        # if it's earliest, can only connect to peers
        early_terminals = [t for t in terminals
                           if infection_times[t] < infection_times[root]]
        same_time_terminals = [t for t in terminals
                               if infection_times[t] == infection_times[root] if t != root]
        late_time_terminals = [t for t in terminals
                               if infection_times[t] > infection_times[root]]
        if debug:
            print('root: {}'.format(root))
            print('early_terminals: {}'.format(early_terminals))
            print('same_time_terminals: {}'.format(same_time_terminals))
            print('late_time_terminals: {}'.format(late_time_terminals))

        if infection_times[root] == infection_times[terminals].min():
            targets = early_terminals + same_time_terminals
        else:
            targets = early_terminals

        targets = list(set(targets) - {cand_source})  # no one can connect to cand_source

        if debug:
            print('targets: {}'.format(targets))
            
        vis = init_visitor(g, root)
        cpbfs_search(g, source=root, visitor=vis,
                     terminals=targets,
                     forbidden_nodes=late_time_terminals,
                     count_threshold=k)
        r2pred[root] = vis.pred
        for root, v, c in get_edges(vis.dist, root, early_terminals):
            if debug:
                print('edge ({}, {})'.format(v, root))
            edges[(v, root)] = c  # from earlier node to root

    if verbose:
        print('returning closure graph')

    gc = Graph(directed=True)

    for _ in range(g.num_vertices()):
        gc.add_vertex()

    for (u, v) in edges:
        gc.add_edge(u, v)

    eweight = gc.new_edge_property('int')
    eweight.set_2d_array(np.array(list(edges.values())))

    return gc, eweight, r2pred
