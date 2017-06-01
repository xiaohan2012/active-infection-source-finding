import networkx as nx
import numpy as np
from tqdm import tqdm
from graph_tool.all import GraphView, BFSVisitor, Graph
from graph_tool.search import cpbfs_search, bfs_iterator
from utils import gt2nx


def extract_edges_from_pred(g, source, target, pred):
    """edges from target to source"""
    edges = []
    c = target
    while c != source and pred[c] != -1:
        edges.append((pred[c], c))
        c = pred[c]
    return edges


def extract_tree(g, source, pred, terminals=None):
    """return a tree from source to terminals based on `pred`"""
    edges = set()

    if terminals:
        visited = set()
        for t in sorted(terminals):
            c = t
            while c != source and c not in visited:
                visited.add(c)
                if pred[c] != -1:
                    edges.add((pred[c], c))
                    c = pred[c]
                else:
                    break
    else:
        for c, p in enumerate(pred.a):
            if p != -1:
                edges.add((c, p))
    efilt = g.new_edge_property('bool')
    for u, v in edges:
        efilt[g.edge(g.vertex(u), g.vertex(v))] = 1
    return GraphView(g, efilt=efilt)


class MyVisitor(BFSVisitor):

    def __init__(self, pred, dist):
        """np.ndarray"""
        self.pred = pred
        self.dist = dist

    def black_target(self, e):
        s, t = int(e.source()), int(e.target())
        if self.pred[t] == -1:
            self.pred[t] = s
            self.dist[t] = self.dist[s] + 1
    
    def tree_edge(self, e):
        s, t = int(e.source()), int(e.target())
        self.pred[t] = s
        self.dist[t] = self.dist[s] + 1


def init_visitor(g, root):
    dist = np.ones(g.num_vertices()) * -1
    dist[root] = 0
    pred = np.ones(g.num_vertices(), dtype=int) * -1
    vis = MyVisitor(pred, dist)
    return vis

# @profile
def get_edges(dist, root, terminals):
    return ((root, t, dist[t])
            for t in terminals
            if dist[t] != -1 and t != root)
    

def build_closure(g, cand_source, terminals, infection_times, k=-1,
                  strictly_smaller=True,
                  debug=False,
                  verbose=False):
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
                 count_threshold=k)
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
    for root in terminals_iter:
        vis = init_visitor(g, root)

        if strictly_smaller:
            late_terminals = [t for t in terminals
                              if infection_times[t] > infection_times[root]]
        else:
            # respect what the paper presents
            late_terminals = [t for t in terminals
                              if infection_times[t] >= infection_times[root]]

        late_terminals = set(late_terminals) - {cand_source}  # no one can connect to cand_source
        if debug:
            print('root: {}'.format(root))
            print('late_terminals: {}'.format(late_terminals))
        cpbfs_search(g, source=root, visitor=vis, terminals=list(late_terminals),
                     forbidden_nodes=list(set(terminals) - set(late_terminals)),
                     count_threshold=k)
        r2pred[root] = vis.pred
        for u, v, c in get_edges(vis.dist, root, late_terminals):
            if debug:
                print('edge ({}, {})'.format(u, v))
            edges[(u, v)] = c

    if verbose:
        print('returning closure graph')

    gc = Graph(directed=True)

    for _ in range(g.num_vertices()):
        gc.add_vertex()

    for (u, v) in edges:
        gc.add_edge(u, v)

    eweight = gc.new_edge_property('int')
    eweight.set_2d_array(np.array(list(edges.values())))
    # for e, c in edges.items():
    #     eweight[e] = c
    return gc, eweight, r2pred

def steiner_tree_mst(g, root, infection_times, source, terminals,
                     strictly_smaller=True,
                     return_closure=False,
                     k=-1,
                     debug=False,
                     verbose=True):
    gc, eweight, r2pred = build_closure(g, root, terminals,
                                        infection_times,
                                        strictly_smaller=strictly_smaller,
                                        k=k,
                                        debug=debug,
                                        verbose=verbose)

    # get the minimum spanning arborescence
    # graph_tool does not provide minimum_spanning_arborescence
    if verbose:
        print('getting mst')
    gx = gt2nx(gc, root, terminals, edge_attrs={'weight': eweight})
#     print('type', type(gx))
#     print('gx.edges()', gx.edges())
    try:
        nx_tree = nx.minimum_spanning_arborescence(gx, 'weight')
#         print('nx_tree.edges()', nx_tree.edges())
    except nx.exception.NetworkXException:
        if debug:
            print('fail to find mst')
        return None

    if verbose:
        print('returning tree')

    mst_tree = Graph(directed=True)
    for _ in range(g.num_vertices()):
        mst_tree.add_vertex()
    
    for u, v in nx_tree.edges():
        mst_tree.add_edge(u, v)

    if verbose:
        print('extract edges from original graph')

    # extract the edges from the original graph   
    
    # sort observations by time
    # and also topological order
    topological_index = {}
    for i, e in enumerate(bfs_iterator(mst_tree, source=root)):
        topological_index[int(e.target())] = i
    sorted_obs = sorted(
        set(terminals) - {root},
        key=lambda o: (infection_times[o], topological_index[o]))
        
    tree_nodes = {root}
    tree_edges = set()
    # print('root', root)
    for u in sorted_obs:
        # print(u)
        v, u = map(int, next(mst_tree.vertex(u).in_edges()))  # v is ancestor
        tree_nodes.add(v)
        
        late_nodes = [n for n in terminals if infection_times[n] > infection_times[u]]
        vis = init_visitor(g, u)
        # from child to any tree node, including v
        
        cpbfs_search(g, source=u, terminals=list(tree_nodes),
                     forbidden_nodes=late_nodes,
                     visitor=vis,
                     count_threshold=1)
        reachable_tree_nodes = set(np.nonzero(vis.dist > 0)[0]).intersection(tree_nodes)
        ancestor = min(reachable_tree_nodes, key=vis.dist.__getitem__)
        
        edges = extract_edges_from_pred(g, u, ancestor, vis.pred)
        edges = {(j, i) for i, j in edges}  # need to reverse it
        if debug:
            print('tree_nodes', tree_nodes)
            print('connecting {} to {}'.format(v, u))
            print('using ancestor {}'.format(ancestor))
            print('adding edges {}'.format(edges))
        tree_nodes |= {u for e in edges for u in e}

        tree_edges |= edges

    t = Graph(directed=True)
    for _ in range(g.num_vertices()):
        t.add_vertex()

    for u, v in tree_edges:
        t.add_edge(t.vertex(u), t.vertex(v))

    tree_nodes = {u for e in tree_edges for u in e}
    vfilt = t.new_vertex_property('bool')
    vfilt.a = False
    for v in tree_nodes:
        vfilt[t.vertex(v)] = True

    t.set_vertex_filter(vfilt)

    if return_closure:
        return t, gc, mst_tree
    else:
        return t
