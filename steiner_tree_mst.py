import networkx as nx
import numpy as np
from graph_tool.all import GraphView, pbfs_search, BFSVisitor, Graph
from graph_tool.search import cpbfs_search
from utils import gt2nx


def extract_edges_from_pred(g, source, target, pred):
    edges = set()
    c = target
    while c != source:
        if pred[c] != -1:
            edges.add((pred[c], c))
            c = pred[c]
        else:
            break
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
        self.pred = pred
        self.dist = dist

    def black_target(self, e):
        s, t = e.source(), e.target()
        if self.pred[t] == -1:
            self.pred[t] = s
            self.dist[e.target()] = self.dist[s] + 1
    
    def tree_edge(self, e):
        self.pred[e.target()] = int(e.source())
        self.dist[e.target()] = self.dist[e.source()] + 1


def init_visitor(g, root):
    dist = g.new_vertex_property("int")
    dist.set_2d_array(np.ones(g.num_vertices()) * -1)
    dist[root] = 0
    pred = g.new_vertex_property("int64_t")
    pred.set_2d_array(np.ones(g.num_vertices()) * -1)
    vis = MyVisitor(pred, dist)
    return vis


def build_closure(g, cand_source, terminals, infection_times, k=-1, debug=False):
    """
    build a clojure graph in which cand_source + terminals are all connected to each other.
    the number of neighbors of each node is determined by k

    the larger the k, the denser the graph"""
    r2pred = {}
    edges = {}
    terminals = list(terminals)

    def get_edges(dist, root, terminals):
        return ((root, t, dist[t])
                for t in terminals
                if dist[t] != -1 and t != root)

    # from cand_source to terminals
    vis = init_visitor(g, cand_source)
    pbfs_search(g, source=cand_source, visitor=vis, terminals=terminals, count_threshold=k)
    r2pred[cand_source] = vis.pred
    for u, v, c in get_edges(vis.dist, cand_source, terminals):
        edges[(u, v)] = c
    if debug:
        print('cand_source: {}'.format(cand_source))
        print('#terminals: {}'.format(len(terminals)))
        print('edges from cand_source: {}'.format(edges))
    # from terminal to other terminals
    for root in terminals:
        vis = init_visitor(g, root)
        early_terminals = [t for t in terminals
                           if infection_times[t] > infection_times[root]]
        if debug:
            print('root: {}'.format(root))
            print('early_terminals: {}'.format(early_terminals))
        cpbfs_search(g, source=root,
                     visitor=vis,
                     terminals=early_terminals,
                     forbidden_nodes=list(set(terminals) - set(early_terminals)),
                     count_threshold=k)
        r2pred[root] = vis.pred
        for u, v, c in get_edges(vis.dist, root, early_terminals):
            if debug:
                print('edge ({}, {})'.format(u, v))
            edges[(u, v)] = c

    gc = Graph(directed=True)

    for _ in range(g.num_vertices()):
        gc.add_vertex()

    for (u, v) in edges:
        gc.add_edge(u, v)

    eweight = gc.new_edge_property('int')
    for e, c in edges.items():
        eweight[e] = c
    return gc, eweight, r2pred


def steiner_tree_mst(g, root, infection_times, source, terminals, return_closure=False, debug=False):
    gc, eweight, r2pred = build_closure(g, root, terminals, infection_times)
    
    # get the minimum spanning arborescence
    # graph_tool does not provide minimum_spanning_arborescence
    gx = gt2nx(gc, root, terminals, edge_attrs={'weight': eweight})
    try:
        nx_tree = nx.minimum_spanning_arborescence(gx, 'weight')
    except nx.exception.NetworkXException:
        if debug:
            print('fail to find mst')
        return None
    
    tree_efilt = gc.new_edge_property('bool')
    for u, v in nx_tree.edges():
        tree_efilt[gc.edge(gc.vertex(u), gc.vertex(v))] = True
    
    vfilt = gc.new_vertex_property('bool')
    vfilt[root] = 1
    for t in terminals:
        vfilt[t] = 1
    mst_tree = GraphView(gc, directed=True, efilt=tree_efilt, vfilt=vfilt)

    # extract the edges from the original graph
    edges = set()
    for u, v in mst_tree.edges():
        pred = r2pred[u]
        c = v
        while c != u and pred[c] != -1:
            edges.add((pred[c], c))
            c = pred[c]
    efilt = g.new_edge_property('bool')
    for e in edges:
        efilt[e] = True
    original_tree = GraphView(g, directed=True, efilt=efilt)
    if return_closure:
        return original_tree, gc, mst_tree
    else:
        return original_tree
