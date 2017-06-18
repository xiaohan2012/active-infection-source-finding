import networkx as nx
import numpy as np
from graph_tool import Graph
from graph_tool.search import bfs_search, BFSVisitor


def get_leaves(t):
    return np.nonzero((t.degree_property_map(deg='in').a == 1)
                      & (t.degree_property_map(deg='out').a == 0))[0]


def get_roots(t):
    return np.nonzero((t.degree_property_map(deg='out').a > 0)
                      & (t.degree_property_map(deg='in').a == 0))[0]


def extract_edges(g):
    return [(int(u), int(v)) for u, v in g.edges()]


def gt2nx(g, root, terminals, node_attrs=None, edge_attrs=None):
    if g.is_directed():
        gx = nx.DiGraph()
    else:
        gx = nx.Graph()

    for v in set(terminals) | {root}:
        gx.add_node(v)
        if node_attrs is not None:
            for name, node_attr in node_attrs.items():
                gx.node[v][name] = node_attr[g.vertex(v)]
                
    for e in g.edges():
        u, v = int(e.source()), int(e.target())
        gx.add_edge(u, v)
        if edge_attrs is not None:
            for name, edge_attr in edge_attrs.items():
                gx[u][v][name] = edge_attr[e]
    return gx


def filter_nodes_by_edges(t, edges):
    vfilt = t.new_vertex_property('bool')
    vfilt.a = False
    nodes = {u for e in edges for u in e}
    for n in nodes:
        vfilt[n] = True
    t.set_vertex_filter(vfilt)
    return t


def edges2graph(g, edges):
    tree = Graph(directed=True)
    for _ in range(g.num_vertices()):
        tree.add_vertex()
    for u, v in edges:
        tree.add_edge(int(u), int(v))

    return filter_nodes_by_edges(tree, edges)


def bottom_up_traversal(t, vis=None, debug=False):
    leaves = get_leaves(t)
    s = list(leaves)
    visited = set()
    while len(s) > 0:
        v = s.pop(0)
        if vis:
            vis.examine_vertex(t.vertex(v))
        visited.add(v)
        if debug:
            print('visiting {}'.format(v))
        for e in t.vertex(v).in_edges():
            u = int(e.source())
            if vis:
                vis.tree_edge(e)
            if u not in visited:
                if debug:
                    print('pushing {}'.format(u))
                s.append(u)


def edges_to_directed_tree(g, root, edges):
    t = Graph(directed=False)
    for _ in range(g.num_vertices()):
        t.add_vertex()

    for u, v in edges:
        t.add_edge(u, v)

    class Visitor(BFSVisitor):
        def __init__(self):
            self.edges = set()
            
        def tree_edge(self, e):
            s, t = int(e.source()), int(e.target())
            self.edges.add((s, t))

    vis = Visitor()
    bfs_search(t, source=root, visitor=vis)

    t.clear_edges()
    t.set_directed(True)
    for u, v in vis.edges:
        t.add_edge(u, v)

    return filter_nodes_by_edges(t, edges)
