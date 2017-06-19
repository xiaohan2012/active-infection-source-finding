import networkx as nx
import numpy as np
from graph_tool import Graph, GraphView
from graph_tool.search import bfs_search, BFSVisitor
from graph_tool.topology import label_components


def get_leaves(t):
    """for directed graph
    """
    return np.nonzero((t.degree_property_map(deg='in').a == 1)
                      & (t.degree_property_map(deg='out').a == 0))[0]


def get_roots(t):
    """for undirected graph
    """
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


class EdgeCollectorVisitor(BFSVisitor):
    def __init__(self):
        self.edges = set()
        
    def tree_edge(self, e):
        s, t = int(e.source()), int(e.target())
        self.edges.add((s, t))


def edges_to_directed_tree(g, root, edges):
    t = Graph(directed=False)
    for _ in range(g.num_vertices()):
        t.add_vertex()

    for u, v in edges:
        t.add_edge(u, v)

    vis = EdgeCollectorVisitor()
    bfs_search(t, source=root, visitor=vis)

    t.clear_edges()
    t.set_directed(True)
    for u, v in vis.edges:
        t.add_edge(u, v)

    return filter_nodes_by_edges(t, edges)


def is_arborescence(tree):
    # is tree?
    l, _ = label_components(GraphView(tree, directed=False))
    if not np.all(np.array(l.a) == 0):
        print('not connected')
        print(np.array(l.a))
        return False

    in_degs = np.array([v.in_degree() for v in tree.vertices()])
    if in_degs.max() > 1:
        print('in_degree.max() > 1')
        return False
    if np.sum(in_degs == 1) != (tree.num_vertices() - 1):
        print('should be: only root has no parent')
        return False

    roots = get_roots(tree)
    assert len(roots) == 1, '>1 roots'
    
    return True


def is_tree(tree):
    # is tree?
    l, _ = label_components(GraphView(tree, directed=False))
    if not np.all(np.array(l.a) == 0):
        print('not connected')
        print(np.array(l.a))
        return False

    if tree.num_edges() != (tree.num_vertices() - 1):
        print('n. edges != n. nodes - 1')
        return False
    
    return True


def remove_redundant_edges_by_bfs(g, root):
    """for undirected grap, remove redundant edges unvisited by BFS"""
    vis = EdgeCollectorVisitor()
    bfs_search(g, source=root, visitor=vis)

    efilt = g.new_edge_property('bool')
    efilt.a = False

    for u, v in vis.edges:
        try:
            efilt[g.edge(u, v)] = True
        except ValueError:
            efilt[g.edge(v, u)] = True

    g.set_edge_filter(efilt)
    return g
