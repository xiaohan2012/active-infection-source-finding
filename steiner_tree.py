"""
find minimum steiner tree (undirected)
"""
import numpy as np
from graph_tool import Graph, GraphView
from graph_tool.search import pbfs_search
from steiner_tree_mst import get_edges
from utils import init_visitor, extract_edges_from_pred
from graph_tool.topology import min_spanning_tree
from gt_utils import edges_to_directed_tree, build_minimum_tree, extract_edges


# @profile
def build_closure(g, terminals,
                  debug=False,
                  verbose=False):
    terminals = list(terminals)
    # build closure
    gc = Graph(directed=False)

    for _ in range(g.num_vertices()):
        gc.add_vertex()

    edges_with_weight = set()
    r2pred = {}

    for r in terminals:
        if debug:
            print('root {}'.format(r))
        vis = init_visitor(g, r)
        pbfs_search(g, source=r, terminals=terminals, visitor=vis)
        new_edges = set(get_edges(vis.dist, r, terminals))
        if debug:
            print('new edges {}'.format(new_edges))
        edges_with_weight |= new_edges
        r2pred[r] = vis.pred
    
    for u, v, c in edges_with_weight:
        gc.add_edge(u, v)
        
    eweight = gc.new_edge_property('int')
    weights = np.array([c for _, _, c in edges_with_weight])
    eweight.set_2d_array(weights)

    vfilt = gc.new_vertex_property('bool')
    vfilt.a = False
    for v in terminals:
        vfilt[v] = True
    gc.set_vertex_filter(vfilt)
    return gc, eweight, r2pred


# @profile
def get_steiner_tree(g, root, obs_nodes, debug=False, verbose=False):
    gc, eweight, r2pred = build_closure(g, obs_nodes,
                                        debug=debug, verbose=verbose)

    tree_map = min_spanning_tree(gc, eweight, root=None)
    tree = GraphView(gc, directed=False, efilt=tree_map)

    tree_edges = set()
    for e in tree.edges():
        u, v = map(int, e)
        for i, j in extract_edges_from_pred(g, u, v, r2pred[u]):
            tree_edges.add((j, i))

    # a bit involved...
    und_tree = edges_to_directed_tree(g, root, tree_edges)
    return build_minimum_tree(g, root, obs_nodes, extract_edges(und_tree))

    # return build_minimum_tree(g, root, obs_nodes, tree_edges, directed=True)
    # t = Graph(directed=False)

    # for _ in range(g.num_vertices()):
    #     t.add_vertex()
    # for u, v in tree_edges:
    #     t.add_edge(u, v)

    # t = filter_nodes_by_edges(t, tree_edges)
    # return remove_redundant_edges_by_bfs(t, root)
