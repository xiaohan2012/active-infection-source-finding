"""minimum spanning tree on the forrest

Before doing mst, we connect adjacent nodes into small trees, then we span these trees (as supernodes).
"""
import numpy as np
import networkx as nx
from utils import gt2nx

from itertools import combinations
from copy import copy
from graph_tool.search import cpbfs_search
from graph_tool.all import Graph

from steiner_tree_mst import init_visitor
from utils import edges2graph


def connect_adjacent_infections(g, obs_nodes, infection_times):
    sorted_nodes = list(sorted(obs_nodes, key=infection_times.__getitem__))
    obs_set = set(obs_nodes)
    visited = np.zeros(g.num_vertices(), dtype=bool)
    regions = []
    while len(sorted_nodes) > 0:
        r = sorted_nodes.pop(0)
        region = {'nodes': [r], 'edges': []}
        queue = [r]
        while len(queue) > 0:
            v = queue.pop(0)
            visited[v] = True
            for u in g.vertex(v).all_neighbours():
                u = int(u)
                if not visited[u]:
                    if u in obs_set and infection_times[v] < infection_times[u]:
                        region['nodes'].append(u)
                        region['edges'].append((v, u))
                        visited[u] = True
                        queue.append(u)
                        sorted_nodes.remove(u)
        regions.append(region)
    for r in regions:
        r['head'] = min(r['nodes'], key=infection_times.__getitem__)
        r['head_time'] = infection_times[r['head']]
    return {i: r for i, r in enumerate(regions)}


def build_region_closure(g, root, regions, infection_times, obs_nodes, debug=False):
    """return a closure graph on the the components"""
    regions = copy(regions)
    root_region = {'nodes': {root}, 'head': root, 'head_time': -float('inf')}
    regions[len(regions)] = root_region

    gc = Graph(directed=True)
    for _ in range(len(regions)):
        gc.add_vertex()

    # connect each region
    gc_edges = []
    original_edge_info = {}
    for i, j in combinations(regions, 2):
        # make group i the one with *later* head
        if regions[i]['head_time'] < regions[j]['head_time']:
            i, j = j, i
        
        if debug:
            print('i, j={}, {}'.format(i, j))
        # only need to connect head i to one of the nodes in group j
        # where nodes in j have time stamp < head i
        # then an edge from region j to region i (because j is earlier)

        head_i = regions[i]['head']
        
        def get_pseudo_time(n):
            if n == root:
                return - float('inf')
            else:
                return infection_times[n]

        targets = [n for n in regions[j]['nodes'] if get_pseudo_time(n) < regions[i]['head_time']]

        if debug:
            print('head_i: {}'.format(head_i))
            print('targets: {}'.format(targets))
            print('regions[j]["nodes"]: {}'.format(regions[j]['nodes']))
 
        if len(targets) == 0:
            continue
            
        visitor = init_visitor(g, head_i)
        forbidden_nodes = list(set(regions[i]['nodes']) | (set(regions[j]['nodes']) - set(targets)))

        if debug:
            print('forbidden_nodes: {}'.format(forbidden_nodes))
            
        # NOTE: count_threshold = 1
        cpbfs_search(g, source=head_i,
                     terminals=targets,
                     forbidden_nodes=forbidden_nodes,
                     visitor=visitor,
                     count_threshold=1)
    
        reachable_targets = [t for t in targets if visitor.dist[t] > 0]

        if debug:
            print('reachable_targets: {}'.format(reachable_targets))
            
        if len(reachable_targets) == 0:
            # cannot reach there
            continue

        source = min(reachable_targets, key=visitor.dist.__getitem__)
        dist = visitor.dist[source]

        assert dist > 0

        gc_edges.append(((j, i, dist)))
        original_edge_info[(j, i)] = {
            'dist': dist,
            'pred': visitor.pred,
            'original_edge': (source, head_i)
        }
    for u, v, _ in gc_edges:
        gc.add_edge(u, v)

    eweight = gc.new_edge_property('int')
    for u, v, c in gc_edges:
        eweight[gc.edge(gc.vertex(u), gc.vertex(v))] = c

    return gc, eweight, original_edge_info


def steiner_tree_region_mst(g, root, infection_times, source, terminals, return_closure=False, debug=False):
    regions = connect_adjacent_infections(g, terminals, infection_times)

    gc, eweight, orginal_edge_info = build_region_closure(
        g, root, regions,
        infection_times, terminals)
    
    root = gc.num_vertices() - 1  # last node is root
    gx = gt2nx(gc, root, list(map(int, gc.vertices())), edge_attrs={'weight': eweight})
    try:
        nx_tree = nx.minimum_spanning_arborescence(gx, 'weight')
    except nx.NetworkXException:
        # cannot find any MST
        return None

    # now we reconstruct the super node tree to the original tree
    orig_edges = []
    for i, j in nx_tree.edges():
        einfo = orginal_edge_info[(i, j)]
        u, v = einfo['original_edge']
        pred = einfo['pred']
        c = u
        while c != v and pred[c] != -1:
            orig_edges.append((c, pred[c]))
            c = pred[c]

    efilt = g.new_edge_property('bool')
    efilt.a = False
    
    all_edges = [e for r in regions.values() for e in r['edges']]
    all_edges += orig_edges

    steiner_tree = edges2graph(g, all_edges)
    ret = steiner_tree
    if return_closure:
        region_graph = edges2graph(g, [e for r in regions.values() for e in r['edges']])
        ret = (ret, gc, region_graph)
    return ret
