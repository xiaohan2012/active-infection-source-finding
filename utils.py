import numpy as np
import networkx as nx
from graph_tool import GraphView, Graph
from graph_tool.search import pbfs_search
from collections import Counter
from graph_tool.all import BFSVisitor, shortest_distance, shortest_path

MAXINT = np.iinfo(np.int32).max


def get_infection_time(g, source):
    time = shortest_distance(g, source=source).a
    time[time == MAXINT] = -1
    return time


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
def weighted_sample_with_replacement(pool, weights, N):
    assert len(pool) == len(weights)
    np.testing.assert_almost_equal(np.sum(weights), 1)
    cs = np.tile(np.cumsum(weights), (N, 1))
    rs = np.tile(np.random.rand(N)[:, None], (1, len(weights)))
    indices = np.sum(cs < rs, axis=1)
    return list(map(pool.__getitem__, indices))


def test_weighted_sample_with_replacement():
    pool = [1, 2, 3]
    ps = [0.2, 0.3, 0.5]
    samples = weighted_sample_with_replacement(pool, ps, 10000)
    cnt = Counter(samples)
    total = sum(cnt.values())
    cnt[1] /= total
    cnt[2] /= total
    cnt[3] /= total
    np.testing.assert_almost_equal(sorted(cnt.values()), ps, decimal=2)


def test_generalized_jaccard_similarity():
    a = [1, 1, 2]
    b = [1, 2, 3]
    assert generalized_jaccard_similarity(a, b) == 0.5
    assert generalized_jaccard_similarity(a, a) == 1.0


def infeciton_time2weight(ts):
    """invert the infection times so that earlier infected nodes have larger weight"""
    ts = np.array(ts)  # copy it
    max_val = np.max(ts)
    ts[ts == -1] = max_val + 1
    return np.array(
        [(max_val - t + 1)
         for n, t in enumerate(ts)])


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def sp_len_2d(g, dtype=np.float64):
    n = g.number_of_nodes()
    d = np.zeros((n, n), dtype=dtype)
    sp_len = nx.shortest_path_length(g)
    for i in np.arange(n):
        d[i, :] = [sp_len[i][j] for j in np.arange(n)]
    return d


def get_rank_index(array, id_):
    """if value of array[id_] is not unqiue, take the middle
    the larger the better
    """
    val = array[id_]
    sorted_array = np.sort(array)[::-1]
    idx = np.nonzero(sorted_array == val)[0]
    return idx[0] - 1 + np.ceil(len(idx) / 2)


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


def earliest_obs_node(obs_nodes, infection_times):
    return min(obs_nodes, key=infection_times.__getitem__)


def build_minimum_tree(g, root, terminals, edges, directed=True):
    """remove redundant edges from `edges` so that root can reach each node in terminals
    """
    # build the tree
    t = Graph(directed=directed)

    for _ in range(g.num_vertices()):
        t.add_vertex()

    for (u, v) in edges:
        t.add_edge(u, v)

    # mask out redundant edges
    vis = init_visitor(t, root)
    pbfs_search(t, source=root, terminals=list(terminals), visitor=vis)

    minimum_edges = {e
                     for u in terminals
                     for e in extract_edges_from_pred(t, root, u, vis.pred)}
    # print(minimum_edges)
    efilt = t.new_edge_property('bool')
    efilt.a = False
    for u, v in minimum_edges:
        efilt[u, v] = True
    t.set_edge_filter(efilt)

    vfilt = t.new_vertex_property('bool')
    vfilt.a = False
    tree_nodes = {u for e in minimum_edges for u in e}
    for v in tree_nodes:
        vfilt[v] = True
    t.set_vertex_filter(vfilt)
    return t

# @profile
def to_directed(g, t, root):
    new_t = Graph(directed=True)
    all_edges = set()
    leaves = [v for v in t.vertices()
              if (v.out_degree() + v.in_degree()) == 1 and t != root]
    for target in leaves:
        path = shortest_path(t, source=root, target=target)[0]
        edges = set(zip(path[:-1], path[1:]))
        all_edges |= edges

    for _ in range(g.num_vertices()):
        new_t.add_vertex()
    for u, v in all_edges:
        new_t.add_edge(int(u), int(v))
    return new_t


def get_leaves(t):
    # print([(int(v), v.out_degree(), v.in_degree()) for v in t.vertices()])
    return np.nonzero((t.degree_property_map(deg='in').a == 1)
                      & (t.degree_property_map(deg='out').a == 0))[0]


def get_paths(t, source, terminals):
    return [list(map(int, shortest_path(t, source=source, target=t.vertex(int(n)))[0]))
            for n in terminals]


