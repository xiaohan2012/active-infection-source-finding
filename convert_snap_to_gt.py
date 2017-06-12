from graph_tool import Graph
from tqdm import tqdm

dataset = 'slashdot'
n_nodes = 77360
n_edges = 905468


g = Graph(directed=True)
for _ in tqdm(range(n_nodes), total=n_nodes):
    g.add_vertex()

with open('data/{}/network.txt'.format(dataset)) as f:
    for l in tqdm(f, total=n_edges):
        u, v = l.split()
        if not g.edge(u, v):
            g.add_edge(u, v)

g.save('data/{}/graph.gt'.format(dataset))
