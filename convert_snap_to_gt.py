from graph_tool import Graph
from tqdm import tqdm

dataset = 'pokec'
n_nodes = 1632803
n_edges = 30622564
g = Graph(directed=True)
for _ in tqdm(range(n_nodes), total=n_nodes):
    g.add_vertex()

with open('data/{}/network.txt'.format(dataset)) as f:
    for l in tqdm(f, total=n_edges):
        u, v = l.split()
        g.add_edge(u, v)

g.save('data/{}/graph.gt'.format(dataset))
