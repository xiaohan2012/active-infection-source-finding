import networkx as nx


dataset = 'dblp-collab'
with open('data/{}/network.txt'.format(dataset)) as f:
    g = nx.Graph()
    for l in f:
        u, v = map(int, l.split())
        g.add_edge(u, v)

    ccs = nx.connected_components(g)
    lcc = max(ccs, key=len)
    lcc_g = g.subgraph(lcc)
    nx.write_gpickle(lcc_g, 'data/{}/graph.pkl'.format(dataset))
