import sys
import networkx as nx

dirs = ['er/2-10', 'pl-tree/2-10', 'barabasi/2-10',
        'kr-hier/10-10', 'kr-peri/10-10', 'kr-rand/10-10']
for d in dirs:
    # g = nx.read_gpickle(sys.argv[1]+'/graph.gpkl')
    g = nx.read_gpickle('data/' + d + '/graph.gpkl')
    print(d)
    print('|V|={}'.format(g.number_of_nodes()))
    print('|E|={}'.format(g.number_of_edges()))
    print('diameter={}'.format(nx.diameter(g)))
