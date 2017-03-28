import sys
import networkx as nx

g = nx.read_gpickle(sys.argv[1])
print('|V|={}'.format(g.number_of_nodes()))
print('|E|={}'.format(g.number_of_edges()))
