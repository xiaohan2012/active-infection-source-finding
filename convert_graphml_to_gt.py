import sys
from graph_tool.all import load_graph

path = sys.argv[1]

g = load_graph('data/{}/graph.graphml'.format(path))
g.save('data/{}/graph.gt'.format(path))
