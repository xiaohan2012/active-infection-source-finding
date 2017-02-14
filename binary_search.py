import networkx as nx


def get_nodes_on_and_off_direction(g, start, end):
    pos_nodes = set()
    neg_nodes = set()
    for t, path in nx.shortest_path(g, source=start).items():
        if len(path) == 1:
            pos_nodes.add(t)
        elif path[1] == end:
            # print(t, path)
            pos_nodes.add(t)
        else:
            neg_nodes.add(t)
    return pos_nodes, neg_nodes
