import networkx as nx


def grid_2d(n, p):
    """
    n: height and width
    p: infection proba
    """
    g = nx.grid_2d_graph(n, n)
        
    for i, j in g.edges_iter():
        g[i][j]['p'] = p
        g[i][j]['d'] = 1
    return g
