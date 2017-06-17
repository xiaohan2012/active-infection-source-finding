import numpy as np
from graph_tool.search import bfs_search, BFSVisitor
from graph_tool.topology import shortest_distance
from gt_utils import bottom_up_traversal


class TopDownVisitor(BFSVisitor):
    def __init__(self, pred, root, obs_nodes):
        pred[root] = root
        self.obs = set(obs_nodes)
        self.pred = pred
        
    def tree_edge(self, e):
        s, t = int(e.source()), int(e.target())
        if t in self.obs:
            self.pred[t] = t
        else:
            self.pred[t] = self.pred[s]
    

class BottomUpVisitor():
    def __init__(self, pred, root, obs_nodes):
        pred[root] = root
        self.obs = set(obs_nodes)
        self.pred = pred

    def examine_vertex(self, v):
        v = int(v)
        if v in self.obs:
            self.pred[v] = v

    def tree_edge(self, e):
        """edge from t to s"""
        s, t = int(e.source()), int(e.target())
        if s in self.obs:
            self.pred[s] = s
        else:
            self.pred[s] = self.pred[t]


def fill_missing_time(g, t, root, obs_nodes, infection_times, debug=False):
    # get ancestor and descendent
    td_vis = TopDownVisitor(np.ones(g.num_vertices(), dtype=np.int) * -1, root, obs_nodes)
    bfs_search(t, source=root, visitor=td_vis)

    bu_vis = BottomUpVisitor(np.ones(g.num_vertices(), dtype=np.int) * -1, root, obs_nodes)
    bottom_up_traversal(t, vis=bu_vis)

    # infer the time
    hidden_nodes = set(map(int, t.vertices())) - set(obs_nodes)
    pred_infection_times = np.array(infection_times)
    dist = shortest_distance(t, source=root)
    for v in hidden_nodes:
        ans, des = td_vis.pred[v], bu_vis.pred[v]
        assert ans != -1
        assert des != -1

        if debug:
            print(v, ans, des)
            
        denum = dist[des] - dist[ans]
        numer = dist[v] - dist[ans]
        pred_infection_times[v] = (infection_times[ans] +
                                   numer / denum * (infection_times[des] - infection_times[ans]))
        
        if debug:
            assert pred_infection_times[v] > infection_times[ans]
            assert pred_infection_times[v] < infection_times[des]

            print('t(ans), t(des): {}, {}'.format(infection_times[ans], infection_times[des]))
            print('numer {}'.format(numer))
            print('denum {}'.format(denum))
            print('pred time {}'.format(pred_infection_times[v]))

    return pred_infection_times
