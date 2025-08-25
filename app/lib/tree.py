import networkx as nx

class Tree:

    def __init__(self):
        self.tree = nx.DiGraph()
        self.node_idx = 0
        self.root_node = None
        self.last_node = None

    def add_node(self, state):
        self.tree.add_node(state, idx=self.node_idx)
        self.node_idx += 1
        if self.node_idx == 1:
            self.root_node = state
        self.last_node = state

    def add_edge(self, state_src, state_tar, action, states_between):
        assert self.tree.has_node(state_tar)
        assert self.tree.has_node(state_tar)
        self.tree.add_edge(state_src, state_tar, action=action, states_between=states_between)

    def get_in_edge(self, state):
        assert self.tree.has_node(state)
        in_edges = list(self.tree.in_edges(state, data=True))
        assert len(in_edges) <= 1
        if len(in_edges) == 0:
            return None
        else:
            return in_edges[0]

    def get_out_edges(self, state):
        assert self.tree.has_node(state)
        return list(self.tree.out_edges(state, data=True))

    def has_path(self, start_state, end_state):
        state = end_state
        while True:
            if state == start_state:
                return True
            in_edge = self.get_in_edge(state)
            if in_edge is None:
                return False
            else:
                state = in_edge[0]

    def get_path(self, start_state, end_state):
        path = []
        state = end_state
        while True:
            path.append(state)
            if state == start_state:
                break
            in_edge = self.get_in_edge(state)
            if in_edge is None:
                return None # failed
            else:
                state_pred, states_betweeen = in_edge[0], in_edge[2]['states_between']
                path.extend(states_betweeen[::-1])
                state = state_pred
        return path[::-1]

    def get_root_path(self, state):
        return self.get_path(self.root_node, state)

    def get_nodes(self):
        return list(self.tree.nodes)

    def get_edges(self):
        return list(self.tree.edges(data=True))

    def get_predecessor(self, state):
        preds = list(self.tree._pred[state].keys())
        assert len(preds) <= 1
        if len(preds) == 0:
            return None
        else:
            return preds[0]

    def get_successors(self, state):
        return list(self.tree._succ[state].keys())

    def get_in_degree(self, state):
        return self.tree.in_degree(state)

    def get_out_degree(self, state):
        return self.tree.out_degree(state)

    def get_node_attr(self, node, attr_name):
        return self.tree.nodes[node][attr_name]

    def set_node_attr(self, node, attr_name, attr_val):
        self.tree.nodes[node][attr_name] = attr_val

    def draw(self):
        from networkx.drawing.nx_pydot import graphviz_layout
        import matplotlib.pyplot as plt
        pos = graphviz_layout(self.tree, prog='dot')
        nx.draw_networkx(self.tree, pos, arrows=True, with_labels=False, node_size=10)
        plt.show()