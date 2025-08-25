class State:

    def __init__(self, q, qdot):
        self.q = q
        self.qdot = qdot

    def __repr__(self):
        return f'[State object at {hex(id(self))}]'