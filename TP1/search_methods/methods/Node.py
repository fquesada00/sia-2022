class Node:
    id = -1

    def __init__(self, state, parent, action, path_cost, depth, estimated_cost=0):
        Node.id = Node.id+1
        self.id = Node.id

        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.estimated_cost = estimated_cost
        self.depth = depth

    def __lt__(self, node):
        return self.estimated_cost < node.estimated_cost


    def __str__(self):
        if self.parent is not None:
            return f"State:{self.state}\tParent State:{self.parent.state}\tAction:{self.action}\tPath Cost:{self.path_cost}\tEstimated Cost:{self.estimated_cost}\n"
        return f"State:{self.state}\tParent State:{self.parent}\tAction:{self.action}\tPath Cost:{self.path_cost}\tEstimated Cost:{self.estimated_cost}\n"
