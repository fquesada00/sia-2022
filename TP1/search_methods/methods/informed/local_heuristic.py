from collections import deque
from search_methods.methods.Node import Node
from search_methods.utils import get_actions, get_next_state, is_goal_state
from pyvis.network import Network


def local_heuristic_search(initial_state, n, heuristic):
    root = Node(initial_state, None, None, 0, 0)
    frontier = deque([root])
    visited_node_count = 1

    visited_states = {}
    visited_states[tuple(initial_state)] = True

    tree_vis = Network(notebook=True, layout='hierarchical')
    frontier_len = len(frontier)
    tree_vis.add_node(root.id, label=str(tuple(root.state)),
                      level=root.depth, color='#dd4b39')

    while frontier_len > 0:
        node = frontier.popleft()
        visited_states[tuple(node.state)] = True
        visited_node_count += 1
        frontier_len -= 1

        if is_goal_state(node.state):
            tree_vis.prep_notebook()
            tree_vis.show('example.html')
            return node
        else:
            actions = get_actions(node.state, n)
            successors = []

            for action in actions:
                new_state = get_next_state(node.state, action, n)
                new_node = Node(new_state, node, action,
                                node.path_cost + 1, node.depth+1, heuristic(new_state, n))
                tree_vis.add_node(new_node.id, label="Estimated cost: " +str(new_node.estimated_cost) + "\n" + str(
                    tuple(new_node.state)), level=new_node.depth)
                tree_vis.add_edge(new_node.id, new_node.parent.id,
                                  label=str(new_node.action))
                if not visited_states.get(tuple(new_state)):
                    # insort(successors, new_node,*[n.estimated_cost for n in successors])
                    # insort(successors, new_node, *["estimated_cost"])
                    successors.append(new_node)
                    frontier_len += 1
            successors.sort(key=lambda n : n.estimated_cost, reverse=True)
            frontier.extendleft(successors)