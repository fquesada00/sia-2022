from collections import deque
from search_methods.methods.Node import Node
from search_methods.utils import get_actions, get_next_state, is_goal_state
import networkx as nx


def local_heuristic_search(initial_state, n, heuristic):
    root = Node(initial_state, None, None, 0, 0)
    frontier = deque([root])

    visited_states = {}

    tree = nx.Graph()
    frontier_len = len(frontier)
    tree.add_node(root.id, label=str(tuple(root.state)), level=root.depth)

    while frontier_len > 0:
        node = frontier.popleft()
        visited_states[tuple(node.state)] = True
        frontier_len -= 1

        if is_goal_state(node.state):
            expanded_nodes = tree.number_of_nodes() - len([node for node in tree if nx.degree(tree, node) == 1])
            return node, tree, frontier_len, expanded_nodes
        else:
            actions = get_actions(node.state, n)
            successors = []

            for action in actions:
                new_state = get_next_state(node.state, action, n)
                new_node = Node(new_state, node, action,
                                node.path_cost + 1, node.depth+1, heuristic(new_state, n))

                tree.add_node(new_node.id, label="Estimated cost: " + str(new_node.estimated_cost) + "\n" + str(
                    tuple(new_node.state)), level=new_node.depth)
                tree.add_edge(new_node.id, new_node.parent.id,
                              label=str(new_node.action))

                if not visited_states.get(tuple(new_state)):
                    visited_states[tuple(initial_state)] = True
                    successors.append(new_node)
                    frontier_len += 1

            successors.sort(key=lambda n: n.estimated_cost, reverse=True)
            frontier.extendleft(successors)
