from collections import deque

from search_methods.SearchMethod import SearchMethod
from search_methods.methods.Node import Node
from search_methods.utils import get_actions, get_next_state, is_goal_state
import networkx as nx


def uninformed_search(initial_state, n, search_method):
    root = Node(initial_state, None, None, 0, 0)
    frontier = deque([root])

    if search_method == SearchMethod.BFS:
        # queue
        add_to_frontier = frontier.append
    else:
        # stack
        add_to_frontier = frontier.appendleft

    visited_states = {}
    visited_states[tuple(initial_state)] = True

    tree = nx.Graph()
    tree.add_node(root.id, label=str(tuple(root.state)), level=root.depth)

    frontier_len = len(frontier)

    while frontier_len > 0:
        node = frontier.popleft()
        frontier_len -= 1

        if is_goal_state(node.state):
            return node, tree, frontier_len
        else:
            actions = get_actions(node.state, n)

            for action in actions:
                new_state = get_next_state(node.state, action, n)
                new_node = Node(new_state, node, action,
                                node.path_cost + 1, node.depth+1)

                tree.add_node(new_node.id, label=str(
                    tuple(new_node.state)), level=new_node.depth)
                tree.add_edge(new_node.id, new_node.parent.id,
                              label=str(new_node.action))

                if not visited_states.get(tuple(new_state),False):
                    visited_states[tuple(new_state)] = True
                    add_to_frontier(new_node)
                    frontier_len += 1
