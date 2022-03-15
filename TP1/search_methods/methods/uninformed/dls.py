from collections import deque
from search_methods.methods.Node import Node
from search_methods.utils import get_actions, get_next_state, is_goal_state
import networkx as nx


def depth_limited_search(initial_state, n, max_depth):
    root = Node(initial_state, None, None, 0, 0)
    frontier = deque([root])

    visited_states = {}
    visited_states[tuple(initial_state)] = 0

    tree = nx.Graph()
    tree.add_node(root.id, label=str(tuple(root.state)), level=root.depth)

    frontier_len = len(frontier)

    while frontier_len > 0:
        node = frontier.popleft()
        frontier_len -= 1

        if node.depth > max_depth:
            continue
        elif is_goal_state(node.state):
            expanded_nodes = tree.number_of_nodes() - len([node for node in tree if nx.degree(tree, node) == 1])
            return node, tree, frontier_len, expanded_nodes

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

                if visited_states.get(tuple(new_state)) is None or visited_states.get(tuple(new_state)) > new_node.depth:
                    visited_states[tuple(node.state)] = node.depth
                    frontier.appendleft(new_node)
                    frontier_len += 1
    expanded_nodes = tree.number_of_nodes() - len([node for node in tree if nx.degree(tree, node) == 1])
    return None, tree, frontier_len,expanded_nodes
