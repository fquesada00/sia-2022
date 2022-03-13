from collections import deque
from search_methods.SearchMethod import SearchMethod
from search_methods.methods.Node import Node
from search_methods.utils import get_actions, get_next_state, is_goal_state
from pyvis.network import Network


def uninformed_search(initial_state, n, search_method):
    root = Node(initial_state, None, None, 0, 0)
    frontier = deque([root])

    if search_method == SearchMethod.BFS:
        #queue
        add_to_frontier = frontier.append
    else:
        #stack
        add_to_frontier = frontier.appendleft

    visited_states = {}
    visited_states[tuple(initial_state)] = True
    visited_node_count = 1

    tree_vis = Network(notebook=True, layout='hierarchical')
    frontier_len = len(frontier)
    tree_vis.add_node(root.id, label=str(tuple(root.state)),
                      level=root.depth, color='#dd4b39')
    while frontier_len > 0:
        node = frontier.popleft()
        print(node)
        visited_states[tuple(node.state)] = True
        visited_node_count += 1
        frontier_len -= 1
        if is_goal_state(node.state):
            tree_vis.prep_notebook()
            tree_vis.show('example.html')
            return node
        else:
            actions = get_actions(node.state, n)
            for action in actions:
                new_state = get_next_state(node.state, action, n)
                new_node = Node(new_state, node, action,
                                node.path_cost + 1, node.depth+1)
                tree_vis.add_node(new_node.id, label=str(
                    tuple(new_node.state)), level=new_node.depth)
                tree_vis.add_edge(new_node.id, new_node.parent.id,
                                  label=str(new_node.action))
                if not visited_states.get(tuple(new_state)):
                    add_to_frontier(new_node)
                    frontier_len += 1