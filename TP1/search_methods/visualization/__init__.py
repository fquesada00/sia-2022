from pyvis.network import Network
import networkx as nx

ROOT_COLOR = '#F49097'
GOAL_COLOR = '#FFF388'
PATH_COLOR = '#55D6C2'
NORMAL_COLOR = '#F2F5FF'


def color_solution_path(tree, goal_node):
    nx.set_node_attributes(tree, NORMAL_COLOR, 'color')

    tree.nodes[goal_node.id]['color'] = GOAL_COLOR
    tree.nodes[goal_node.id]['shape'] = 'star'

    current = goal_node

    while current.parent is not None:
        tree.nodes[current.parent.id]['color'] = PATH_COLOR
        current = current.parent

    tree.nodes[current.id]['color'] = ROOT_COLOR

    tree.nodes.data('color', NORMAL_COLOR)


def plot_tree(tree, goal_node):
    color_solution_path(tree, goal_node)

    g = Network(height='100%', width='100%',
                notebook=True, layout='hierarchical')
    g.from_nx(tree)

    g.show('tree.html')
