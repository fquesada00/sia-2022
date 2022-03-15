import networkx as nx

from search_methods.statistics.Transition import Transition


class Statistics:
    def __init__(self, tree, goal_node,frontier_len, execution_time):
        self.solution_depth = goal_node.depth
        self.solution_path_cost = goal_node.path_cost
        self.frontier_count = frontier_len
        self.expanded_count = tree.number_of_nodes() - self.__count_leaf_nodes(tree) + 1
        self.solution_path = self.__get_solution_path(goal_node)
        self.execution_time = execution_time

    def __count_leaf_nodes(self, tree):
        count = 0

        for node in tree:
            if nx.degree(tree, node) == 1:
                count += 1

        return count

    def __get_solution_path(self, goal_node):
        path = []
        current = goal_node
        while current.parent is not None:
            path.insert(0,
                        Transition(current.action, current.state))
            current = current.parent

        return path

    def __str__(self):
        return f'Solution path cost: {self.solution_path_cost} \n' \
               f'Solution depth: {self.solution_depth} \n' \
               f'Frontier node count: {self.frontier_count} \n' \
               f'Expanded node count: {self.expanded_count} \n' \
               f'Execution time: {self.execution_time} \n' \
            f'Solution path: {[f"{step}" for step in self.solution_path]}'
