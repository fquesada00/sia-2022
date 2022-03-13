from search_methods.SearchMethod import SearchMethod
from search_methods.methods.informed.local_heuristic import local_heuristic_search
from search_methods.methods.informed.global_heuristic import global_frontier_heuristic_search
from search_methods.methods.uninformed.dls import depth_limited_search
from search_methods.methods.uninformed.ids import iterative_depth_search
from search_methods.methods.uninformed import uninformed_search
from search_methods.utils import is_solvable


def get_f(heuristic, w, n):
    return lambda node: (1-w)*node.path_cost + w*heuristic(node.state, n)


def search(initial_state, search_method, n, heuristic=None, max_depth=-1, initial_limit=-1):
    if not is_solvable(initial_state, n):
        print('problem not solvable')
        return None

    if search_method in [SearchMethod.GHS, SearchMethod.A_STAR]:
        if search_method == SearchMethod.GHS:
            w = 1
        else:
            w = 0.5

        return global_frontier_heuristic_search(initial_state, n, get_f(heuristic, w, n))
    elif search_method == SearchMethod.LHS:
        return local_heuristic_search(initial_state, n, heuristic)
    elif search_method == SearchMethod.IDS:
        return iterative_depth_search(initial_state, n, initial_limit)
    elif search_method == SearchMethod.DLS:
        return depth_limited_search(initial_state, n, max_depth)
    elif search_method in [SearchMethod.BFS, SearchMethod.DFS]:
        return uninformed_search(initial_state, n, search_method)
