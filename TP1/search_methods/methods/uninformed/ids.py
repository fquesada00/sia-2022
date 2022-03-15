import time
from search_methods.methods.uninformed.dls import depth_limited_search
from search_methods.statistics.Statistics import Statistics
import networkx as nx

def iterative_depth_search(initial_state, n, initial_limit):
    limit = initial_limit
    result, tree, frontier_len,expanded_nodes = depth_limited_search(initial_state, n, limit)
    if result is None:
        while result is None:
            limit += 1
            result, tree, frontier_len,new_expanded_nodes = depth_limited_search(initial_state, n, limit)
            expanded_nodes += new_expanded_nodes

    else:
        lower_limit = 0
        upper_limit = result.depth

        while lower_limit <= upper_limit:
            mid = lower_limit + (upper_limit - lower_limit) // 2
            result, tree, frontier_len,new_expanded_nodes = depth_limited_search(
                initial_state, n, mid)
            expanded_nodes += new_expanded_nodes

            if result is None:
                lower_limit = mid + 1
            else:
                upper_limit = mid - 1

    return result, tree, frontier_len,expanded_nodes
