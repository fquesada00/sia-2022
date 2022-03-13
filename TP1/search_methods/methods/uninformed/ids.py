from search_methods.methods.uninformed.dls import depth_limited_search


def iterative_depth_search(initial_state, n, initial_limit):
    limit = initial_limit
    result = depth_limited_search(initial_state, n, limit)

    if result is None:
        while result is None:
            limit += 1
            result = depth_limited_search(initial_state, n, limit)
    else:
        lower_limit = 0
        upper_limit = result.depth

        while lower_limit <= upper_limit:
            mid = lower_limit + (upper_limit - lower_limit) // 2
            result = depth_limited_search(initial_state, n, mid)

            if result is None:
                lower_limit = mid + 1
            else:
                upper_limit = mid - 1

    return result
