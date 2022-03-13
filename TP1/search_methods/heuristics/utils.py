def sum_over_tiles_heuristic(state, calculate, n):
    sum = 0
    for tile_index, tile in enumerate(state):
        sum += calculate(tile_index, tile, n)
    return sum