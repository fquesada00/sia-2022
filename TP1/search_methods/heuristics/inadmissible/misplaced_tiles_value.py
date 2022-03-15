from search_methods.constants import BLANK
from search_methods.heuristics.utils import sum_over_tiles_heuristic
        
def misplaced_tile_value(tile_index, tile, n): 
    if tile_index == tile - 1 or tile == BLANK:
        return 0
    
    return tile

def misplaced_tiles_value_heuristic(state, n):
    return sum_over_tiles_heuristic(state, misplaced_tile_value, n)