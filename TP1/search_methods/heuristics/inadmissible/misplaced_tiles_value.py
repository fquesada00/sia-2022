from search_methods.constants import BLANK
from search_methods.heuristics.utils import sum_over_tiles_heuristic
        
def misplaced_tile_value(tile_index, tile, n): 
    if tile_index == tile - 1 or tile == BLANK:
        return 0
    
    return tile

def misplaced_tiles_value_heuristic(state, n):
    return sum_over_tiles_heuristic(state, misplaced_tile_value, n)

# def init():
#     state = [8, 6, 7, 2, 5, 4, 3, BLANK, 1]
#     # state = [2,1,3,4,5,6,7,0,8]

#     print(misplaced_tiles_value_heuristic(state, 3))