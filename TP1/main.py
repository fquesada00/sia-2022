from search_methods.SearchMethod import SearchMethod 
from search_methods.heuristics.Heuristic import Heuristic
from search_methods.methods.search import search
from search_methods.constants import BLANK, N

initial_state = [4, 6, 1, 3, 2, BLANK, 5, 7, 8]

if __name__ == '__main__':
  print(search(initial_state, SearchMethod.GHS, N, Heuristic.MANHATTAN_DISTANCE,max_depth=17))
