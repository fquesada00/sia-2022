from argparse import ArgumentError
import json
from search_methods.SearchMethod import SearchMethod 
from search_methods.heuristics.Heuristic import Heuristic
from search_methods.methods.search import search
from search_methods.constants import BLANK, N

def print_initial_data(n,initial_state,heuristic,initial_limit,max_depth):
  print(f"Searching solution for grid of size {N} and initial state: {initial_state}")
  print(f"Search method: {search_method}")
  if heuristic:
    print(f"Performing informed method with heuristic: {heuristic}")
  elif initial_limit:
    print(f"Initial Limit set as {initial_limit}")
  elif max_depth:
    print(f"Max depth set as {max_depth}")


if __name__ == '__main__':
  # open config file
  f = open("config.json")

  # Read config json
  config = json.load(f)

  # Load initial parameters
  N = config['grid_size']
  initial_state = config['initial_state']
  search_method = config['search_method'] 
  heuristic = config['heuristic']
  initial_limit = config['iterative_depth_search_initial_limit']
  max_depth = config['depth_limited_search_max_depth']
  parsed_heuristic = None
  
  try:
    SearchMethod[search_method]
  except:
    print("Invalid search method")
    exit()

  # Load heuristic in case that exists
  if heuristic == 'manhattan':
    parsed_heuristic = Heuristic.MANHATTAN_DISTANCE
  elif heuristic == 'euclidean':
    parsed_heuristic = Heuristic.EUCLIDEAN_DISTANCE
  elif heuristic == 'misplaced_tiles':
    parsed_heuristic = Heuristic.MISPLACED_TILES

  # Print search parameters
  print_initial_data(N,initial_state,heuristic,initial_limit,max_depth)
  
  # Perform desired search
  print(search(initial_state,SearchMethod[search_method], N,parsed_heuristic))
