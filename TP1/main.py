from argparse import ArgumentParser
import json
import time
from search_methods.SearchMethod import SearchMethod 
from search_methods.heuristics.Heuristic import Heuristic
from search_methods.methods.search import search
from search_methods.constants import BLANK, N
from search_methods.visualization import plot_tree

def print_initial_data(n,initial_state,search_method,heuristic,initial_limit,max_depth):
  print(f"Searching solution for grid of size {n} and initial state: {initial_state}")
  print(f"Search method: {search_method}")

  if heuristic:
    print(f"Performing informed method with heuristic: {heuristic}")
  elif initial_limit:
    print(f"Initial Limit set as {initial_limit}")
  elif max_depth:
    print(f"Max depth set as {max_depth}")

def main():

  parser = ArgumentParser()
  parser.add_argument("-f", "--file", dest="config_file",
                      help="read configuration file",default='config.json')

  args = parser.parse_args()

  # open config file
  try:
    f = open(args.config_file)
  except:
    print("Config file not found!")
    exit()

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
    SearchMethod[search_method.upper()]
  except:
    print("Invalid search method")
    exit()

  # Load heuristic in case that exists
  if heuristic.lower() == 'manhattan':
    parsed_heuristic = Heuristic.MANHATTAN_DISTANCE
  elif heuristic.lower() == 'euclidean':
    parsed_heuristic = Heuristic.EUCLIDEAN_DISTANCE
  elif heuristic.lower() == 'misplaced_tiles':
    parsed_heuristic = Heuristic.MISPLACED_TILES

  # Print search parameters
  print_initial_data(N,initial_state,search_method,heuristic,initial_limit,max_depth)
  
  # Perform desired search
  start=time.perf_counter()
  result = search(initial_state,SearchMethod[search_method.upper()], N,parsed_heuristic)
  end = time.perf_counter()
  if result is not None:
    print(f"Solution Found!\n{str(result[0])}")
  print(f"Search completed in {end - start:0.4f} seconds")


if __name__ == '__main__':
  main()