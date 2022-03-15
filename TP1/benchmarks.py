from argparse import ArgumentParser
import json
from random import shuffle
import statistics
import time
from search_methods.SearchMethod import SearchMethod
from search_methods.heuristics.Heuristic import Heuristic
from search_methods.methods.search import search

from search_methods.statistics.Statistics import Statistics

best_initial_state = [0, 1, 2, 4, 5, 3, 7, 8, 6]
medium_initial_state = [2, 7, 1, 3, 0, 8, 6, 5, 4]
worst_initial_state = [8, 6, 7, 2, 5, 4, 3, 0, 1]
N = 3


def benchmark(state, search_method, heuristic, heuristic_name, initial_limit=10, max_depth=10, repeats=1):
    print(f"Benchmarking {search_method}{f' with initial limit {initial_limit}' if initial_limit > 0 else ''}{f' with heuristic {heuristic_name}' if heuristic_name != '' else ''} {f'with max depth {max_depth}' if max_depth > 0 else ''}")
    print(f"Initial State: {state}")
    statistics = []
    for _ in range(0, repeats):
        start_time = time.process_time()
        goal_node, tree, frontier_len, expanded_nodes = search(state, search_method,
                                                               N, heuristic, initial_limit=initial_limit, max_depth=max_depth)
        if goal_node is not None:
            end_time = time.process_time()
            statistics.append(Statistics(
                tree, goal_node, frontier_len, expanded_nodes, end_time - start_time))
    return statistics


def benchmark_best(search_method, heuristic, heuristic_name, initial_limit=10, max_depth=10, repeats=1):
    return benchmark(best_initial_state, search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)


def benchmark_medium(search_method, heuristic, heuristic_name, initial_limit=10, max_depth=10, repeats=1):
    return benchmark(medium_initial_state, search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)


def benchmark_worst(search_method, heuristic, heuristic_name, initial_limit=10, max_depth=10, repeats=1):
    return benchmark(worst_initial_state, search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)


def benchmark_all(search_method, heuristic, heuristic_name, initial_limit=10, max_depth=10, repeats=1):
    return benchmark_best(search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats), benchmark_medium(search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats), benchmark_worst(search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)


def run_benchmark(type, search_method="", heuristic="", heuristic_name="", initial_limit=10, max_depth=10, repeats=1,test_state=None):
    if type == 0:
        best_case_stats, medium_case_stats, worst_case_stats = benchmark_all(
            search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)
    elif type == 1:
        best_case_stats = benchmark_best(
            search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)
    elif type == 2:
        medium_case_stats = benchmark_medium(
            search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)
    elif type == 3:
        worst_case_stats = benchmark_worst(
            search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)

    if best_case_stats is not None and len(best_case_stats) > 0:
        best_case_mean_stat = {"cost": statistics.mean(
            [stat.solution_path_cost for stat in best_case_stats]),
            "depth": statistics.mean([stat.solution_depth for stat in best_case_stats]),
            "frontier": statistics.mean([stat.frontier_count for stat in best_case_stats]),
            "expanded": statistics.mean([stat.expanded_count for stat in best_case_stats]),
            "execution_time": statistics.mean([stat.execution_time for stat in best_case_stats]),
            "standard_deviation": statistics.stdev([stat.execution_time for stat in best_case_stats])if repeats > 1 else '0.0', }
    else:
        best_case_mean_stat = {"result": "No solution found"}

    if medium_case_stats is not None and len(medium_case_stats) > 0:
        medium_case_mean_stat = {"cost": statistics.mean(
            [stat.solution_path_cost for stat in medium_case_stats]),
            "depth": statistics.mean([stat.solution_depth for stat in medium_case_stats]),
            "frontier": statistics.mean([stat.frontier_count for stat in medium_case_stats]),
            "expanded": statistics.mean([stat.expanded_count for stat in medium_case_stats]),
            "execution_time": statistics.mean([stat.execution_time for stat in medium_case_stats]),
            "standard_deviation": statistics.stdev([stat.execution_time for stat in medium_case_stats])if repeats > 1 else '0.0', }
    else:
        medium_case_mean_stat = {"result": "No solution found"}

    if worst_case_stats is not None and len(worst_case_stats) > 0:
        worst_case_mean_stat = {"cost": statistics.mean(
            [stat.solution_path_cost for stat in worst_case_stats]),
            "depth": statistics.mean([stat.solution_depth for stat in worst_case_stats]),
            "frontier": statistics.mean([stat.frontier_count for stat in worst_case_stats]),
            "expanded": statistics.mean([stat.expanded_count for stat in worst_case_stats]),
            "execution_time": statistics.mean([stat.execution_time for stat in worst_case_stats]),
            "standard_deviation": statistics.stdev([stat.execution_time for stat in worst_case_stats]) if repeats > 1 else '0.0', }
    else:
        worst_case_mean_stat = {"result": "No solution found"}

    stats = {
        "best_case": best_case_mean_stat,
        "medium_case": medium_case_mean_stat,
        "worst_case": worst_case_mean_stat,
        "times": [statistics.mean([stat.execution_time for stat in best_case_stats]), statistics.mean([stat.execution_time for stat in medium_case_stats]), statistics.mean([stat.execution_time for stat in worst_case_stats])],
        "stdev": [statistics.stdev([stat.execution_time for stat in best_case_stats])if repeats > 1 else '0.0', statistics.stdev([stat.execution_time for stat in medium_case_stats])if repeats > 1 else '0.0', statistics.stdev([stat.execution_time for stat in worst_case_stats]) if repeats > 1 else '0.0'],
    }

    out_file = open(
        f"benchmark_results/{search_method}{f'_{heuristic_name}' if heuristic_name != '' else ''}{f'_{initial_limit}' if initial_limit > 0 else ''}{f'_{max_depth}' if max_depth > 0 else ''}_{repeats}.json", "w")

    json.dump(stats, out_file, indent=4)


def main():
    test_all_search_methods = False

    parser = ArgumentParser()
    benchmark_group = parser.add_mutually_exclusive_group(required=True)
    benchmark_group.add_argument(
        "-a", "--all", action="store_const", const=0, dest="benchmark_type")
    benchmark_group.add_argument(
        "-b", "--best", action="store_const", const=1, dest="benchmark_type")
    benchmark_group.add_argument(
        "-m", "--medium", action="store_const", const=2, dest="benchmark_type")
    benchmark_group.add_argument(
        "-w", "--worst", action="store_const", const=3, dest="benchmark_type")
    parser.add_argument("-s", "--search", dest="search_method",
                        help="run benchmarks with selected search method", default="")
    parser.add_argument("-he", "--heuristic", dest="heuristic",
                        help="run benchmarks with selected heuristic", default="")
    parser.add_argument("-i", "--initial_limit", dest="initial_limit",
                        help="set initial limit for iterative depth search", default=0)
    parser.add_argument("-md", "--max_depth", dest="max_depth", action="store",
                        help="set max depth for depth limited search", default=0)
    parser.add_argument("-r", "--repeats", dest="repeats",
                        action="store", default=1)

    args = parser.parse_args()
    search_method = args.search_method.lower()
    heuristic = args.heuristic.lower()
    repeats = int(args.repeats)
    parsed_heuristic = None
    heuristic_name = ""
    if search_method == "":
        test_all_search_methods = True
    elif search_method in ["a_star", "lhs", "ghs"]:
        if heuristic == 'manhattan':
            parsed_heuristic = Heuristic.MANHATTAN_DISTANCE
            heuristic_name = "Manhattan Distance"
        elif heuristic == 'euclidean':
            parsed_heuristic = Heuristic.EUCLIDEAN_DISTANCE
            heuristic_name = "Euclidean Distance"
        elif heuristic == 'misplaced_tiles':
            parsed_heuristic = Heuristic.MISPLACED_TILES
            heuristic_name = "Misplaced Tiles"
        elif heuristic == 'misplaced_tiles_value':
            parsed_heuristic = Heuristic.MISPLACED_TILES_VALUE
            heuristic_name = "Misplaced Tiles Value"
        elif heuristic == 'visited_tiles_value':
            parsed_heuristic = Heuristic.VISITED_TILES_VALUE
            heuristic_name = "Visited Tiles Value"
        else:
            print("Invalid heuristic")
            exit()

    benchmark_type = args.benchmark_type
    if test_all_search_methods:
        for search_method in SearchMethod:
            run_benchmark(benchmark_type, search_method, parsed_heuristic, heuristic_name, int(
                args.initial_limit), int(args.max_depth), repeats)
    else:
        # Perform search in selected method
        run_benchmark(benchmark_type,
                      SearchMethod[search_method.upper()], parsed_heuristic, heuristic_name, int(args.initial_limit), int(args.max_depth), repeats)


if __name__ == '__main__':
    main()
