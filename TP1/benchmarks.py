from argparse import ArgumentParser
import statistics
import time
from search_methods.SearchMethod import SearchMethod
from search_methods.heuristics.Heuristic import Heuristic
from search_methods.methods.search import search
from matplotlib import pyplot as plt
import numpy as np

best_initial_state = [1, 2, 3, 4, 5, 6, 7, 0, 8]
medium_initial_state = [0, 1, 2, 4, 5, 3, 7, 8, 6]
worst_initial_state = [8, 6, 7, 2, 5, 4, 3, 0, 1]
N = 3


def benchmark(state, search_method, heuristic, heuristic_name, initial_limit=10, max_depth=10, repeats=1):
    print(f"Benchmarking {search_method} {f'with initial limit {initial_limit}' if initial_limit > 0 else ''} {f'with heuristic {heuristic_name}' if heuristic_name != '' else ''} {f'with max depth {max_depth}' if max_depth > 0 else ''}")
    print(f"Initial State: {state}")
    times = []
    for _ in range(0, repeats):
        start_time = time.process_time()
        search(state, search_method, N, heuristic, initial_limit, max_depth)
        end_time = time.process_time()
        times.append(end_time-start_time)
    return times


def benchmark_best(search_method, heuristic, heuristic_name, initial_limit=10, max_depth=10, repeats=1):
    return benchmark(best_initial_state, search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)


def benchmark_medium(search_method, heuristic, heuristic_name, initial_limit=10, max_depth=10, repeats=1):
    return benchmark(medium_initial_state, search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)


def benchmark_worst(search_method, heuristic, heuristic_name, initial_limit=10, max_depth=10, repeats=1):
    return benchmark(worst_initial_state, search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)


def benchmark_all(search_method, heuristic, heuristic_name, initial_limit=10, max_depth=10, repeats=1):
    return benchmark_best(search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats), benchmark_medium(search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats), benchmark_worst(search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)


def run_benchmark(type, search_method="", heuristic="", heuristic_name="", initial_limit=10, max_depth=10, repeats=1):
    if type == 0:
        best_times, medium_times, worst_times = benchmark_all(
            search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)
    elif type == 1:
        best_times = benchmark_best(
            search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)
    elif type == 2:
        medium_times = benchmark_medium(
            search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)
    elif type == 3:
        worst_times = benchmark_worst(
            search_method, heuristic, heuristic_name, initial_limit, max_depth, repeats)

    print(f"Best times: {statistics.mean(best_times) if best_times is not None else ''} {f'standard deviation: {statistics.stdev(best_times)}' if best_times is not None and repeats > 1 else ''}" if best_times is not None else '')
    print(f"Medium times: {statistics.mean(medium_times) if medium_times is not None else ''} { f'standard deviation: {statistics.stdev(medium_times)}' if medium_times is not None and repeats > 1 else ''}"if medium_times is not None else '')
    print(f"Worst times: {statistics.mean(worst_times) if worst_times is not None else ''} {f'standard deviation: {statistics.stdev(worst_times)}' if worst_times is not None and repeats > 1 else ''}" if worst_times is not None else '')
    # labels = ['G1']
    # x = np.arange(len(labels))
    # fig, ax = plt.subplots()
    # width = 0.35
    # rects1 = ax.bar(x - width, best_times,width, color='b', label='Best Case')
    # rects2 = ax.bar(x, medium_times,width, color='g', label='Medium Case')
    # rects3 = ax.bar(x + width, worst_times,width, color='r', label='Worst Case')
    # ax.set_ylabel('Time (s)')
    # ax.set_title("Benchmarking " + str(search_method) + " " + heuristic_name)
    # ax.set_xticks(x)
    # ax.legend()

    # ax.bar_label(rects1, padding=3)
    # ax.bar_label(rects2, padding=3)
    # ax.bar_label(rects3, padding=3)

    # fig.tight_layout()
    # plt.show()


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
    parser.add_argument("-d", "--max_depth", dest="max_depth",
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
