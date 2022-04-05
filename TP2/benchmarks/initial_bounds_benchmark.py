from ..genetic_algorithms.mutation_methods import MutationMethod
from ..genetic_algorithms import generate_initial_population, optimize
from ..benchmarks.utils.get_results_data import get_results_data
from ..benchmarks.utils.read_benchmark_parameters_from_config import read_benchmark_parameters_from_config
from ..optimization_problem.functions import fitness_function
from matplotlib import pyplot as plt


def run_initial_bounds_benchmark(selection_parameters, cut_condition_parameters, crossover_parameters, mutation_parameters, population_size, bounds, output_filename):
    tmp_results_filename = 'benchmark_tmp.csv'
    plt.figure(figsize=(13, 5))

    for bound in bounds:

        initial_population = generate_initial_population(
            population_size, bound['from'], bound['to'])
        print("Bounds: {} - {}".format(bound['from'], bound['to']))

        summary = optimize(initial_population, fitness_function, selection_parameters,
                           crossover_parameters, mutation_parameters, cut_condition_parameters, output_filename=tmp_results_filename)

        print(summary)

        generation_numbers, fitness_values = get_results_data(
            tmp_results_filename)

        line, = plt.plot(generation_numbers, fitness_values)
        line.set_label(
            f"[{bound['from']}, {bound['to']}] - {round(summary.fitness,2)}")

    plt.legend()
    plt.savefig('./TP2/benchmarks/output/' + output_filename + '.png', dpi=300)


if __name__ == '__main__':
    parameters = read_benchmark_parameters_from_config()

    run_initial_bounds_benchmark(parameters.selection_parameters, parameters.cut_condition_parameters, parameters.crossover_parameters,
                                 parameters.mutation_parameters, parameters.population_size, parameters.bounds, parameters.initial_bounds_benchmark_output_filename)
