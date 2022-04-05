from ..models import Summary
from ..benchmarks.utils.read_benchmark_parameters_from_config import read_benchmark_parameters_from_config
from ..benchmarks.utils.get_results_data import get_results_data
from ..genetic_algorithms.crossover_methods import CrossoverMethod
from ..genetic_algorithms.mutation_methods import MutationMethod
from ..genetic_algorithms.cut_conditions import CutCondition
from ..genetic_algorithms.crossover_methods import CrossoverParameters
from ..genetic_algorithms.mutation_methods import MutationParameters
from ..genetic_algorithms.cut_conditions import CutConditionParameters
from ..genetic_algorithms import optimize
from ..genetic_algorithms.selection_methods import SelectionMethod
from ..genetic_algorithms.selection_methods import SelectionParameters
from ..genetic_algorithms import generate_initial_population
from ..optimization_problem import fitness_function
from matplotlib import pyplot as plt


def run_selection_benchmark(selection_parameters, cut_condition_parameters, crossover_parameters, mutation_parameters, population_size, min_real, max_real, output_filename):
    initial_population = generate_initial_population(
        population_size, min_real, max_real)
    tmp_results_filename = 'benchmark_tmp.csv'

    plt.figure(figsize=(10, 5))

    summary_file = open('./TP2/benchmarks/output/' +
                        output_filename + '.csv', 'w')
    summary_file.write(f"selection_method,{Summary.csv_header()}")

    for selection_method in SelectionMethod:
        selection_parameters.selection_method = selection_method

        print("Selection method: {}".format(selection_method))

        summary = optimize(initial_population, fitness_function, selection_parameters,
                           crossover_parameters, mutation_parameters, cut_condition_parameters, output_filename=tmp_results_filename)

        summary_file.write(
            f"{selection_method},{summary.to_csv()}")

        generation_numbers, fitness_values = get_results_data(
            tmp_results_filename)

        line, = plt.plot(generation_numbers, fitness_values)
        line.set_label(f"{selection_method} - {round(summary.fitness, 3)}")

    plt.legend()
    plt.xlabel('Número de generación')
    plt.ylabel('Fitness máximo de generación')
    plt.savefig('./TP2/benchmarks/output/' + output_filename + '.png', dpi=300)

    summary_file.close()


if __name__ == '__main__':

    parameters = read_benchmark_parameters_from_config()

    run_selection_benchmark(parameters.selection_parameters, parameters.cut_condition_parameters, parameters.crossover_parameters,
                            parameters.mutation_parameters, parameters.population_size, parameters.min_real, parameters.max_real, parameters.selection_benchmark_output_filename)
