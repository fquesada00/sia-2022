from ..genetic_algorithms.cut_conditions import CutCondition
from ..genetic_algorithms import generate_initial_population, optimize
from ..benchmarks.utils.get_results_data import get_results_data
from ..benchmarks.utils.read_benchmark_parameters_from_config import read_benchmark_parameters_from_config
from ..optimization_problem.functions import fitness_function


def run_cut_condition_benchmark(selection_parameters, cut_condition_parameters, crossover_parameters, mutation_parameters, population_size, min_real, max_real, output_filename):
    initial_population = generate_initial_population(
        population_size, min_real, max_real)
    tmp_results_filename = 'benchmark_tmp.csv'

    with open('./TP2/benchmarks/output/' + output_filename + '.csv', 'w') as f:
        f.write("cut_condition,fitness\n")

        for cut_condition in CutCondition:
            cut_condition_parameters.cut_condition_method = cut_condition

            print("Cut condition: {}".format(cut_condition))

            summary = optimize(initial_population, fitness_function, selection_parameters,
                               crossover_parameters, mutation_parameters, cut_condition_parameters, output_filename=tmp_results_filename)

            print(summary)

            f.write(f"{cut_condition},{round(summary.fitness, 4)}\n")

            # print method - fitness value to csv file


if __name__ == '__main__':

    parameters = read_benchmark_parameters_from_config()

    run_cut_condition_benchmark(parameters.selection_parameters, parameters.cut_condition_parameters, parameters.crossover_parameters,
                                parameters.mutation_parameters, parameters.population_size, parameters.min_real, parameters.max_real, parameters.cut_condition_benchmark_output_filename)
