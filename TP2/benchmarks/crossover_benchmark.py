
from ..models import Summary
from ..genetic_algorithms.crossover_methods import CrossoverMethod
from ..genetic_algorithms import generate_initial_population, optimize
from ..benchmarks.utils.get_results_data import get_results_data
from ..benchmarks.utils.read_benchmark_parameters_from_config import read_benchmark_parameters_from_config
from ..optimization_problem.functions import fitness_function
from matplotlib import pyplot as plt


def run_crossover_benchmark(selection_parameters, cut_condition_parameters, crossover_parameters, mutation_parameters, population_size, min_real, max_real, output_filename, multiple_points):
    initial_population = generate_initial_population(
        population_size, min_real, max_real)
    tmp_results_filename = './TP2/benchmarks/benchmark_tmp.csv'
    plt.figure(figsize=(10, 5))

    summary_file = open('./TP2/benchmarks/output/' +
                        output_filename + '.csv', 'w')
    summary_file.write(f"crossover_method,{Summary.csv_header()}")

    for crossover_method in CrossoverMethod:
        crossover_parameters.crossover_method = crossover_method

        print("Crossover method: {}".format(crossover_method))

        if crossover_method == CrossoverMethod.MULTIPLE_POINT:
            for n_points in multiple_points:
                print("Multiple points: {}".format(n_points))
                crossover_parameters.multiple_point_crossover_points = n_points
                summary = optimize(initial_population, fitness_function, selection_parameters,
                                   crossover_parameters, mutation_parameters, cut_condition_parameters, output_filename=tmp_results_filename)
                generation_numbers, fitness_values = get_results_data(
                    tmp_results_filename)
                line, = plt.plot(generation_numbers, fitness_values)
                line.set_label(
                    f"{crossover_method} {n_points} - {round(summary.fitness,3)}")
                summary_file.write(
                    f"{crossover_method},{summary.to_csv()}")
        else:
            summary = optimize(initial_population, fitness_function, selection_parameters,
                               crossover_parameters, mutation_parameters, cut_condition_parameters, output_filename=tmp_results_filename)

            summary_file.write(
                f"{crossover_method},{summary.to_csv()}")

            generation_numbers, fitness_values = get_results_data(
                tmp_results_filename)

            line, = plt.plot(generation_numbers, fitness_values)
            line.set_label(f"{crossover_method} - {round(summary.fitness,3)}")

    plt.legend()
    plt.xlabel('Número de generación')
    plt.ylabel('Fitness máximo de generación')
    plt.savefig('./TP2/benchmarks/output/' + output_filename + '.png', dpi=300)

    summary_file.close()


if __name__ == '__main__':

    parameters = read_benchmark_parameters_from_config()

    run_crossover_benchmark(parameters.selection_parameters, parameters.cut_condition_parameters, parameters.crossover_parameters,
                            parameters.mutation_parameters, parameters.population_size, parameters.min_real, parameters.max_real, parameters.crossover_benchmark_output_filename,
                            parameters.multiple_points)
