import argparse
import json
from genetic_algorithms.crossover_methods import CrossoverMethod
from genetic_algorithms.mutation_methods import MutationMethod
from genetic_algorithms.cut_conditions import CutCondition
from genetic_algorithms.crossover_methods import CrossoverParameters
from genetic_algorithms.mutation_methods import MutationParameters
from genetic_algorithms.cut_conditions import CutConditionParameters
from genetic_algorithms import optimize
from genetic_algorithms.selection_methods import SelectionMethod
from genetic_algorithms.selection_methods import SelectionParameters
from genetic_algorithms import generate_initial_population
from optimization_problem import fitness_function
from matplotlib import pyplot as plt


def run_selection_benchmark(selection_parameters, cut_condition_parameters, crossover_parameters, mutation_parameters, population_size, min_real, max_real):
    initial_population = generate_initial_population(
        population_size, min_real, max_real)
    tmp_results_filename = 'benchmark_tmp.csv'

    for selection_method in SelectionMethod:
        selection_parameters.selection_method = selection_method

        print("Selection method: {}".format(selection_method))

        summary = optimize(tmp_results_filename, initial_population, fitness_function, selection_parameters,
                 crossover_parameters, mutation_parameters, cut_condition_parameters)

        print(summary)


        generation_numbers, fitness_values = get_results_data(
            tmp_results_filename)

        line, = plt.plot(generation_numbers, fitness_values)
        line.set_label(f"{selection_method} - {summary.fitness}")

    plt.legend()
    plt.show()


def get_results_data(results_filename):
    x_data = []
    y_data = []

    with open(results_filename) as results_file:
        for i, line in enumerate(results_file):
            if i < 18:
                continue
            line_data = line.split("\t")
            x_data.append(int(line_data[0]))
            y_data.append(float(line_data[1]))

        # print(x_data,y_data)
        return x_data, y_data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=argparse.FileType('r'), default="benchmark_config.json",
                        help="Benchmark configuration in JSON format", dest="input_file", required=False)

    parser.add_argument("--output_filename", type=str, default="benchmark_tmp.csv",
                        help="Name of the generated CSV output file", dest="output_filename", required=False)

    args = parser.parse_args()
    output_file = open(args.output_filename, "w")

    config = json.load(args.input_file)

    min_real = config["min_real"]
    max_real = config["max_real"]
    population_size = config["population_size"]


    cut_condition_config = config["cut_condition"]
    cut_condition_parameters = CutConditionParameters(cut_condition_method=CutCondition.from_str(cut_condition_config["method"]),
                                                      max_generations=cut_condition_config["max_generations"],
                                                      max_time=cut_condition_config["max_time"],
                                                      min_fitness_value=cut_condition_config["min_fitness"],
                                                      fitness_required_generations_repeats=cut_condition_config["fitness_required_generations_repeats"],
                                                      fitness_threshold=cut_condition_config["fitness_threshold"]
                                                      )

    crossover_config = config["crossover"]
    crossover_parameters = CrossoverParameters(crossover_method=CrossoverMethod.from_str(crossover_config["method"]),
                                               multiple_point_crossover_points=config[
                                                   "crossover"]["multiple_point_number"]
                                               )
    mutation_config = config["mutation"]
    mutation_parameters = MutationParameters(mutation_method=MutationMethod.from_str(mutation_config["method"]),
                                             mutation_rate=mutation_config["mutation_rate"],
                                             uniform_mutation_bound=mutation_config["uniform_mutation_bound"],
                                             normal_mutation_std=mutation_config["normal_mutation_standard_deviation"]
                                             )

    selection_config = config["selection"]
    selection_parameters = SelectionParameters(initial_temperature=selection_config["boltzmann_initial_temperature"],
                                               final_temperature=selection_config["boltzmann_final_temperature"],
                                               exp_rate=selection_config["boltzmann_decay_rate"],
                                               k=selection_config["truncate_size"],
                                               threshold=selection_config["tournament_threshold"]
                                               )

    run_selection_benchmark(selection_parameters, cut_condition_parameters, crossover_parameters,
                            mutation_parameters, population_size, min_real, max_real)
