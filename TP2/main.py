import random
import argparse
import json
from .models import Parameters
from .genetic_algorithms.cut_conditions import CutCondition
from .genetic_algorithms.selection_methods import SelectionMethod
from .optimization_problem.functions import fitness_function
from .genetic_algorithms import optimize, generate_initial_population
from .genetic_algorithms.crossover_methods import CrossoverParameters
from .genetic_algorithms.crossover_methods import CrossoverMethod
from .genetic_algorithms.cut_conditions import CutConditionParameters
from .genetic_algorithms.mutation_methods import MutationParameters
from .genetic_algorithms.mutation_methods import MutationMethod
from .genetic_algorithms.selection_methods import SelectionParameters
from .constants import MAX_REAL, MIN_REAL, POPULATION_SIZE

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=argparse.FileType('r'), default="TP2/main_config.json",
                        help="Main program configuration in JSON format", dest="input_file", required=False)

    args = parser.parse_args()

    config = json.load(args.input_file)
    min_real = config["min_real"]
    max_real = config["max_real"]
    population_size = config["population_size"]
    output_filename = config["output_filename"]

    cut_condition_config = config["cut_condition"]
    crossover_config = config["crossover"]
    mutation_config = config["mutation"]
    selection_config = config["selection"]

    cut_condition_parameters = CutConditionParameters(cut_condition_method=CutCondition.from_str(cut_condition_config["method"]),
                                                      max_generations=cut_condition_config["max_generations"],
                                                      max_time=cut_condition_config["max_time"],
                                                      min_fitness_value=cut_condition_config["min_fitness"],
                                                      fitness_required_generations_repeats=cut_condition_config[
                                                          "fitness_required_generations_repeats"]
                                                      )

    crossover_parameters = CrossoverParameters(crossover_method=CrossoverMethod.from_str(crossover_config["method"]),
                                               multiple_point_crossover_points=config[
                                                   "crossover"]["multiple_point_number"]
                                               )
    mutation_parameters = MutationParameters(mutation_method=MutationMethod.from_str(mutation_config["method"]),
                                             mutation_rate=mutation_config["mutation_rate"],
                                             uniform_mutation_bound=mutation_config["uniform_mutation_bound"],
                                             normal_mutation_std=mutation_config["normal_mutation_standard_deviation"]
                                             )

    selection_parameters = SelectionParameters(selection_method=SelectionMethod.from_str(selection_config["method"]),
                                               initial_temperature=selection_config["boltzmann_initial_temperature"],
                                               final_temperature=selection_config["boltzmann_final_temperature"],
                                               exp_rate=selection_config["boltzmann_decay_rate"],
                                               k=selection_config["truncate_size"],
                                               threshold=selection_config["tournament_threshold"],
                                               )

    parameters = Parameters(selection_parameters, crossover_parameters,
                            mutation_parameters, cut_condition_parameters)

    initial_population = generate_initial_population(
        config['population_size'], config['min_real'], config['max_real'])

    summary = optimize(initial_population,
                       fitness_function, selection_parameters=selection_parameters, crossover_parameters=crossover_parameters, mutation_parameters=mutation_parameters, cut_condition_parameters=cut_condition_parameters)
    print(parameters)
    print(summary)

    with open('./TP2/' + output_filename + '.csv', 'w') as f:
        f.write(
            'execution_time,fitness,W_1,W_2,W_3,w_1,w_2,w_3,w_4,w_5,w_6,w_0_1,w_0_2,error,F_1,F_2,F_3\n')
        f.write(summary.to_csv())
