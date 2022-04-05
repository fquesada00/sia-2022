import argparse
import json
from ...benchmarks import BenchmarkParameters
from ...models.Parameters import Parameters
from ...genetic_algorithms.crossover_methods import CrossoverMethod, CrossoverParameters
from ...genetic_algorithms.cut_conditions import CutCondition, CutConditionParameters
from ...genetic_algorithms.mutation_methods import MutationMethod, MutationParameters
from ...genetic_algorithms.selection_methods import SelectionMethod, SelectionParameters


def read_benchmark_parameters_from_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=argparse.FileType('r'), default="TP2/benchmark_config.json",
                        help="Benchmark configuration in JSON format", dest="input_file", required=False)

    args = parser.parse_args()

    config = json.load(args.input_file)

    min_real = config["min_real"]
    max_real = config["max_real"]
    population_size = config["population_size"]

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

    bounds_config = config["initial_bounds"]

    return BenchmarkParameters(selection_parameters=selection_parameters,
                      crossover_parameters=crossover_parameters,
                      mutation_parameters=mutation_parameters,
                      cut_condition_parameters=cut_condition_parameters,
                        population_size=population_size,
                        min_real=min_real,
                        max_real=max_real,
                        bounds= bounds_config["input"],
                        selection_output_filename=selection_config["benchmark_output_filename"],
                        crossover_output_filename=crossover_config["benchmark_output_filename"],
                        mutation_output_filename=mutation_config["benchmark_output_filename"],
                        cut_condition_output_filename=cut_condition_config["benchmark_output_filename"],
                        initial_bounds_output_filename=bounds_config["benchmark_output_filename"])
                        
