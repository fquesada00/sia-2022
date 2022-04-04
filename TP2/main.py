import random
from models import Parameters
from genetic_algorithms.cut_conditions import CutCondition
from genetic_algorithms.selection_methods import SelectionMethod
from optimization_problem.functions import fitness_function
from genetic_algorithms import optimize, generate_initial_population
from genetic_algorithms.crossover_methods import CrossoverParameters
from genetic_algorithms.cut_conditions import CutConditionParameters
from genetic_algorithms.mutation_methods import MutationParameters
from genetic_algorithms.selection_methods import SelectionParameters
from constants import MAX_REAL, MIN_REAL, POPULATION_SIZE

if __name__ == '__main__':
    selection_parameters = SelectionParameters(
        selection_method=SelectionMethod.ELITE)
    crossover_parameters = CrossoverParameters()
    mutation_parameters = MutationParameters()
    cut_condition_parameters = CutConditionParameters(
        cut_condition_method=CutCondition.MAX_GENERATIONS)

    parameters = Parameters(selection_parameters, crossover_parameters, mutation_parameters, cut_condition_parameters)
    
    initial_population = generate_initial_population(POPULATION_SIZE, MIN_REAL, MAX_REAL)

    summary = optimize("test.txt", initial_population,
                       fitness_function, selection_parameters=selection_parameters, crossover_parameters=crossover_parameters, mutation_parameters=mutation_parameters, cut_condition_parameters=cut_condition_parameters)
    print(parameters)
    print(summary)

