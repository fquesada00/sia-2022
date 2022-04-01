from constants import MAX_REAL, MIN_REAL, MUTATION_RATE
from genetic_algorithms.cut_conditions import max_generations_cut_condition, population_variation_cut_condition, fitness_variation_cut_condition
from genetic_algorithms import optimize
from genetic_algorithms.mutation_methods import uniform_mutation
from optimization_problem.functions import fitness_function
from genetic_algorithms.selection_methods import roulette_selection
from genetic_algorithms.crossover_methods import multiple_point_crossover

if __name__ == '__main__':
    optimize(10,
             fitness_function, roulette_selection, multiple_point_crossover, uniform_mutation, MUTATION_RATE, fitness_variation_cut_condition, MIN_REAL, MAX_REAL)
