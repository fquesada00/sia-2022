from constants import MAX_GENERATIONS
import genetic_algorithms


def max_generations_cut_condition(population, elapsed_time, fitness_function):
    return genetic_algorithms.number_of_generations == MAX_GENERATIONS
