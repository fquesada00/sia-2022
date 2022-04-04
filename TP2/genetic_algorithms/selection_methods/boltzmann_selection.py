import math
from constants import EXP_RATE, FINAL_TEMPERATURE, INITIAL_TEMPERATURE
from genetic_algorithms.selection_methods.roulette_selection import roulette_selection
import genetic_algorithms


def temperature_function(number_of_generations, initial_temperature, final_temperature, exp_rate):
    return final_temperature + (initial_temperature - final_temperature) * math.exp(-exp_rate * number_of_generations)


def get_pseudo_fitness_function(fitness_function, temperature, total_pseudo_fitness):
    return lambda i: math.exp(fitness_function(i) / temperature) / total_pseudo_fitness


def boltzmann_selection(population, fitness_function, selection_size):
    # Calculate the total pseudo fitness of the population
    temperature = temperature_function(
        genetic_algorithms.number_of_generations, INITIAL_TEMPERATURE, FINAL_TEMPERATURE, EXP_RATE)

    total_fitness = sum(math.exp(fitness_function(j) / temperature)
                        for j in population)

    return roulette_selection(population, get_pseudo_fitness_function(fitness_function, temperature, total_fitness), selection_size)
