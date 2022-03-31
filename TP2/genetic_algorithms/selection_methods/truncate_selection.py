
import random

def truncate_selection(population, fitness_function, selection_size):
    
    # Discard the less fit individuals from the population
    sorted_population = sorted(population, key=fitness_function, reverse=True)
    return sorted_population[:selection_size]
