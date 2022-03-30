
import random


def uniform_selection(population, fitness_function, selection_size):
    population_length = len(population)

    if population_length < selection_size:
        raise ValueError("The population size is smaller than the selection size")
    
    selection = []

    # Select the selection_size individuals from the population, randomly
    for _ in range(selection_size):
        random_index = random.randint(0, population_length - 1)
        selection.append(population[random_index])
        population.pop(random_index)
        population_length -= 1
    
    return selection
