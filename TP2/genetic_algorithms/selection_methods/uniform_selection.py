
import random


def uniform_selection(population, fitness_function, selection_size):
    
    selection = []

    # Select the selection_size individuals from the population, randomly
    for _ in range(selection_size):
        random_index = random.randint(0, len(population) - 1)
        selection.append(population.pop(random_index))
    
    return selection
