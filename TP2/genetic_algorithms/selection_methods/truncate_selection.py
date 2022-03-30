
import random

def truncate_selection(population, fitness_function, selection_size, k):
    if (len(population) - k) < selection_size:
        raise ValueError("The truncated population size is smaller than the selection size")
    
    # Discard the less fit k individuals from the population
    sorted_population = sorted(population, key=fitness_function)
    truncated_population = sorted_population[k:]

    truncated_population_length = len(truncated_population)

    selection = []

    # Select the selection_size individuals from the truncated population, randomly
    for _ in range(selection_size):
        random_index = random.randint(0, truncated_population_length - 1)
        selection.append(truncated_population[random_index])
        truncated_population.pop(random_index)
        truncated_population_length -= 1
    
    return selection