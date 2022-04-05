import random


def uniform_selection(population: list, fitness_function, selection_size: int,selection_parameter):

    new_population = population.copy()

    selection = []

    # Select the selection_size individuals from the population, randomly
    for _ in range(selection_size):
        random_index = random.randint(0, len(new_population) - 1)
        selection.append(new_population.pop(random_index))

    return selection