import random


def pseudo_fitness_function(i, population_size, rank):
    return (population_size - rank[tuple(i)]) / population_size


def calculate_rank_dict(population, fitness_function):
    rank_dict = {}

    sorted_population = sorted(population, key=fitness_function, reverse=True)

    # we rank from 0 to p-1
    for index, i in enumerate(sorted_population):
        rank_dict[tuple(i)] = index

    return rank_dict


def rank_selection(population, fitness_function, selection_size,selection_parameter):
    selection = []
    population_copy = population.copy()
    for _ in range(selection_size):
        # Create rank structure
        rank = calculate_rank_dict(population, fitness_function)

        # Calculate the total fitness of the population
        total_fitness = sum(pseudo_fitness_function(
            x, len(population), rank) for x in population)

        # Calculate the fitness of each individual
        fitness_list = [pseudo_fitness_function(
            x, len(population), rank) for x in population]

        # Calculate the probability of each individual
        probability_list = [
            fitness / total_fitness for fitness in fitness_list]

        # Calculate the cumulative probability of each individual
        cumulative_probability_list = [
            sum(probability_list[:i + 1]) for i in range(len(probability_list))]

        # Generate a random number between 0 and 1
        random_number = random.random()

        # Find the index of the individual with the corresponding probability
        for i in range(len(cumulative_probability_list)):
            if cumulative_probability_list[i] > random_number:
                selection.append(population_copy.pop(i))
                break

    return selection
