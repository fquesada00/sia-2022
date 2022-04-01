import random


def roulette_selection(population, fitness_function, selection_size):
    selection = []
    population_copy = population.copy()

    for _ in range(selection_size):
        # Calculate the total fitness of the population
        total_fitness = sum(fitness_function(x) for x in population_copy)

        # Calculate the fitness of each individual
        fitness_list = [fitness_function(x) for x in population_copy]

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
