
import random

def tournament_selection(population, fitness_function, selection_size):

    selection = []
    threshold = random.random()
    population_length = len(population)
    number_of_initial_couples = 2

    for _ in range(selection_size):

        # TODO: make it generic for n couples
        # couples = []
        # for _ in range(number_of_initial_couples):
        #     random_index = random.randint(0, len(population) - 1)
        #     couples.append(population[random_index])


        # Obtain two random couple of individuals from the population and apply tournament algorithm
        first_individual_index = random.choice(range(len(population)))
        population.pop(first_individual_index)

        second_individual_index = random.choice(range(len(population)))
        population.pop(second_individual_index)

        third_individual_index = random.choice(range(len(population)))
        population.pop(third_individual_index)

        fourth_individual_index = random.choice(range(len(population)))
        population.pop(fourth_individual_index)

        first_couple = [population[first_individual_index], population[second_individual_index]]
        first_couple_winner_index, first_couple_loser_index = tournament(first_couple, fitness_function, threshold)

        second_couple = [population[third_individual_index], population[fourth_individual_index]]
        second_couple_winner_index, second_couple_loser_index = tournament(second_couple, fitness_function, threshold)

        # Make another couple from the winners of both tournaments
        third_couple = [first_couple[first_couple_winner_index], second_couple[second_couple_winner_index]]
        third_couple_winner_index, third_couple_loser_index = tournament(third_couple, fitness_function, threshold)

        # Add the winner of the third couple to the selection
        selection.append(third_couple[third_couple_winner_index])

        # Add the losers of all couples to the population
        population.extend([first_couple[first_couple_loser_index], second_couple[second_couple_loser_index], third_couple[third_couple_loser_index]])

    return selection

def tournament(couple, fitness_function, threshold):
    first_individual = couple[0]
    second_individual = couple[1]

    first_fitness = fitness_function(first_individual)
    second_fitness = fitness_function(second_individual)

    random_number = random.random()
    most_fit_individual_index = 0
    least_fit_individual_index = 1

    if first_fitness < second_fitness:
        most_fit_individual_index = 1
        least_fit_individual_index = 0

    # If random_number is lower than threshold, the most fit individual is selected
    if random_number < threshold:
        return most_fit_individual_index, least_fit_individual_index
    else:
        return least_fit_individual_index, most_fit_individual_index
