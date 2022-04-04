import genetic_algorithms


def max_generations_cut_condition(population, fitness_function, elapsed_time, cut_condition_parameters):
    return genetic_algorithms.number_of_generations == cut_condition_parameters.max_generations
