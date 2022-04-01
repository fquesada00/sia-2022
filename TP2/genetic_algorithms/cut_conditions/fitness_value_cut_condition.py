from constants import MIN_FITNESS


def fitness_value_cut_condition(population, elapsed_time, fitness_function):
    fitnesses = sorted([fitness_function(x) for x in population], reverse=True)
    return fitnesses[0] >= MIN_FITNESS
