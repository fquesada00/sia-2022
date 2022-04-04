

def fitness_value_cut_condition(population, fitness_function, elapsed_time, cut_condition_parameters):
    fitnesses = sorted([fitness_function(x) for x in population], reverse=True)
    return fitnesses[0] >= cut_condition_parameters.min_fitness_value
