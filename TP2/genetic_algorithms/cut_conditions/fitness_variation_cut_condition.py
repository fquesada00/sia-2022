

prev_best_fitness = None

repeated_generations = 0


def fitness_variation_cut_condition(population, fitness_function, elapsed_time,cut_condition_parameters):
    best_fitness = sorted([fitness_function(x)
                          for x in population], reverse=True)[0]

    global prev_best_fitness
    global repeated_generations
    if prev_best_fitness is None:
        prev_best_fitness = best_fitness
        return False

    if abs(prev_best_fitness - best_fitness) > cut_condition_parameters.fitness_threshold:
        prev_best_fitness = best_fitness
        repeated_generations = 0
        return False

    prev_best_fitness = best_fitness
    repeated_generations += 1
    return repeated_generations >= cut_condition_parameters.fitness_required_generations_repeats
