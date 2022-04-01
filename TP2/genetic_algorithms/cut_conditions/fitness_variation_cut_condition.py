
from constants import FITNESS_DISTANCE, REQUIRED_GENERATION_REPEATS


prev_best_fitness = None

repeated_generations = 0


def fitness_variation_cut_condition(population, elapsed_time, fitness_function):
    best_fitness = sorted([fitness_function(x)
                          for x in population], reverse=True)[0]

    global prev_best_fitness
    global repeated_generations
    if prev_best_fitness is None:
        prev_best_fitness = best_fitness
        return False

    if abs(prev_best_fitness - best_fitness) > FITNESS_DISTANCE:
        prev_best_fitness = best_fitness
        repeated_generations = 0
        return False

    prev_best_fitness = best_fitness
    repeated_generations += 1
    return repeated_generations >= REQUIRED_GENERATION_REPEATS
