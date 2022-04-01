import numpy

from constants import CHROMOSOME_DISTANCE, REPEATED_STRUCTURE_PERCENTAGE, REQUIRED_GENERATION_REPEATS

prev_population = None

repeated_generations = 0


def population_variation_cut_condition(population, elapsed_time, fitness_function):
    sorted_population = sorted(population, key=fitness_function, reverse=True)

    global prev_population
    global repeated_generations
    if prev_population is None:
        prev_population = sorted_population
        return False
    v1_list = [numpy.array(x[::]) for x in prev_population]
    v2_list = [numpy.array(x[::]) for x in sorted_population]

    prev_population = sorted_population

    repeated_chromosomes = 0

    for v1, v2 in zip(v1_list, v2_list):
        if numpy.linalg.norm(v1-v2) <= CHROMOSOME_DISTANCE:
            repeated_chromosomes += 1
    if repeated_chromosomes >= REPEATED_STRUCTURE_PERCENTAGE:
        repeated_generations = 0
        return False
    repeated_generations += 1
    return repeated_generations >= REQUIRED_GENERATION_REPEATS


# [[1,2,3],[2,3,4]]
# [[1,1,1],[1,2,3]]
