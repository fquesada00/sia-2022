import random

from TP2.constants import MAX_REAL, MIN_REAL


def uniform_mutation(individual, mutation_rate):
    """
    Mutate an individual by modifying each one of its genes at random
    """
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.uniform(MIN_REAL, MAX_REAL)

    return individual
