import random

from TP2.constants import MAX_REAL, MIN_REAL


def single_mutation(individual, mutation_rate):
    """
    Mutate an individual by modifying one of its genes at random
    """

    if random.random() < mutation_rate:
        i = random.randint(0, len(individual) - 1)
        individual[i] = random.uniform(MIN_REAL, MAX_REAL)

    return individual
