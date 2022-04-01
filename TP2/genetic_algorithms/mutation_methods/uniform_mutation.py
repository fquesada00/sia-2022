import random

from TP2.constants import MAX_REAL, MIN_REAL


def uniform_mutation(genes, mutation_rate):
    """
    Mutate an individual by modifying each one of its genes at random
    """
    for i in range(len(genes)):
        if random.random() < mutation_rate:
            genes[i] = random.uniform(MIN_REAL, MAX_REAL)

    return genes
