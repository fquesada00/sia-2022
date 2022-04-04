import random
from constants import MAX_REAL, MIN_REAL


def single_mutation(genes, mutation_parameters):
    """
    Mutate an individual by modifying one of its genes at random
    """

    if random.random() < mutation_parameters.mutation_rate:
        i = random.randint(0, len(genes) - 1)
        genes[i] = random.uniform(MIN_REAL, MAX_REAL)

    return genes
