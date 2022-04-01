import random
from constants import UNIFORM_MUTATION_BOUND


def uniform_mutation(genes, mutation_rate):
    """
    Mutate an individual by modifying each one of its genes at random
    """
    for i in range(len(genes)):
        if random.random() < mutation_rate:
            genes[i] += random.uniform(-UNIFORM_MUTATION_BOUND,
                                       UNIFORM_MUTATION_BOUND)

    return genes
