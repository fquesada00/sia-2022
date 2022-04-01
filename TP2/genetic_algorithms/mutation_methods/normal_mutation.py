import random
from constants import NORMAL_MUTATION_STD


def normal_mutation(genes, mutation_rate):
    """
    Mutate an individual by modifying each one of its genes at random
    """
    for i in range(len(genes)):
        if random.random() < mutation_rate:
            genes[i] += random.normalvariate(0, NORMAL_MUTATION_STD)

    return genes
