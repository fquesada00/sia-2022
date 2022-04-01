import random


def swap_mutation(genes, mutation_rate):
    """
    Mutate an individual by swapping pairs of genes randomly.
    """
    for i in range(len(genes)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(genes) - 1)
            genes[i], genes[j] = genes[j], genes[i]

    return genes
