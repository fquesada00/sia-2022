import random



def swap_mutation(genes, mutation_parameters):
    """
    Mutate an individual by swapping pairs of genes randomly.
    """
    for i in range(len(genes)):
        if random.random() < mutation_parameters.mutation_rate:
            j = i
            while j == i:
                j = random.randint(0, len(genes) - 1)
            genes[i], genes[j] = genes[j], genes[i]

    return genes
