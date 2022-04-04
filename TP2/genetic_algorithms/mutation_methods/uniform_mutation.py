import random



def uniform_mutation(genes, mutation_parameters):
    """
    Mutate an individual by modifying each one of its genes at random
    """
    for i in range(len(genes)):
        if random.random() < mutation_parameters.mutation_rate:
            genes[i] += random.uniform(-mutation_parameters.uniform_mutation_bound,
                                       mutation_parameters.uniform_mutation_bound)

    return genes
