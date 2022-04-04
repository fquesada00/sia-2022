import random



def normal_mutation(genes, mutation_parameters):
    """
    Mutate an individual by modifying each one of its genes at random
    """
    for i in range(len(genes)):
        if random.random() < mutation_parameters.mutation_rate:
            genes[i] += random.normalvariate(0, mutation_parameters.normal_mutation_std)

    return genes
