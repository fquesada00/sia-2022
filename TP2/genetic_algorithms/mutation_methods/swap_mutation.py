import random


def uniform_mutation(individual, mutation_rate):
    """
    Mutate an individual by swapping pairs of genes randomly.
    """
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(individual) - 1)
            individual[i], individual[j] = individual[j], individual[i]

    return individual
