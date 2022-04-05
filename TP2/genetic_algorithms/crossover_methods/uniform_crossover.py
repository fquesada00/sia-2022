import random
from ...models.Chromosome import Chromosome


def uniform_crossover(first_parent, second_parent, genes_length,crossover_parameters):

    first_child_genes = []
    second_child_genes = []

    for i in range(genes_length):
        if random.random() <= 0.5:
            first_child_genes.append(second_parent[i])
            second_child_genes.append(first_parent[i])
        else:
            first_child_genes.append(first_parent[i])
            second_child_genes.append(second_parent[i])
    return Chromosome(first_child_genes), Chromosome(second_child_genes)
