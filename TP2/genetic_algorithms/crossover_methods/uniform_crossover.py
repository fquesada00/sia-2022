import random

from TP2.models.Chromosome import Chromosome

def uniform_crossover(first_parent, second_parent, genes_length):


    first_parent_genes = first_parent.get_genes()
    second_parent_genes = second_parent.get_genes()

    first_child_genes = []
    second_child_genes = []

    for i in range(genes_length):
        if random.random() <= 0.5:
            first_child_genes.append(second_parent_genes[i])
            second_child_genes.append(first_parent_genes[i])
        else:
            first_child_genes.append(first_parent_genes[i])
            second_child_genes.append(second_parent_genes[i])

    return Chromosome(first_child_genes), Chromosome(second_child_genes)   