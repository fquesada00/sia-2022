import random

from TP2.models.Chromosome import Chromosome

def simple_crossover(first_parent, second_parent, genes_length):

    random_index = random.randint(1, genes_length - 2)

    first_parent_genes = first_parent.get_genes()
    second_parent_genes = second_parent.get_genes()

    first_child_genes = second_parent_genes[:random_index] + first_parent_genes[random_index:]
    second_child_genes = first_parent_genes[:random_index] + second_parent_genes[random_index:]

    return Chromosome(first_child_genes), Chromosome(second_child_genes)   