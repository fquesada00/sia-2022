import random

from TP2.models.Chromosome import Chromosome

def simple_crossover(first_parent, second_parent, genes_length):

    random_index = random.randint(1, genes_length - 2)

    first_child_genes = second_parent[:random_index] + first_parent[random_index:]
    second_child_genes = first_parent[:random_index] + second_parent[random_index:]

    return Chromosome(first_child_genes), Chromosome(second_child_genes)   